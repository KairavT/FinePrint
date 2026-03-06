"""
AI Terms & Conditions Risk Scanner - FastAPI Backend
"""

import io
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pdfplumber
from openai import OpenAI

app = FastAPI(title="T&C Risk Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_client():
    """Lazy-init OpenAI client so the app boots even if key isn't set yet."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WORDS = 20_000
CATEGORIES = [
    "Data Collection", "Data Sharing", "Payments & Subscriptions",
    "Arbitration", "Liability", "Termination", "Content Ownership",
    "Modifications to Terms", "Other",
]

RISK_KEYWORDS: dict[str, list[str]] = {
    "high": [
        r"mandatory\s+arbitration", r"binding\s+arbitration",
        r"waive\s+(your\s+)?right\s+to\s+(a\s+)?jury",
        r"class\s+action\s+waiver", r"indemnif(y|ication)",
        r"irrevocable\s+license", r"perpetual\s+license",
        r"sole\s+discretion.{0,40}(terminat|suspend|modif)",
        r"without\s+(prior\s+)?notice.{0,30}(terminat|chang|modif)",
        r"unlimited\s+licen[cs]e", r"waive\s+all\s+(claims|liability)",
    ],
    "medium": [
        r"auto(matic(ally)?)?[\s-]?renew", r"recurring\s+(charge|billing|payment)",
        r"third[\s-]?part(y|ies).{0,30}shar(e|ing)",
        r"collect.{0,30}(personal|location|biometric|health)\s+(data|info)",
        r"no\s+refund", r"non[\s-]?refundable",
        r"limit(ation)?\s+of\s+liability", r"as[\s-]?is",
        r"without\s+warrant(y|ies)", r"unilateral(ly)?",
        r"reserve\s+the\s+right\s+to\s+(change|modify|update|amend)",
        r"at\s+any\s+time\s+without\s+notice",
    ],
    "low": [
        r"cookies?", r"analytics", r"aggregate(d)?\s+data",
        r"opt[\s-]?out", r"unsubscribe",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def segment_clauses(text: str, max_tokens: int = 800) -> list[dict]:
    """Split document into logical clause chunks."""
    heading_pattern = re.compile(
        r"^(?:"
        r"\d+[\.\)]\s+[A-Z]|"                  # 1. HEADING or 1) Heading
        r"[A-Z][A-Z\s]{4,}$|"                   # ALL CAPS HEADING
        r"(?:Section|Article|Clause)\s+\d|"      # Section 1, Article 2
        r"[IVXLC]+\.\s+[A-Z]"                   # Roman numeral headings
        r")",
        re.MULTILINE,
    )

    # Split on headings first
    splits = heading_pattern.split(text)
    heading_matches = heading_pattern.findall(text)

    raw_sections: list[str] = []
    for i, section in enumerate(splits):
        if i > 0 and i - 1 < len(heading_matches):
            section = heading_matches[i - 1] + section
        section = section.strip()
        if section:
            raw_sections.append(section)

    if not raw_sections:
        # Fallback: split on double newlines / paragraph breaks
        raw_sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]

    # Re-chunk by approximate token length
    clauses: list[dict] = []
    current_chunk = ""
    for section in raw_sections:
        words = section.split()
        if len(current_chunk.split()) + len(words) > max_tokens and current_chunk:
            clauses.append({
                "index": len(clauses),
                "text": current_chunk.strip(),
            })
            current_chunk = section
        else:
            current_chunk += "\n\n" + section if current_chunk else section

    if current_chunk.strip():
        clauses.append({
            "index": len(clauses),
            "text": current_chunk.strip(),
        })

    return clauses


def rule_based_risk(text: str) -> dict:
    """Apply regex rule-based risk detection."""
    lower = text.lower()
    signals: list[dict] = []
    max_level = "low"

    for level in ["high", "medium", "low"]:
        for pattern in RISK_KEYWORDS[level]:
            matches = re.findall(pattern, lower)
            if matches:
                signals.append({"level": level, "pattern": pattern, "count": len(matches)})
                if level == "high":
                    max_level = "high"
                elif level == "medium" and max_level != "high":
                    max_level = "medium"

    return {"max_level": max_level, "signals": signals}


def compute_risk_score(clauses: list[dict]) -> dict:
    """Compute document-level risk score 0-10."""
    high_count = sum(1 for c in clauses if c.get("risk_level") == "high")
    medium_count = sum(1 for c in clauses if c.get("risk_level") == "medium")
    total = len(clauses) or 1

    # Weighted formula
    raw = (high_count * 3.0 + medium_count * 1.5) / total * 10
    score = min(round(raw, 1), 10.0)

    # Boost for critical types
    critical_categories = {"Arbitration", "Liability", "Data Sharing"}
    has_critical_high = any(
        c.get("risk_level") == "high" and c.get("category") in critical_categories
        for c in clauses
    )
    if has_critical_high and score < 7:
        score = max(score, 5.0)

    factors: list[str] = []
    if high_count:
        factors.append(f"{high_count} high-risk clause(s) detected")
    if medium_count:
        factors.append(f"{medium_count} medium-risk clause(s) detected")
    if has_critical_high:
        factors.append("Critical categories (Arbitration/Liability/Data Sharing) flagged")
    if not factors:
        factors.append("No significant risk signals detected")

    return {"score": score, "factors": factors}


# ---------------------------------------------------------------------------
# AI Integration
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are a legal document analyst. You analyze Terms & Conditions clauses.

For each clause, return a JSON object with these exact fields:
- "category": one of: Data Collection, Data Sharing, Payments & Subscriptions, Arbitration, Liability, Termination, Content Ownership, Modifications to Terms, Other
- "explanation": 1-3 sentence plain-English explanation of what this clause means for the user
- "risk_level": "low", "medium", or "high"
- "risk_reason": brief reason for the risk level, or null if low risk

Respond ONLY with valid JSON. No markdown, no backticks, no extra text."""

SUMMARY_SYSTEM_PROMPT = """You are a legal document analyst. Given a set of analyzed clauses from a Terms & Conditions document, generate a structured summary.

Return ONLY valid JSON with these fields:
- "overview": array of 5-10 bullet point strings summarizing the document
- "most_concerning": array of 2-5 strings describing the most concerning clauses
- "key_obligations": array of 3-6 strings describing key user obligations
- "document_type": brief description of what kind of service/product these terms cover

No markdown, no backticks, no extra text. Return ONLY the JSON object."""


async def analyze_clause_with_ai(clause_text: str) -> dict:
    """Send clause to OpenAI for classification."""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this Terms & Conditions clause:\n\n{clause_text[:3000]}"},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        # Validate category
        if result.get("category") not in CATEGORIES:
            result["category"] = "Other"
        if result.get("risk_level") not in ("low", "medium", "high"):
            result["risk_level"] = "low"
        return result
    except Exception as e:
        return {
            "category": "Other",
            "explanation": "Unable to analyze this clause automatically.",
            "risk_level": "low",
            "risk_reason": None,
            "error": str(e),
        }


async def generate_summary(clauses: list[dict], risk_score: dict) -> dict:
    """Generate document summary from analyzed clauses."""
    # Build context for the summary
    clause_summaries = []
    for c in clauses:
        clause_summaries.append(
            f"[{c.get('category', 'Other')}] (Risk: {c.get('risk_level', 'low')}) "
            f"{c.get('explanation', 'N/A')}"
        )

    context = "\n".join(clause_summaries)
    prompt = (
        f"Risk Score: {risk_score['score']}/10\n"
        f"Risk Factors: {'; '.join(risk_score['factors'])}\n\n"
        f"Analyzed Clauses:\n{context}"
    )

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "overview": ["Unable to generate summary."],
            "most_concerning": [],
            "key_obligations": [],
            "document_type": "Unknown",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str


@app.post("/api/analyze")
async def analyze_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Main analysis endpoint. Accepts PDF upload or raw text."""
    start_time = time.time()

    # Extract text
    raw_text = ""
    source_type = "text"

    if file:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported for upload.")
        file_bytes = await file.read()
        raw_text = extract_text_from_pdf(file_bytes)
        source_type = "pdf"
        if not raw_text.strip():
            raise HTTPException(400, "Could not extract text from PDF. The file may be scanned/image-based.")
    elif text:
        raw_text = text
    else:
        raise HTTPException(400, "Please provide either a PDF file or text input.")

    # Validate length
    word_count = len(raw_text.split())
    if word_count > MAX_WORDS:
        raise HTTPException(
            400,
            f"Document exceeds maximum length of {MAX_WORDS:,} words. "
            f"Your document has {word_count:,} words.",
        )

    if word_count < 20:
        raise HTTPException(400, "Document is too short to analyze. Please provide a longer document.")

    # Segment into clauses
    clauses = segment_clauses(raw_text)

    # Analyze each clause: AI + rule-based hybrid
    analyzed_clauses: list[dict] = []
    for clause in clauses:
        # AI analysis
        ai_result = await analyze_clause_with_ai(clause["text"])

        # Rule-based analysis
        rule_result = rule_based_risk(clause["text"])

        # Merge: take the higher risk level
        risk_priority = {"high": 3, "medium": 2, "low": 1}
        ai_level = ai_result.get("risk_level", "low")
        rule_level = rule_result["max_level"]

        if risk_priority.get(rule_level, 0) > risk_priority.get(ai_level, 0):
            final_level = rule_level
            ai_result["risk_level"] = final_level
            if not ai_result.get("risk_reason"):
                ai_result["risk_reason"] = f"Rule-based detection: {len(rule_result['signals'])} risk signal(s)"

        analyzed_clauses.append({
            "index": clause["index"],
            "original_text": clause["text"],
            "category": ai_result.get("category", "Other"),
            "explanation": ai_result.get("explanation", ""),
            "risk_level": ai_result.get("risk_level", "low"),
            "risk_reason": ai_result.get("risk_reason"),
            "rule_signals": len(rule_result["signals"]),
        })

    # Compute risk score
    risk_score = compute_risk_score(analyzed_clauses)

    # Generate summary
    summary = await generate_summary(analyzed_clauses, risk_score)

    elapsed = round(time.time() - start_time, 2)

    return {
        "metadata": {
            "source_type": source_type,
            "word_count": word_count,
            "clause_count": len(analyzed_clauses),
            "processing_time_seconds": elapsed,
            "document_hash": hashlib.sha256(raw_text.encode()).hexdigest()[:16],
        },
        "clauses": analyzed_clauses,
        "risk_score": risk_score,
        "summary": summary,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# Serve frontend
FRONTEND_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found. Place index.html in backend/static/"}


# Serve other static files if needed
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
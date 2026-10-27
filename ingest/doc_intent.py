"""
ingest/doc_intent.py -- Document intent recognition and classification.
Identifies generic requests ("analyze the pdf") and classifies documents
into financial, resume, or other out-of-scope types.
"""

import re
from typing import List, Any

# Generic analysis request patterns
GENERIC_DOC_PATTERNS = [
    r"\banalyze\b.*\b(pdf|doc|document|file|slip|form|statement)\b",
    r"\bsummarize\b.*\b(pdf|doc|document|file|slip|form|statement)\b",
    r"\breview\b.*\b(pdf|doc|document|file|slip|form|statement)\b",
    r"\bexplain\b.*\b(pdf|doc|document|file|slip|form|statement)\b",
    r"\bwhat\s+does\s+(this|the)\s+(pdf|doc|document|file)\s+say\b",
    r"\bwhat('?s|\s+is)\s+in\s+(this|the)\s+(pdf|doc|document|file)\b",
    r"\boverview\b.*\b(pdf|doc|document|file)\b",
    r"^analyze\s+(the\s+|this\s+)?(pdf|doc|document|file)$",
    r"^summarize\s+(the\s+|this\s+)?(pdf|doc|document|file)$",
]

# Keywords strongly indicative of Indian / global financial documents
FINANCIAL_MARKERS = [
    "form 16", "form no. 16", "part a", "part b", "salary slip", "payslip",
    "gross salary", "net pay", "taxable income", "total income", "tds",
    "tax deducted", "pan", "tan", "assessment year", "financial year",
    "deductions under chapter", "annual information statement", "form 26as",
    "bank statement", "loan account", "interest certificate", "provident fund",
    "section 80c", "section 80d", "section 87a", "itr-1", "itr-2", "itr-4",
    "income tax department", "basic salary", "dearness allowance", "house rent allowance",
    "hra", "epf", "uan"
]

# Keywords strongly indicative of resume / CV / professional profile
RESUME_MARKERS = [
    "curriculum vitae", "resume", "work experience", "professional experience",
    "education", "bachelor", "master", "b.tech", "m.tech", "b.sc", "m.sc",
    "technical skills", "skills", "projects", "certifications", "github",
    "linkedin", "achievements", "personal details", "declaration", "objective",
    "employment history", "responsibilities", "academic background", "hobbies"
]


def is_generic_doc_request(text: str) -> bool:
    """Check if the user prompt is a generic document request like 'analyze the pdf' or 'summarize this doc'."""
    if not text:
        return False
    t = text.strip().lower()
    return any(re.search(pat, t) for pat in GENERIC_DOC_PATTERNS)


def classify_document(chunks: List[Any]) -> str:
    """Classify document chunks into 'financial', 'resume', or 'other'."""
    if not chunks:
        return "other"

    texts = []
    for c in chunks:
        t = getattr(c, "text", None) or (c.get("text", "") if isinstance(c, dict) else str(c))
        texts.append(t)
    full_text = " ".join(texts).lower()

    if not full_text.strip():
        return "other"

    fin_count = sum(1 for kw in FINANCIAL_MARKERS if re.search(rf"\b{re.escape(kw)}\b", full_text))
    res_count = sum(1 for kw in RESUME_MARKERS if re.search(rf"\b{re.escape(kw)}\b", full_text))

    # Definitive phrase anchors
    if "curriculum vitae" in full_text or "resume" in full_text or "professional summary" in full_text:
        if fin_count < 2:
            return "resume"

    if any(p in full_text for p in ["form 16", "form no. 16", "salary slip", "payslip", "annual information statement", "form 26as"]):
        return "financial"

    if fin_count >= 2 and fin_count >= res_count:
        return "financial"

    if res_count >= 2 and res_count > fin_count:
        return "resume"

    return "financial" if fin_count > res_count else ("resume" if res_count > 0 else "other")

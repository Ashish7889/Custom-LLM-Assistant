"""
ingest/form26as_parser.py -- Structured parser for Form 26AS (Annual Tax Statement).

Form 26AS contains:
  Part A  – TDS on salary, interest, rent, professional fees
  Part B  – TCS (Tax Collected at Source)
  Part C  – Advance tax / Self-assessment tax payments
  Part F  – SFT (Statement of Financial Transactions)

This parser extracts structured fields from plain-text Form 26AS exports
(downloaded from TRACES portal) and returns a list of SourceChunks with
rich metadata labels that doc_facts.py can answer questions about.
"""

import re
from typing import List, Dict, Any
from orchestrator.types import SourceChunk


# ─────────────────────────────────────────────────────────────
# Regex patterns for Form 26AS sections
# ─────────────────────────────────────────────────────────────

# TDS deductor entry pattern: TAN, Name, Amount, Date
_TDS_ENTRY = re.compile(
    r"(?P<tan>[A-Z]{4}\d{5}[A-Z])\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<amount>[\d,]+(?:\.\d{2})?)\s+"
    r"(?P<date>\d{2}-\d{2}-\d{4})",
    re.IGNORECASE,
)

# Amount after known labels
_LABEL_AMOUNT = re.compile(
    r"(?P<label>(?:Total TDS|Total TCS|Gross TDS|Tax Deposited|Amount Paid|Tax Collected|SFT Amount|Tax Payable|Refund Due|Advance Tax|Self Assessment Tax)[^\n]{0,40}?)"
    r"\s*[:=]?\s*"
    r"(?P<amount>[\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)

# Deductor section headers
_SECTION_HEADER = re.compile(
    r"PART\s+([A-F])\s*[:\-]?\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def _clean_amount(s: str) -> float:
    """Convert '1,23,456.78' → 123456.78"""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def parse_form26as(text: str, filename: str = "form26as.txt") -> List[SourceChunk]:
    """
    Parse plain-text Form 26AS export into structured SourceChunks.

    Returns one chunk per major section (Part A, B, C summary) plus
    a human-readable summary chunk for doc_facts.py extraction.
    """
    chunks: List[SourceChunk] = []
    chunk_id = 1

    # ── Extract Part A: TDS entries ──
    part_a_start = re.search(r"PART\s*A\b", text, re.IGNORECASE)
    part_b_start = re.search(r"PART\s*B\b", text, re.IGNORECASE)

    part_a_text = ""
    if part_a_start:
        end = part_b_start.start() if part_b_start else len(text)
        part_a_text = text[part_a_start.start():end]

    tds_entries: List[Dict[str, Any]] = []
    for m in _TDS_ENTRY.finditer(part_a_text):
        tds_entries.append({
            "tan": m.group("tan"),
            "deductor": m.group("name").strip(),
            "tds_amount": _clean_amount(m.group("amount")),
            "date": m.group("date"),
        })

    # ── Extract all labelled amounts ──
    labelled: Dict[str, float] = {}
    for m in _LABEL_AMOUNT.finditer(text):
        label = re.sub(r"\s+", " ", m.group("label")).strip().rstrip(":=-").strip()
        amt = _clean_amount(m.group("amount"))
        if amt > 0:
            labelled[label] = amt

    # ── Build summary text (used by doc_facts for QA) ──
    lines = ["FORM 26AS ANNUAL TAX STATEMENT SUMMARY"]
    lines.append(f"Source file: {filename}")
    lines.append("")

    # TDS summary
    total_tds = sum(e["tds_amount"] for e in tds_entries)
    if total_tds > 0:
        lines.append(f"Total TDS deducted: {total_tds:,.2f}")
        for e in tds_entries:
            lines.append(f"TDS by {e['deductor']} (TAN: {e['tan']}): {e['tds_amount']:,.2f} on {e['date']}")
    else:
        lines.append("Total TDS deducted: 0.00")

    # Labelled amounts
    for label, amt in labelled.items():
        lines.append(f"{label}: {amt:,.2f}")

    # If no structured data found, keep original text in a single chunk
    if not tds_entries and not labelled:
        lines.append("")
        lines.append("(Raw Form 26AS content follows)")
        lines.append(text[:3000])

    summary_text = "\n".join(lines)

    chunks.append(SourceChunk(
        id=chunk_id,
        text=summary_text,
        source_name=filename,
        page=1,
        trusted_level="official",   # Form 26AS is an official government document
    ))
    chunk_id += 1

    # ── Individual TDS entries as chunk 2 (for citation precision) ──
    if tds_entries:
        detail_lines = ["FORM 26AS: PART A - TDS DEDUCTION DETAILS"]
        for e in tds_entries:
            detail_lines.append(
                f"Deductor: {e['deductor']} | TAN: {e['tan']} | "
                f"TDS amount: {e['tds_amount']:,.2f} | Date: {e['date']}"
            )
        chunks.append(SourceChunk(
            id=chunk_id,
            text="\n".join(detail_lines),
            source_name=filename,
            page=2,
            trusted_level="official",
        ))

    return chunks


def is_form26as(text: str) -> bool:
    """Heuristic: detect if text is a Form 26AS document."""
    indicators = [
        "annual tax statement",
        "form 26as",
        "traces",
        "tds/tcs credit",
        "part a",
        "deductor tan",
        "tds deposited",
    ]
    lower = text.lower()
    return sum(1 for ind in indicators if ind in lower) >= 2

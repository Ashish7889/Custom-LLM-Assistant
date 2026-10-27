"""
ingest/itr_parser.py -- Structured parser for ITR-1 and ITR-2 JSON/text exports.

Handles:
  - ITR-1 (Sahaj): Salary, one house property, other income
  - ITR-2: Salary, capital gains (CG), multiple properties, foreign assets
  - ITR-4 (Sugam): Presumptive business income (44AD / 44ADA)

Input can be:
  - Plain-text ITR acknowledgement/verification slip
  - JSON export from income tax e-filing portal

Returns a list of SourceChunks with clean structured text
ready for doc_facts.py QA.
"""

import re
import json
from typing import List, Dict, Any, Optional
from orchestrator.types import SourceChunk


# ─────────────────────────────────────────────────────────────
# Label patterns common across ITR forms
# ─────────────────────────────────────────────────────────────
_LABEL_AMOUNT = re.compile(
    r"(?P<label>"
    r"(?:Gross salary|Total income|Tax payable|Tax paid|Refund|"
    r"Net tax payable|Total TDS|Advance tax|Self-assessment tax|"
    r"Capital gains|Short term capital gain|Long term capital gain|"
    r"Income from other sources|Income from house property|"
    r"Gross total income|Taxable income|Standard deduction|"
    r"Total deductions?|Section 87A rebate|Surcharge|Cess|"
    r"Agricultural income|Exempt income)[^\n]{0,40}?)"
    r"\s*[:=]?\s*"
    r"(?P<amount>-?[\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)

_SCHEDULE_CG = re.compile(
    r"Schedule\s*CG.{0,200}?(?:LTCG|STCG|Long.?term|Short.?term).{0,100}?"
    r"([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE | re.DOTALL,
)

_PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_AY_PATTERN = re.compile(r"(?:Assessment\s*Year|AY)\s*[:\s]*(\d{4}-\d{2,4})", re.IGNORECASE)
_ITR_FORM_PATTERN = re.compile(r"\bITR[-\s]?([1-9]|Sahaj|Sugam)\b", re.IGNORECASE)


def _clean_amount(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def _extract_json_fields(data: Dict[str, Any]) -> Dict[str, float]:
    """Recursively extract numeric fields from ITR JSON export."""
    results: Dict[str, float] = {}
    KEYS_OF_INTEREST = {
        "grossSalary", "GrossSalary", "TotalIncome", "TaxPayable",
        "TaxPaid", "TaxRefund", "LTCG", "STCG", "TDSonSalary",
        "TotalTDS", "AdvanceTax", "SelfAssessmentTax", "GrossTotalIncome",
        "TaxableTotalIncome", "Section87ARebate", "Surcharge", "EducationCess",
    }

    def recurse(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = k if not prefix else prefix + "." + k
                recurse(v, key)
        elif isinstance(obj, (int, float)) and obj != 0:
            short_key = prefix.split(".")[-1]
            if any(ki.lower() == short_key.lower() for ki in KEYS_OF_INTEREST):
                results[prefix] = float(obj)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, prefix + f"[{i}]")

    recurse(data)
    return results


def parse_itr(text: str, filename: str = "itr.txt") -> List[SourceChunk]:
    """
    Parse ITR plain-text or JSON export into structured SourceChunks.
    """
    chunks: List[SourceChunk] = []
    chunk_id = 1

    # Try JSON first
    json_data: Optional[Dict] = None
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json_data = json.loads(stripped)
        except json.JSONDecodeError:
            pass

    lines = ["ITR FILING SUMMARY"]
    lines.append(f"Source file: {filename}")
    lines.append("")

    # Assessment year
    ay_match = _AY_PATTERN.search(text)
    if ay_match:
        lines.append(f"Assessment Year: {ay_match.group(1)}")

    # ITR form type
    form_match = _ITR_FORM_PATTERN.search(text)
    if form_match:
        lines.append(f"ITR Form: ITR-{form_match.group(1)}")

    # PAN (redacted for safety)
    pan_matches = _PAN_PATTERN.findall(text)
    if pan_matches:
        lines.append(f"PAN (masked): XXXXX{pan_matches[0][5:]}")

    lines.append("")

    # Extract from JSON
    if json_data:
        json_fields = _extract_json_fields(json_data)
        for key, val in json_fields.items():
            short = key.split(".")[-1]
            lines.append(f"{short}: {val:,.2f}")
    else:
        # Extract from text labels
        seen: Dict[str, float] = {}
        for m in _LABEL_AMOUNT.finditer(text):
            label = re.sub(r"\s+", " ", m.group("label")).strip().rstrip(":=-").strip()
            amt = _clean_amount(m.group("amount"))
            if label not in seen:
                seen[label] = amt
                lines.append(f"{label}: {amt:,.2f}")

        # Schedule CG
        cg_match = _SCHEDULE_CG.search(text)
        if cg_match:
            lines.append(f"Capital gain amount: {_clean_amount(cg_match.group(1)):,.2f}")

    summary_text = "\n".join(lines)
    chunks.append(SourceChunk(
        id=chunk_id,
        text=summary_text,
        source_name=filename,
        page=1,
        trusted_level="verified",
    ))

    return chunks


def is_itr_document(text: str) -> bool:
    """Heuristic detection for ITR documents."""
    indicators = [
        "income tax return",
        "itr-1", "itr-2", "itr-4",
        "sahaj", "sugam",
        "assessment year",
        "gross total income",
        "tax payable",
        "acknowledgement number",
        "schedule cg",
        "schedule s",
    ]
    lower = text.lower()
    return sum(1 for ind in indicators if ind in lower) >= 2

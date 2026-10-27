"""
ingest/doc_facts.py -- High-precision structured fact extractor from financial documents.
Extracts (label, amount, chunk_id) tuples from raw text and matches user queries directly.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class DocFact:
    label: str
    amount: str
    chunk_id: int
    raw_amount: float


# Regex for amounts: comma-grouped numbers (Indian or Western) or numbers with 2 decimals
# e.g., 12,00,000.00 | 12,00,000 | 93,800.00 | 50000.00
AMOUNT_REGEX = re.compile(
    r"(?:Rs\.?|INR|₹)?\s*(\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\b\d+\.\d{2})\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "what", "is", "the", "my", "pdf", "document", "slip", "form",
    "in", "of", "a", "an", "for", "and", "to", "this", "that", "it",
    "tell", "me", "how", "much", "are", "there", "can", "you", "show",
    "give", "please", "about", "as", "per", "details", "value", "amount",
    "gross",  # Note: keep gross in keywords if part of question, but remove general query filler
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
}
# Don't strip key financial nouns
FINANCIAL_CORE_NOUNS = {"gross", "net", "pay", "salary", "tax", "income", "tds", "deduction", "allowance", "exemption", "hra", "pf"}
QUERY_STOPWORDS = {w for w in STOPWORDS if w not in FINANCIAL_CORE_NOUNS}


def normalize_synonyms(text: str) -> str:
    """Normalize domain synonyms to canonical tokens."""
    t = text.lower()
    t = re.sub(r"\btax\s+deducted\s+at\s+source\b", "tds", t)
    t = re.sub(r"\btake\s+home\b", "net pay", t)
    t = re.sub(r"\bpf\b", "provident fund", t)
    return t


def stem_word(w: str) -> str:
    """Simple English plural/noun stemmer."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3 and not w.endswith("ses"):
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize_and_stem(text: str) -> Set[str]:
    """Tokenize, replace synonyms, strip stopwords & month names, and apply plural stemming."""
    normalized = normalize_synonyms(text)
    words = re.findall(r"\b[a-z0-9]+\b", normalized)
    tokens = set()
    for w in words:
        if w in QUERY_STOPWORDS:
            continue
        st = stem_word(w)
        if st not in QUERY_STOPWORDS and len(st) > 1:
            tokens.add(st)
    return tokens


def clean_label_text(raw_label: str) -> str:
    """Clean candidate label text according to rules."""
    # 1. If (Rs.) appears: if text after it exists, keep text after it; otherwise keep text before it
    if re.search(r"\(rs\.?\)", raw_label, re.IGNORECASE):
        parts = re.split(r"\(rs\.?\)", raw_label, flags=re.IGNORECASE)
        after = parts[-1].strip()
        before = parts[0].strip()
        raw_label = after if after else before

    # 2. Strip "Less:" / "Add:"
    raw_label = re.sub(r"\b(less|add)\s*:\s*", "", raw_label, flags=re.IGNORECASE)

    # 3. Normalize whitespace
    raw_label = re.sub(r"\s+", " ", raw_label).strip()

    # 4. Strip leading numbering/bullets like "1.", "(a)", "•", "-"
    raw_label = re.sub(r"^(?:[\d\.\:\-\*\•\#]+\s+|\([a-zA-Z0-9]+\)\s+|[a-zA-Z]\.\s+)", "", raw_label).strip()
    raw_label = raw_label.strip(" :.-,")

    # 5. Maximum 8 words (keep the words immediately preceding the amount)
    words = raw_label.split()
    if len(words) > 8:
        words = words[-8:]
    return " ".join(words).strip()


def is_year_or_id(amount_str: str) -> bool:
    """Check if matched string is likely a calendar/fiscal year or an ID."""
    clean = amount_str.replace(",", "")
    try:
        val = float(clean)
        # Plain 4-digit year like 2023.00, 2024.00, 2025.00 without commas
        if re.match(r"^(?:19|20)\d{2}(?:\.00)?$", amount_str) and "," not in amount_str:
            return True
        if val <= 0:
            return True
    except ValueError:
        return True
    return False


def extract_facts(chunks: List[Any]) -> List[DocFact]:
    """Find (label, amount, chunk_id) pairs in flattened chunk texts."""
    facts: List[DocFact] = []

    for chunk in chunks:
        c_text = getattr(chunk, "text", None) or (chunk.get("text", "") if isinstance(chunk, dict) else str(chunk))
        c_id = getattr(chunk, "id", None) or (chunk.get("id", 1) if isinstance(chunk, dict) else 1)

        prev_end = 0
        for match in AMOUNT_REGEX.finditer(c_text):
            amt_str = match.group(1).strip()
            if is_year_or_id(amt_str):
                continue

            raw_label = c_text[prev_end : match.start()]
            label = clean_label_text(raw_label)
            prev_end = match.end()

            if not label:
                continue

            try:
                num_val = float(amt_str.replace(",", ""))
            except ValueError:
                continue

            facts.append(DocFact(label=label, amount=amt_str, chunk_id=c_id, raw_amount=num_val))

    return facts


def answer_from_facts(question: str, chunks: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Answer user query directly from structured document facts.
    Score = |q ∩ label| / |q|; accept >= 0.75;
    Return up to 3 sentences like 'Gross salary is Rs. 12,00,000.00.' each with chunk id.
    Return None if nothing matches.
    """
    facts = extract_facts(chunks)
    if not facts:
        return None

    q_tokens = tokenize_and_stem(question)
    if not q_tokens:
        return None

    matches: List[Tuple[float, DocFact]] = []
    for f in facts:
        l_tokens = tokenize_and_stem(f.label)
        if not l_tokens:
            continue
        intersection = q_tokens & l_tokens
        score = len(intersection) / len(q_tokens)
        if score >= 0.75:
            matches.append((score, f))

    if not matches:
        return None

    # Sort descending by score, then preserve order of appearance
    matches.sort(key=lambda x: x[0], reverse=True)

    seen_amounts = set()
    results: List[Dict[str, Any]] = []

    for score, f in matches:
        if f.amount in seen_amounts:
            continue
        seen_amounts.add(f.amount)

        clean_lbl = f.label
        # Format label title
        clean_lbl = clean_lbl[0].upper() + clean_lbl[1:] if clean_lbl else "Amount"
        # Standardize Rs. formatting
        amt_disp = f.amount if f.amount.startswith("Rs.") or f.amount.startswith("₹") else f"Rs. {f.amount}"
        sent_text = f"{clean_lbl} is {amt_disp}."

        results.append({
            "text": sent_text,
            "chunk_id": f.chunk_id,
            "label": f.label,
            "amount": f.amount,
        })
        if len(results) >= 3:
            break

    return results if results else None


def overview_from_facts(chunks: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Extract up to 6 key figures (gross salary, gross earnings, taxable income, tax payable, TDS, net pay), priority first."""
    facts = extract_facts(chunks)
    if not facts:
        return None

    # Priority search keys
    priority_categories = [
        ("Gross salary", ["gross salary", "gross pay", "salary as per provisions"]),
        ("Gross earnings", ["gross earnings", "total earnings", "total gross"]),
        ("Taxable income", ["taxable income", "total taxable income", "total income"]),
        ("Tax payable", ["tax payable", "total tax payable", "net tax payable", "tax liability"]),
        ("TDS", ["tds", "tax deducted", "tax deducted at source"]),
        ("Net pay", ["net pay", "take home", "net salary", "net amount"]),
    ]

    selected_facts: List[Dict[str, Any]] = []
    seen_amounts = set()

    for display_name, candidates in priority_categories:
        for candidate in candidates:
            c_tokens = tokenize_and_stem(candidate)
            best_fact = None
            best_score = 0.0

            for f in facts:
                if f.amount in seen_amounts:
                    continue
                f_tokens = tokenize_and_stem(f.label)
                if not f_tokens:
                    continue
                score = len(c_tokens & f_tokens) / len(c_tokens)
                if score >= 0.75 and score > best_score:
                    best_score = score
                    best_fact = f

            if best_fact:
                seen_amounts.add(best_fact.amount)
                amt_disp = f"Rs. {best_fact.amount}" if not best_fact.amount.startswith("Rs.") else best_fact.amount
                selected_facts.append({
                    "text": f"{display_name} is {amt_disp}.",
                    "chunk_id": best_fact.chunk_id,
                    "label": display_name,
                    "amount": best_fact.amount,
                })
                break

        if len(selected_facts) >= 6:
            break

    return selected_facts if selected_facts else None

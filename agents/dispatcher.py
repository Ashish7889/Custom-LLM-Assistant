"""
agents/dispatcher.py -- Intake Clerk Agent (TinyLlama-1.1B).
Responsible for:
1. Fast client intake and prompt sanitization (<100ms).
2. Entity normalization to standard Rupees (1 Lakh = 100,000, 1 Crore = 10,000,000, 50k = 50,000).
3. Intent classification and preliminary open question tagging.
4. Emitting a warm, immediate client acknowledgment (ack).
"""

import re
import json
import logging
from typing import Dict, Any, Optional

from agents.firm_prompt import build_prompt, SHARED_COMPACT, ROLE_CARDS

logger = logging.getLogger("finsage.agents.dispatcher")


def normalize_indian_amount(val_str: str) -> Optional[float]:
    """Convert Indian currency notations like '7.5 lakh', '1.5L', '50k', '1.2 crore' to numeric rupees."""
    if not val_str:
        return None
    s = str(val_str).lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").strip()
    
    m_cr = re.search(r"([\d\.]+)\s*(?:cr|crore|crores)\b", s)
    if m_cr:
        try:
            return float(m_cr.group(1)) * 10000000.0
        except ValueError:
            pass

    m_lakh = re.search(r"([\d\.]+)\s*(?:l|lakh|lakhs|lac|lacs)\b", s)
    if m_lakh:
        try:
            return float(m_lakh.group(1)) * 100000.0
        except ValueError:
            pass

    m_k = re.search(r"([\d\.]+)\s*(?:k|thousand)\b", s)
    if m_k:
        try:
            return float(m_k.group(1)) * 1000.0
        except ValueError:
            pass

    m_plain = re.search(r"\b(\d{4,12}(?:\.\d+)?)\b", s)
    if m_plain:
        try:
            return float(m_plain.group(1))
        except ValueError:
            pass

    return None


class IntakeDispatcher:
    """Intake Clerk agent for FinSage & Associates."""

    def __init__(self, generator=None):
        self.generator = generator

    def process(self, client_message: str) -> Dict[str, Any]:
        """Process client query and return structured FACTS memo."""
        msg = client_message.strip()
        lower = msg.lower()

        # Regex fallback baseline (deterministic safety net)
        facts: Dict[str, Any] = {
            "gross_income_rupees": None,
            "employment_type": "salaried" if any(w in lower for w in ["salaried", "salary", "job", "employee", "clerk", "ctc"]) else "other",
            "regime_preference": "new" if "new regime" in lower else ("old" if "old regime" in lower else None),
            "financial_year": "2024-25" if ("2024" in lower or "fy24" in lower) else "2025-26",
            "deductions_rupees": {},
            "loan": {},
            "ticker": None,
        }

        # Check for income figures in message
        income_match = normalize_indian_amount(msg)
        if income_match:
            facts["gross_income_rupees"] = income_match

        # Extract deductions
        if "80c" in lower:
            facts["deductions_rupees"]["80C"] = 150000.0
        if "80d" in lower:
            facts["deductions_rupees"]["80D"] = 25000.0

        # Classify intent
        intent = "concept"
        needs_web = False
        ack = "Understood. The firm is gathering official sources and calculation sheets for you now."

        if any(w in lower for w in ["tax", "itr", "regime", "slab", "pay", "standard deduction"]):
            intent = "tax_calc" if facts["gross_income_rupees"] else "regime_compare"
            ack = "Understood. Our team is retrieving the verified Indian tax slabs and running the computations."
        elif any(w in lower for w in ["notice", "scrutiny", "143", "148", "penalty", "ais", "26as"]):
            intent = "notice_risk"
            needs_web = True
            ack = "Understood. Reviewing your query against official CBDT scrutiny rules and disclosure factors."
        elif any(w in lower for w in ["sukanya", "ssy", "pm-kisan", "pm kisan", "atal", "apy", "ppf", "nps", "scheme"]):
            intent = "scheme_info"
            ack = "Understood. Pulling verified sovereign scheme rules and gazette circulars."
        elif any(w in lower for w in ["loan", "emi", "interest rate", "prepayment", "home loan"]):
            intent = "loan_emi"
            ack = "Understood. Handing your loan details to our analyst to compute amortization and interest schedules."
        elif any(w in lower for w in ["share price", "stock", "pe ratio", "market cap", "quote", "nifty", "sensex"]):
            intent = "market_quote"
            ack = "Understood. Fetching real-time exchange quotes with verified timestamped data."

        if any(w in lower for w in ["latest", "news", "recent", "update", "today", "budget 2025", "budget 2026", "circular"]):
            needs_web = True

        open_questions = []
        if intent == "tax_calc" and not facts["gross_income_rupees"]:
            open_questions.append("gross annual income in rupees")

        return {
            "from": "intake",
            "to": "analyst",
            "type": "FACTS",
            "content": {
                "intent": intent,
                "facts": facts,
                "open_questions": open_questions,
                "needs_web": needs_web,
                "ack": ack,
            },
            "cites": [],
            "flags": [],
        }

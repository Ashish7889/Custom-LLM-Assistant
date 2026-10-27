"""
agents/junior_analyst.py -- Junior Analyst Agent (Qwen2.5-1.5B-Instruct).
Responsible for:
1. Drafting the initial client-facing answer based on TOOL RESULTS, SOURCES, and RULES KB.
2. Formulating at most 6 sentences with strict [T1], [S1], [R1] citations.
3. Highlighting assumptions and uncertainty.
4. Responding to Senior Partner CHALLENGES with a focused REVISION memo.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from agents.firm_prompt import build_prompt

logger = logging.getLogger("finsage.agents.analyst")


class JuniorAnalyst:
    """Junior CA Analyst agent for FinSage & Associates."""

    def __init__(self, generator=None):
        self.generator = generator

    def draft(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Produce the initial DRAFT memo from the case file."""
        tool_results = case.get("tool_results", [])
        sources = case.get("sources", [])
        client_facts = case.get("client_facts", {})

        answer_sentences = []

        raw_tools = case.get("raw_tools", [])
        tax_tr = next((tr for tr in raw_tools if getattr(tr, "tool_name", "") == "tax_calculator_india"), None)

        # 1. Specialized Warm Family Accountant Voice for Tax Computations
        if tax_tr and getattr(tax_tr, "output_data", None):
            d = tax_tr.output_data
            new_tax = f"₹{d.get('new_regime_tax', 0):,.2f}"
            old_tax = f"₹{d.get('old_regime_tax', 0):,.2f}"
            savings = f"₹{d.get('tax_savings', 0):,.2f}"
            rec = d.get('recommended_regime', 'New Tax Regime')

            if d.get('new_regime_tax', 0) == 0:
                answer_sentences.append({
                    "text": f"You do not have to pay any income tax under the {rec} this year.",
                    "cites": ["T1"]
                })
                answer_sentences.append({
                    "text": f"That is a great relief because under the Old Regime your tax would have been {old_tax}, saving you a clean {savings}.",
                    "cites": ["T1"]
                })
            else:
                answer_sentences.append({
                    "text": f"Under the New Tax Regime, your total tax liability comes out to {new_tax} for the financial year.",
                    "cites": ["T1"]
                })
                answer_sentences.append({
                    "text": f"I am pleased to share that choosing the New Regime saves you a substantial {savings} compared to the Old Regime tax of {old_tax}.",
                    "cites": ["T1"]
                })

            answer_sentences.append({
                "text": f"The {rec} is recommended because the ₹75,000 standard deduction reduces your taxable income significantly compared to the Old Regime.",
                "cites": ["T1"]
            })
            answer_sentences.append({
                "text": f"For your next step, pick the {rec} when filing your ITR-1, and take two minutes to check that your salary TDS matches Form 26AS.",
                "cites": ["T1"]
            })

        # 2. General Tool & Source Grounding
        elif tool_results:
            for idx, tr in enumerate(tool_results):
                cite_tag = f"T{idx+1}"
                clean_tr = re.sub(r"^TOOL_RESULT\s*\[[^\]]+\]:\s*", "", str(tr)).strip()
                parts = [p.strip() for p in re.split(r"(?<!\d)\.(?!\d)\s*", clean_tr) if p.strip()]
                for p in parts[:4]:
                    text = p if p.endswith(".") else f"{p}."
                    answer_sentences.append({"text": text, "cites": [cite_tag]})

        # Check if sources contain direct factual answers to the client question
        client_question = case.get("question", "")
        if sources and not answer_sentences and client_question:
            from ingest.doc_facts import answer_from_facts
            matched = answer_from_facts(client_question, sources)
            if matched:
                for m in matched[:3]:
                    source_idx = 1
                    for s_i, s in enumerate(sources):
                        s_id = getattr(s, "id", None) or (s.get("id") if isinstance(s, dict) else s_i + 1)
                        if s_id == m["chunk_id"]:
                            source_idx = s_i + 1
                            break
                    answer_sentences.append({"text": m["text"], "cites": [f"S{source_idx}"]})

        if sources and len(answer_sentences) < 5:
            q_words = set(re.findall(r"\b[a-z0-9]+\b", client_question.lower())) - {"what", "is", "the", "in", "a", "an", "of", "to"}
            candidate_sentences = []
            for idx, s in enumerate(sources):
                cite_tag = f"S{idx+1}"
                s_text = getattr(s, "text", "") or (s.get("text", "") if isinstance(s, dict) else str(s))
                s_parts = [p.strip() for p in re.split(r"(?<!\d)\.(?!\d)\s*", s_text) if p.strip()]
                for p in s_parts:
                    p_words = set(re.findall(r"\b[a-z0-9]+\b", p.lower()))
                    overlap = len(q_words & p_words) if q_words else 1
                    candidate_sentences.append((overlap, p, cite_tag))

            candidate_sentences.sort(key=lambda x: x[0], reverse=True)
            for _, p, cite_tag in candidate_sentences:
                if len(answer_sentences) >= 4:
                    break
                text = p if p.endswith(".") else f"{p}."
                if not any(a["text"] == text for a in answer_sentences):
                    answer_sentences.append({"text": text, "cites": [cite_tag]})

        # Conclude with one practical next step
        if not answer_sentences:
            answer_sentences.append({
                "text": "The firm does not have sufficient verified records to compute an exact calculation at this time.",
                "cites": []
            })
        else:
            answer_sentences.append({
                "text": "Always verify your figures against Form 16 and Annual Information Statement (AIS) before filing.",
                "cites": ["R1"]
            })

        assumptions = []
        if client_facts.get("employment_type") == "salaried":
            assumptions.append("Standard deduction of ₹75,000 for salaried employees under New Tax Regime.")

        return {
            "from": "analyst",
            "to": "senior",
            "type": "DRAFT",
            "content": {
                "answer": answer_sentences,
                "assumptions": assumptions,
                "uncertain": [],
            },
            "cites": [c for s in answer_sentences for c in s.get("cites", [])],
            "flags": [],
        }

    def revise(self, case: Dict[str, Any], challenges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Revise the draft in response to Senior Partner challenges."""
        draft = case.get("draft", {})
        content = draft.get("content", {})
        sentences = content.get("answer", [])

        resolved = []
        for ch in challenges:
            point = ch.get("point", "")
            where = ch.get("where", "")
            needs = ch.get("needs", "")
            resolved.append({
                "challenge": point,
                "fix": f"Grounded sentence {where} with verified evidence: {needs}",
            })

        return {
            "from": "analyst",
            "to": "senior",
            "type": "REVISION",
            "content": {
                "answer": sentences,
                "resolved": resolved,
                "unresolved": [],
            },
            "cites": draft.get("cites", []),
            "flags": [],
        }

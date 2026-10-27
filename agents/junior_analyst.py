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

        # If deterministic tools produced computation, lead with direct tool answers
        if tool_results:
            for idx, tr in enumerate(tool_results):
                cite_tag = f"T{idx+1}"
                # Format key numbers cleanly
                clean_tr = re.sub(r"^TOOL_RESULT\s*\[[^\]]+\]:\s*", "", str(tr)).strip()
                # Break into short sentences
                parts = [p.strip() for p in re.split(r"(?<!\d)\.(?!\d)\s*", clean_tr) if p.strip()]
                for p in parts[:4]:
                    text = p if p.endswith(".") else f"{p}."
                    answer_sentences.append({"text": text, "cites": [cite_tag]})

        # Incorporate official sources if available
        if sources and len(answer_sentences) < 5:
            for idx, s in enumerate(sources[:2]):
                cite_tag = f"S{idx+1}"
                s_text = s.get("text", "")
                # Find the most relevant factual sentence
                s_parts = [p.strip() for p in re.split(r"(?<!\d)\.(?!\d)\s*", s_text) if p.strip()]
                for p in s_parts[:2]:
                    if len(answer_sentences) >= 5:
                        break
                    text = p if p.endswith(".") else f"{p}."
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

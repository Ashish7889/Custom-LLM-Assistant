"""
agents/firm_engine.py -- FinSage 3-Agent CA Practice Collaboration Engine.
Orchestrates:
1. Intake Clerk (TinyLlama-1.1B)
2. Junior Analyst (Qwen2.5-1.5B)
3. Senior Partner CA (DeepSeek-R1-Distill-Qwen-1.5B)
Enforces a maximum of 2 review rounds, strict evidence grounding, and delivers full memo audit trails.
"""

import logging
from typing import Dict, Any, List, Optional, Generator

from agents.firm_prompt import render_case_file
from agents.rules_kb import STANDARD_RULES_KB
from agents.dispatcher import IntakeDispatcher
from agents.junior_analyst import JuniorAnalyst
from agents.senior_ca import SeniorPartnerCA

logger = logging.getLogger("finsage.agents.firm")


class FinSageFirm:
    """Virtual Chartered Accountancy Firm collaboration manager."""

    def __init__(self, generator=None):
        self.generator = generator
        self.intake_clerk = IntakeDispatcher(generator)
        self.junior_analyst = JuniorAnalyst(generator)
        self.senior_partner = SeniorPartnerCA(generator)

    def intake(self, client_message: str) -> Dict[str, Any]:
        """Intake clerk reads client query and emits FACTS memo."""
        return self.intake_clerk.process(client_message)

    def init_case(
        self,
        client_message: str,
        facts_memo: Dict[str, Any],
        tool_results: List[Any],
        sources: List[Any],
        client_history: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the structured case file from intake facts, tool outputs, and sources."""
        content = facts_memo.get("content", {})
        client_facts = content.get("facts", {})
        open_questions = content.get("open_questions", [])

        # Format tool results
        t_strings = []
        for tr in tool_results:
            formatted = getattr(tr, "formatted_text", None) or str(tr)
            t_strings.append(formatted)

        # Format sources
        s_dicts = []
        for s in sources:
            s_dicts.append({
                "id": getattr(s, "id", None),
                "trust": getattr(s, "trusted_level", "official"),
                "date": getattr(s, "date", "2026-10-27"),
                "url": getattr(s, "url", "https://incometax.gov.in"),
                "text": getattr(s, "text", str(s)),
            })

        return {
            "client_message": client_message,
            "client_facts": client_facts,
            "open_questions": open_questions,
            "tool_results": t_strings,
            "sources": s_dicts,
            "rules": STANDARD_RULES_KB,
            "memos": [facts_memo],
            "draft": None,
            "client_history": client_history or "none",
        }

    def collaborate(self, case: Dict[str, Any]) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Run the multi-agent firm collaboration protocol, yielding intermediate memos.
        Returns the final sign-off memo.
        """
        # Step 1: Junior Analyst drafts initial memo
        draft_memo = self.junior_analyst.draft(case)
        case["draft"] = draft_memo
        case["memos"].append(draft_memo)
        yield draft_memo

        # Step 2: Senior Partner reviews draft (Round 1)
        review_memo = self.senior_partner.review(case, round_num=1)
        case["memos"].append(review_memo)
        yield review_memo

        verdict = review_memo.get("content", {}).get("verdict", "approve")
        challenges = review_memo.get("content", {}).get("challenges", [])

        # Step 3: If revision required, Junior Analyst responds to challenges
        if verdict == "revise" and challenges:
            revision_memo = self.junior_analyst.revise(case, challenges)
            case["draft"] = revision_memo
            case["memos"].append(revision_memo)
            yield revision_memo

            # Final Partner review (Round 2)
            review_2 = self.senior_partner.review(case, round_num=2)
            case["memos"].append(review_2)
            yield review_2

        # Step 4: Senior Partner issues FINAL memo
        final_memo = self.senior_partner.sign_off(case)
        case["memos"].append(final_memo)
        yield final_memo

        return final_memo

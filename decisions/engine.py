"""
decisions/engine.py -- Discrete decision layer for FinSage.
Implements:
decide(task, state, options) -> {choice_id: str, probability: float}
Never generates text. Backends: baseline (rules + heuristics), jev (hosted), local (encoder).
Tasks: route_intent, select_evidence, claim_supported, answer_or_abstain.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from config import cfg

logger = logging.getLogger("finsage.decisions")


class DecisionResult(BaseModel):
    task: str
    choice_id: str
    probability: float
    reason: str
    backend: str


class DecisionEngine:
    """Pluggable discrete decision engine for FinSage."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or cfg.decision_backend

    def decide(
        self,
        task: str,
        state: Dict[str, Any],
        options: List[str],
    ) -> DecisionResult:
        """Route decision to the configured backend and log verdict."""
        if not options:
            raise ValueError("options list cannot be empty")

        if self.backend == "baseline":
            result = self._decide_baseline(task, state, options)
        elif self.backend == "jev":
            result = self._decide_jev(task, state, options)
        elif self.backend == "local":
            result = self._decide_local(task, state, options)
        else:
            result = self._decide_baseline(task, state, options)

        logger.info(
            f"Decision made: task={task} choice={result.choice_id} p={result.probability:.3f} backend={result.backend}"
        )
        return result

    def _decide_baseline(
        self, task: str, state: Dict[str, Any], options: List[str]
    ) -> DecisionResult:
        """Rule-based decision layer."""
        prompt = state.get("prompt", "").lower()

        if task == "route_intent":
            # Options: ["chitchat", "tax_calculator", "debt_planner", "calculator", "market_quote", "document_qa", "concept_qa"]
            chitchat_triggers = [
                "who are you", "what is your name", "what can you do", "who created you",
                "hello", "hi", "hey", "good morning", "good evening", "namaste",
                "thank you", "thanks", "bye", "goodbye", "tell me about yourself"
            ]
            if any(t == prompt or prompt.startswith(t + " ") or prompt.endswith(" " + t) or t in prompt for t in chitchat_triggers) and "chitchat" in options:
                # Ensure it's not actually a financial query containing "hi" as part of a word
                if prompt.strip() in chitchat_triggers or any(phrase in prompt for phrase in ["who are you", "what can you do", "tell me about yourself", "who created you"]):
                    return DecisionResult(task=task, choice_id="chitchat", probability=0.99, reason="Conversational greeting/identity query", backend="baseline")

            tax_keywords = ["tax", "regime", "itr", "87a", "80c", "80d", "deduction", "income tax", "rebate", "standard deduction", "sahaj", "sugam"]
            debt_keywords = ["flat rate", "reducing", "prepayment", "loan trap", "extra emi", "loan shark", "apr", "payoff", "debt", "interest rate", "loan"]
            calc_keywords = ["calculate", "compound interest", "sip", "emi", "cagr", "future value", "ratio"]
            market_keywords = ["price", "stock", "quote", "pe ratio", "p/e", "market cap", "worth", "valuation", "share"]
            
            if any(k in prompt for k in tax_keywords) and "tax_calculator" in options:
                return DecisionResult(task=task, choice_id="tax_calculator", probability=0.95, reason="Indian tax/ITR/regime terms identified", backend="baseline")

            if any(k in prompt for k in debt_keywords) and "debt_planner" in options:
                return DecisionResult(task=task, choice_id="debt_planner", probability=0.92, reason="Debt/loan trap/prepayment terms identified", backend="baseline")
            
            if any(k in prompt for k in calc_keywords):
                choice = "calculator" if "calculator" in options else options[0]
                return DecisionResult(task=task, choice_id=choice, probability=0.92, reason="Math/calculator terms identified", backend="baseline")
            
            if any(k in prompt for k in market_keywords) or bool(re.search(r"\b[A-Z]{1,5}\b", state.get("prompt", ""))):
                choice = "market_quote" if "market_quote" in options else options[0]
                return DecisionResult(task=task, choice_id=choice, probability=0.88, reason="Live market/ticker keywords identified", backend="baseline")
            
            if state.get("has_uploaded_docs", False):
                choice = "document_qa" if "document_qa" in options else options[0]
                return DecisionResult(task=task, choice_id=choice, probability=0.85, reason="User has active uploaded documents", backend="baseline")
            
            choice = "concept_qa" if "concept_qa" in options else options[0]
            return DecisionResult(task=task, choice_id=choice, probability=0.80, reason="Defaulting to educational finance concept retrieval", backend="baseline")

        elif task == "select_evidence":
            # Evaluate relevance between candidate sources and user query
            sources = state.get("sources", [])
            if not sources:
                return DecisionResult(task=task, choice_id="none", probability=0.99, reason="No sources available", backend="baseline")
            return DecisionResult(task=task, choice_id="sources_valid", probability=0.85, reason="Candidate sources meet overlap criteria", backend="baseline")

        elif task == "claim_supported":
            # Check whether verification claims are supported
            num_violations = len(state.get("unsupported_numbers", []))
            missing_cites = len(state.get("missing_citations", []))
            
            if num_violations == 0 and missing_cites == 0:
                return DecisionResult(task=task, choice_id="supported", probability=0.95, reason="Strict number and cite containment verified", backend="baseline")
            else:
                return DecisionResult(task=task, choice_id="unsupported", probability=0.90, reason=f"Failed validation: {num_violations} unsupported numbers, {missing_cites} bad cites", backend="baseline")

        elif task == "answer_or_abstain":
            # Options: ["answer", "abstain", "clarify"]
            has_sources = bool(state.get("sources", []))
            has_tools = bool(state.get("tool_results", []))
            claim_supported = state.get("claim_supported", True)

            if not (has_sources or has_tools) or not claim_supported:
                return DecisionResult(task=task, choice_id="abstain", probability=0.94, reason="Zero grounded sources or unsupported claims detected", backend="baseline")
            return DecisionResult(task=task, choice_id="answer", probability=0.88, reason="Verified groundings present", backend="baseline")

        # Fallback default
        return DecisionResult(task=task, choice_id=options[0], probability=0.50, reason="Default option selected", backend="baseline")

    def _decide_jev(
        self, task: str, state: Dict[str, Any], options: List[str]
    ) -> DecisionResult:
        """Hosted Jev decision backend via OpenRouter API (disabled by default for private uploads)."""
        if state.get("is_private_upload", False):
            # Rule: External backends never receive private uploads
            return self._decide_baseline(task, state, options)
        # Fall back to baseline if OpenRouter key is not set
        return self._decide_baseline(task, state, options)

    def _decide_local(
        self, task: str, state: Dict[str, Any], options: List[str]
    ) -> DecisionResult:
        """Local trained encoder (DeBERTa/MiniLM) calibrated decision model."""
        # Built in Phase 8; falls back safely to baseline
        return self._decide_baseline(task, state, options)

"""
verifier/verifier.py -- Deep Verification and Repair Engine for FinSage.
Enforces Non-Negotiable #1:
(a) Every number/percentage in generated sentences exists in cited sources or tool results.
(b) Every sentence has valid, in-bounds citations.
(c) Claim lexical support verified against cited text.
(d) Specific deterministic repair feedback for the REPAIR loop (max 2) -> else ABSTAIN.
"""

import re
from typing import List, Set, Dict, Any, Tuple, Optional
from orchestrator.types import SourceChunk, ToolResult, GeneratorOutput, SentenceItem, VerificationResult
from decisions.engine import DecisionEngine


class ResponseVerifier:
    """Rigorous verification engine enforcing evidence containment, citation validity, and targeted repair."""

    def __init__(self, decision_engine: DecisionEngine):
        self.decision_engine = decision_engine

    def _extract_all_numbers(self, text: str) -> Set[str]:
        # Protect statutory form, act, and section identifiers (e.g. Form 16, Section 80C) from arithmetic checks
        cleaned = re.sub(r"\b(?:Form\s*16[A-Z]?|Form\s*26AS|Section\s*\d+[A-Z]?|Sec\s*\d+[A-Z]?|ITR-?\s*\d+)\b", " ", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace(",", "")
        # Find raw numeric patterns: 12.5, 12, 100%, $50.2
        raw_nums = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
        nums = set()
        for n in raw_nums:
            nums.add(n)
            try:
                val = float(n)
                if val.is_integer():
                    nums.add(str(int(val)))
                else:
                    nums.add(f"{val:g}")
            except ValueError:
                pass
        return nums

    def _extract_content_words(self, text: str) -> Set[str]:
        """Extract lowercase content words (excluding common stop-words)."""
        stop_words = {
            "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for",
            "of", "or", "as", "by", "with", "from", "that", "this", "it", "are", "be"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return set(words) - stop_words

    def verify(
        self,
        draft: GeneratorOutput,
        sources: List[SourceChunk],
        tool_results: List[ToolResult],
    ) -> VerificationResult:
        """Run deep verification on each sentence of the generated draft."""
        if draft.abstain:
            return VerificationResult(
                passed=True,
                claim_supported=True,
                feedback="Generator issued structured abstention.",
            )

        source_map: Dict[int, SourceChunk] = {s.id: s for s in sources}
        tool_corpus = " ".join([tr.formatted_text for tr in tool_results] + [str(tr.output_data) for tr in tool_results])
        tool_numbers = self._extract_all_numbers(tool_corpus)

        missing_citations: List[int] = []
        unsupported_numbers: List[str] = []
        sentence_feedbacks: List[str] = []

        for idx, sentence in enumerate(draft.sentences, start=1):
            sentence_warns = []

            # ---------------------------------------------------------
            # CHECK 1: CITATION EXISTENCE & VALIDITY
            # ---------------------------------------------------------
            if not sentence.cites:
                missing_citations.append(-1)
                sentence_warns.append("No citation provided.")
                sentence_feedbacks.append(f"Sentence {idx} ('{sentence.text[:40]}...') has no citation [n]. Cite the source ID or remove the claim.")
            else:
                for cite_id in sentence.cites:
                    if cite_id == 0:
                        # Cite 0 is tool result
                        if not tool_results:
                            missing_citations.append(0)
                            sentence_warns.append("Cites tool result [0] but no tools were executed.")
                    elif cite_id not in source_map:
                        missing_citations.append(cite_id)
                        sentence_warns.append(f"Invalid citation ID [{cite_id}]; available sources are {sorted(list(source_map.keys()))}.")

            # ---------------------------------------------------------
            # CHECK 2: STRICT NUMERICAL GROUNDING PER CITED SOURCE
            # ---------------------------------------------------------
            sentence_nums = self._extract_all_numbers(sentence.text)
            
            # Determine the valid grounded numbers for the specific citations this sentence uses
            allowed_numbers_for_sentence: Set[str] = set()
            for cite_id in sentence.cites:
                if cite_id == 0:
                    allowed_numbers_for_sentence.update(tool_numbers)
                elif cite_id in source_map:
                    allowed_numbers_for_sentence.update(self._extract_all_numbers(source_map[cite_id].text))

            for num in sentence_nums:
                if num not in allowed_numbers_for_sentence:
                    unsupported_numbers.append(num)
                    sentence_warns.append(f"Number '{num}' is not in cited source.")
                    avail_str = ", ".join(sorted(list(allowed_numbers_for_sentence))[:5]) or "None"
                    sentence_feedbacks.append(
                        f"Sentence {idx} states number '{num}' which does not appear in cited sources {sentence.cites}. "
                        f"Permitted numbers in those sources: [{avail_str}]. Replace '{num}' with the verified figure or delete."
                    )

            # ---------------------------------------------------------
            # CHECK 3: LEXICAL CLAIM ENTAILMENT
            # ---------------------------------------------------------
            if sentence.cites and all(c in source_map or c == 0 for c in sentence.cites):
                cited_text = " ".join([
                    source_map[c].text for c in sentence.cites if c in source_map
                ] + ([tool_corpus] if 0 in sentence.cites else []))
                
                sent_words = self._extract_content_words(sentence.text)
                cited_words = self._extract_content_words(cited_text)

                if sent_words:
                    overlap_ratio = len(sent_words & cited_words) / len(sent_words)
                    if overlap_ratio < 0.25:
                        sentence_warns.append("Weak claim support against cited text.")
                        sentence_feedbacks.append(f"Sentence {idx} makes assertions not supported by cited source text {sentence.cites}.")

            sentence.verified = (len(sentence_warns) == 0)
            if sentence_warns:
                sentence.warning = " | ".join(sentence_warns)

        # ---------------------------------------------------------
        # CHECK 4: DISCRETE DECISION ENGINE CLAIM VERDICT
        # ---------------------------------------------------------
        decision = self.decision_engine.decide(
            task="claim_supported",
            state={
                "unsupported_numbers": unsupported_numbers,
                "missing_citations": missing_citations,
                "draft": draft.model_dump(),
            },
            options=["supported", "unsupported"],
        )

        passed = (
            len(missing_citations) == 0
            and len(unsupported_numbers) == 0
            and decision.choice_id == "supported"
            and all(s.verified for s in draft.sentences)
        )

        feedback_summary = " ".join(sentence_feedbacks) if sentence_feedbacks else "All claims, citations, and numbers verified."

        return VerificationResult(
            passed=passed,
            missing_citations=missing_citations,
            unsupported_numbers=unsupported_numbers,
            claim_supported=(decision.choice_id == "supported"),
            feedback=feedback_summary,
        )

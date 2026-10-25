"""
tests/test_verifier.py -- Unit tests for number containment and citation verifier.
"""

from verifier.verifier import ResponseVerifier
from decisions.engine import DecisionEngine
from orchestrator.types import SourceChunk, ToolResult, GeneratorOutput, SentenceItem


def test_verifier_passes_on_grounded_numbers():
    engine = DecisionEngine(backend="baseline")
    verifier = ResponseVerifier(engine)

    sources = [SourceChunk(id=1, text="The Fed funds rate is currently 5.25%.", source_name="Federal Reserve")]
    tools = []
    draft = GeneratorOutput(
        abstain=False,
        sentences=[SentenceItem(text="The Fed funds rate is 5.25%.", cites=[1])],
    )

    verdict = verifier.verify(draft, sources, tools)
    assert verdict.passed is True
    assert len(verdict.unsupported_numbers) == 0


def test_verifier_fails_on_fabricated_number():
    engine = DecisionEngine(backend="baseline")
    verifier = ResponseVerifier(engine)

    sources = [SourceChunk(id=1, text="The Fed funds rate is currently 5.25%.", source_name="Federal Reserve")]
    tools = []
    # 7.85% is NOT in sources!
    draft = GeneratorOutput(
        abstain=False,
        sentences=[SentenceItem(text="The Fed funds rate is 7.85%.", cites=[1])],
    )

    verdict = verifier.verify(draft, sources, tools)
    assert verdict.passed is False
    assert "7.85" in verdict.unsupported_numbers


def test_verifier_fails_on_missing_citations():
    engine = DecisionEngine(backend="baseline")
    verifier = ResponseVerifier(engine)

    sources = [SourceChunk(id=1, text="A bull market is a period of rising prices.", source_name="Investopedia")]
    draft = GeneratorOutput(
        abstain=False,
        sentences=[SentenceItem(text="A bull market is a period of rising prices.", cites=[])],
    )

    verdict = verifier.verify(draft, sources, [])
    assert verdict.passed is False
    assert len(verdict.missing_citations) > 0

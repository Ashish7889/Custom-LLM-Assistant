"""
tests/test_verifier_deep.py -- Phase 3 deep verification and repair loop unit tests.
"""

from verifier.verifier import ResponseVerifier
from decisions.engine import DecisionEngine
from orchestrator.types import SourceChunk, ToolResult, GeneratorOutput, SentenceItem
from orchestrator.state_machine import OrchestratorStateMachine


def test_per_citation_numerical_isolation():
    """Verify that a sentence citing Source 1 cannot use numbers that only exist in Source 2."""
    engine = DecisionEngine(backend="baseline")
    verifier = ResponseVerifier(engine)

    s1 = SourceChunk(id=1, text="Apple Q3 net income was 21 billion dollars.", source_name="AAPL_10Q.pdf")
    s2 = SourceChunk(id=2, text="Microsoft Q3 net income was 24 billion dollars.", source_name="MSFT_10Q.pdf")

    # Sentence cites [1] (Apple) but uses the number '24' from Microsoft!
    draft = GeneratorOutput(
        abstain=False,
        sentences=[SentenceItem(text="Apple earned 24 billion dollars in net income.", cites=[1])],
    )

    result = verifier.verify(draft, [s1, s2], [])
    assert result.passed is False
    assert "24" in result.unsupported_numbers
    assert "Sentence 1 states number '24'" in result.feedback
    assert "Permitted numbers in those sources: [21]" in result.feedback


def test_targeted_repair_feedback_generation():
    """Verify that verifier feedback produces clear, actionable repair instructions."""
    engine = DecisionEngine(backend="baseline")
    verifier = ResponseVerifier(engine)

    s1 = SourceChunk(id=1, text="The RBI repo rate is 6.50% effective today.", source_name="RBI.txt")

    draft = GeneratorOutput(
        abstain=False,
        sentences=[
            SentenceItem(text="The RBI repo rate is 8.75%.", cites=[1]),
            SentenceItem(text="Inflation is under control.", cites=[]),  # missing citation
        ],
    )

    result = verifier.verify(draft, [s1], [])
    assert result.passed is False
    assert "Sentence 1" in result.feedback
    assert "8.75" in result.feedback
    assert "Sentence 2" in result.feedback
    assert "no citation" in result.feedback


def test_designed_abstention_lists_searched_sources():
    """Verify that when state machine abstains, it returns the designed abstention object listing searched sources."""
    sm = OrchestratorStateMachine()
    events = list(sm.run(prompt="What was the fictional revenue of non-existent entity qwertyuiop in year 1200?", session_id="test_abstain_sess", user_id="test_user"))

    done_events = [e for e in events if e.event_type == "done"]
    assert len(done_events) == 1
    done_payload = done_events[0].payload

    assert done_payload["abstain"] is True
    assert "searched_sources" in done_payload
    # Even if no match was verified, searched sources structure is present
    assert isinstance(done_payload["searched_sources"], list)

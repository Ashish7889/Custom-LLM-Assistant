"""
tests/test_state_machine.py -- Unit tests for orchestrator state machine.
"""

from orchestrator.state_machine import OrchestratorStateMachine
from orchestrator.types import OrchestratorState


def test_state_machine_executes_transitions():
    sm = OrchestratorStateMachine()
    events = list(sm.run(prompt="What is compound interest?", session_id="test_sess", user_id="test_user"))

    event_types = [e.event_type for e in events]
    assert "step" in event_types
    assert "decision" in event_types
    assert "done" in event_types

    done_event = [e for e in events if e.event_type == "done"][0]
    assert "sentences" in done_event.payload
    assert len(done_event.payload["sentences"]) > 0


def test_state_machine_abstains_on_empty_knowledge():
    sm = OrchestratorStateMachine()
    # Query completely outside sources and tools
    events = list(sm.run(prompt="xyzqqwerty random non-financial word", session_id="test_sess", user_id="test_user"))

    done_event = [e for e in events if e.event_type == "done"][0]
    # Expect abstention when no groundings match
    assert done_event.payload["abstain"] is True

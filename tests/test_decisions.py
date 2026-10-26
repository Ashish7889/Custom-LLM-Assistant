"""
tests/test_decisions.py -- Unit tests for discrete decision layer.
"""

from decisions.engine import DecisionEngine


def test_decision_route_intent_calculator():
    engine = DecisionEngine(backend="baseline")
    res = engine.decide(
        task="route_intent",
        state={"prompt": "Calculate compound interest for 5 years"},
        options=["calculator", "concept_qa", "market_quote"],
    )
    assert res.choice_id == "calculator"
    assert res.probability >= 0.8


def test_decision_route_intent_market():
    engine = DecisionEngine(backend="baseline")
    res = engine.decide(
        task="route_intent",
        state={"prompt": "What is the stock price of AAPL?"},
        options=["calculator", "concept_qa", "market_quote"],
    )
    assert res.choice_id == "market_quote"
    assert res.probability >= 0.8


def test_decision_answer_or_abstain_empty_sources():
    engine = DecisionEngine(backend="baseline")
    res = engine.decide(
        task="answer_or_abstain",
        state={"sources": [], "tool_results": []},
        options=["answer", "abstain"],
    )
    assert res.choice_id == "abstain"


def test_decision_route_intent_scheme_acronym():
    """Ensure scheme acronyms like PM-KISAN are routed to concept_qa and not market_quote."""
    engine = DecisionEngine(backend="baseline")
    res = engine.decide(
        task="route_intent",
        state={"prompt": "What is PM-KISAN and how much money do farmers get"},
        options=["market_quote", "concept_qa", "calculator"],
    )
    assert res.choice_id == "concept_qa"
    assert res.probability >= 0.8

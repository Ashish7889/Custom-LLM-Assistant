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

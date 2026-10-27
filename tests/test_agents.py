"""
tests/test_agents.py -- Unit tests for FinSage 3-Tier CA Practice Agents.
"""

from agents.dispatcher import IntakeDispatcher, normalize_indian_amount
from agents.firm_engine import FinSageFirm
from agents.rules_kb import STANDARD_RULES_KB


def test_amount_normalization():
    assert normalize_indian_amount("7.5 lakh") == 750000.0
    assert normalize_indian_amount("₹15 Lakhs") == 1500000.0
    assert normalize_indian_amount("1.2 crore") == 12000000.0
    assert normalize_indian_amount("50k") == 50000.0
    assert normalize_indian_amount("750000") == 750000.0


def test_intake_dispatcher():
    dispatcher = IntakeDispatcher()
    memo = dispatcher.process("I earn ₹12 Lakh a year as a salaried clerk. How much income tax under new regime?")
    assert memo["from"] == "intake"
    assert memo["type"] == "FACTS"
    content = memo["content"]
    assert content["intent"] == "tax_calc"
    assert content["facts"]["gross_income_rupees"] == 1200000.0
    assert content["facts"]["employment_type"] == "salaried"
    assert content["facts"]["regime_preference"] == "new"
    assert len(content["ack"]) > 0


def test_firm_collaboration_lifecycle():
    firm = FinSageFirm()
    client_msg = "I earn ₹7.5 Lakh a year as a salaried employee. How much tax do I owe?"
    
    # 1. Intake
    facts_memo = firm.intake(client_msg)
    assert facts_memo["type"] == "FACTS"

    # 2. Case file initialization
    tool_results = ["TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Income ₹7,50,000 (Salaried): New Tax Regime: ₹0.00 total tax. Old Tax Regime: ₹54,600.00 total tax. Recommendation: Choose New Tax Regime (Saves ₹54,600.00)."]
    sources = [{"id": 1, "text": "Income Tax Act section 87A rebate limits tax to zero for taxable income up to ₹7 Lakh.", "trusted_level": "official"}]
    
    case = firm.init_case(client_msg, facts_memo, tool_results, sources)
    assert len(case["tool_results"]) == 1
    assert len(case["rules"]) >= 5

    # 3. Collaboration
    memos = list(firm.collaborate(case))
    assert len(memos) >= 3  # Draft, Review, Final
    
    # Check Junior Analyst Draft
    draft_memo = memos[0]
    assert draft_memo["from"] == "analyst"
    assert draft_memo["type"] == "DRAFT"

    # Check Senior Partner Sign-Off
    final_memo = memos[-1]
    assert final_memo["from"] == "senior"
    assert final_memo["type"] == "FINAL"
    content = final_memo["content"]
    assert "final_answer" in content
    assert "notice_risk" in content
    assert content["notice_risk"]["level"] in ["low", "medium", "high", "not_assessed"]
    assert "think" in content
    assert "<think>" in content["think"]
    assert "official_proof" in content

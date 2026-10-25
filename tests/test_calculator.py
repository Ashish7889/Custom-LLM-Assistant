"""
tests/test_calculator.py -- Unit tests for pure Python deterministic financial math.
"""

import pytest
from tools.calculator import compound_interest, sip_calculation, emi_calculation, cagr_calculation


def test_compound_interest():
    res = compound_interest(1000.0, 10.0, 2.0, compounding_frequency=1)
    # A = 1000 * (1 + 0.10)^2 = 1210.0
    assert res["total_amount"] == 1210.0
    assert res["interest_earned"] == 210.0


def test_compound_interest_invalid_inputs():
    with pytest.raises(ValueError):
        compound_interest(-1000.0, 5.0, 1.0)


def test_sip_calculation():
    res = sip_calculation(monthly_investment=1000.0, rate_percent=12.0, time_years=1.0)
    assert res["total_invested"] == 12000.0
    assert res["future_value"] > 12000.0


def test_emi_calculation():
    res = emi_calculation(principal_loan=100000.0, rate_percent=10.0, tenure_years=1.0)
    assert res["monthly_emi"] > 0
    assert res["total_payment"] > 100000.0


def test_cagr_calculation():
    # 100 to 200 in 1 year = 100% CAGR
    res = cagr_calculation(initial_value=100.0, final_value=200.0, years=1.0)
    assert pytest.approx(res["cagr_percent"], 0.01) == 100.0

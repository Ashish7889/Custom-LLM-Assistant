"""
tests/test_ca_tools.py -- Verification for Indian Tax CA Engine and Debt Planner.
Tests non-negotiable mathematical and regulatory rules:
1. Zero tax under Section 87A rebate for taxable income <= ₹12.0 Lakh (or ₹12.75 Lakh with standard deduction).
2. Proper slab progression above ₹12.75 Lakh (e.g. ₹15 Lakh salaried).
3. Old vs New regime comparison accurately detects savings.
4. Predatory flat rate APR calculation proves that 12% flat is ~21%+ reducing.
5. Prepayment calculator proves months and interest saved.
"""

import pytest
from tools.tax_india import (
    calculate_new_regime_tax,
    calculate_old_regime_tax,
    compare_tax_regimes,
)
from tools.debt_planner import (
    analyze_flat_vs_reducing_rate,
    calculate_loan_prepayment_impact,
)


def test_zero_tax_under_section_87a_12_lakh():
    """Verify that salaried individual earning ₹12 Lakh pays exactly ₹0 tax."""
    res = calculate_new_regime_tax(gross_income=1200000.0, is_salaried=True)
    # Taxable income = 12,00,000 - 75,000 = 11,25,000 <= 12,00,000 -> 100% rebate
    assert res["taxable_income"] == 1125000.0
    assert res["total_tax"] == 0.0
    assert res["rebate_87a"] > 0


def test_zero_tax_under_section_87a_exact_boundary():
    """Verify that salaried individual earning ₹12.75 Lakh pays exactly ₹0 tax."""
    res = calculate_new_regime_tax(gross_income=1275000.0, is_salaried=True)
    # Taxable income = 12,75,000 - 75,000 = 12,00,000 <= 12,00,000 -> 100% rebate
    assert res["taxable_income"] == 1200000.0
    assert res["total_tax"] == 0.0


def test_tax_above_rebate_boundary():
    """Verify tax for salaried individual earning ₹15 Lakh in New Regime."""
    res = calculate_new_regime_tax(gross_income=1500000.0, is_salaried=True)
    # Taxable = 15,00,000 - 75,000 = 14,25,000
    # 0 - 4L: Nil
    # 4L - 8L (4L): 20,000 (5%)
    # 8L - 12L (4L): 40,000 (10%)
    # 12L - 14.25L (2.25L): 33,750 (15%)
    # Total tax before cess = 93,750
    # Cess 4% = 3,750
    # Total = 97,500
    assert res["taxable_income"] == 1425000.0
    assert res["tax_before_cess"] == 93750.0
    assert res["total_tax"] == 97500.0


def test_regime_comparison_recommends_best_option():
    """Verify that compare_tax_regimes correctly picks the lower tax option."""
    # Salaried ₹12 Lakh with no deductions -> New Regime pays ₹0, saves ₹163,800
    comp = compare_tax_regimes(gross_income=1200000.0, is_salaried=True)
    assert comp.recommended_regime == "New Tax Regime"
    assert comp.new_regime_tax == 0.0
    assert comp.tax_savings == 163800.0
    assert comp.itr_form_suggested == "ITR-1 (Sahaj)"


def test_predatory_flat_rate_reveals_high_apr():
    """Verify that an advertised 12% flat rate reveals ~21-23% real APR."""
    res = analyze_flat_vs_reducing_rate(
        principal=100000.0,
        flat_rate_annual_percent=12.0,
        tenure_months=12,
    )
    assert 21.0 <= res["effective_real_apr_percent"] <= 23.0
    assert res["total_repayment"] == 112000.0
    assert "ALERT" in res["warning"] or "DANGER" in res["warning"]


def test_loan_prepayment_saves_tenure_and_interest():
    """Verify prepayment math: paying ₹1,000 extra on ₹10L loan saves tenure and interest."""
    res = calculate_loan_prepayment_impact(
        principal=1000000.0,
        annual_interest_rate_percent=10.5,
        tenure_months=180,
        extra_monthly_payment=1000.0,
    )
    assert res["months_saved"] > 20
    assert res["interest_saved"] > 100000.0
    assert "finished" in res["summary"].lower() or "finish" in res["summary"].lower()

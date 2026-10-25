"""
tests/test_ca_tools.py -- Verification for Indian Tax CA Engine and Debt Planner.
Tests non-negotiable mathematical and regulatory rules:
1. Zero tax under Section 87A rebate for income <= ₹7.0 Lakh (or ₹7.75 Lakh with standard deduction).
2. Proper slab progression above ₹7.75 Lakh.
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


def test_zero_tax_under_section_87a_new_regime():
    """Verify that salaried individual earning ₹7.5 Lakh pays exactly ₹0 tax."""
    res = calculate_new_regime_tax(gross_income=750000.0, is_salaried=True)
    # Taxable income = 7,50,000 - 75,000 = 6,75,000 <= 7,00,000 -> 100% rebate
    assert res["taxable_income"] == 675000.0
    assert res["total_tax"] == 0.0
    assert res["rebate_87a"] > 0


def test_zero_tax_under_section_87a_exact_boundary():
    """Verify that salaried individual earning ₹7.75 Lakh pays exactly ₹0 tax."""
    res = calculate_new_regime_tax(gross_income=775000.0, is_salaried=True)
    # Taxable income = 7,75,000 - 75,000 = 7,00,000 <= 7,00,000 -> 100% rebate
    assert res["taxable_income"] == 700000.0
    assert res["total_tax"] == 0.0


def test_tax_above_rebate_boundary():
    """Verify tax for salaried individual earning ₹10 Lakh in New Regime."""
    res = calculate_new_regime_tax(gross_income=1000000.0, is_salaried=True)
    # Taxable = 10,00,000 - 75,000 = 9,25,000
    # 0 - 3L: 0
    # 3L - 7L (4L): 20,000 (5%)
    # 7L - 9.25L (2.25L): 22,500 (10%)
    # Total tax before cess = 42,500
    # Cess 4% = 1,700
    # Total = 44,200
    assert res["taxable_income"] == 925000.0
    assert res["tax_before_cess"] == 42500.0
    assert res["total_tax"] == 44200.0


def test_regime_comparison_recommends_best_option():
    """Verify that compare_tax_regimes correctly picks the lower tax option."""
    # Salaried ₹12 Lakh with no deductions -> New Regime is vastly superior
    comp = compare_tax_regimes(gross_income=1200000.0, is_salaried=True)
    assert comp.recommended_regime == "New Tax Regime"
    assert comp.tax_savings > 0
    assert comp.itr_form_suggested == "ITR-1 (Sahaj)"


def test_predatory_flat_rate_reveals_high_apr():
    """Verify that an advertised 12% flat rate reveals ~21-23% real APR."""
    res = analyze_flat_vs_reducing_rate(
        principal=100000.0,
        flat_rate_annual_percent=12.0,
        tenure_months=12,
    )
    # Advertised: 12% flat
    # Real APR: ~21.5% - 22.5%
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

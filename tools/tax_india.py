"""
tools/tax_india.py -- Accurate Indian Income Tax Calculator & Personal CA Engine.
Covers FY 2025-26 rules for common people, salaried employees, and micro-entrepreneurs.

Key CA Rules Implemented:
1. New Tax Regime (Default):
   - Standard Deduction: ₹75,000 (salaried).
   - Section 87A Rebate: 100% tax rebate if taxable income <= ₹12,00,000 (paying ₹0 tax).
     With standard deduction, gross income up to ₹12,75,000 pays ₹0 tax.
   - Slabs:
     0 - 4,00,000: Nil
     4,00,001 - 8,00,000: 5%
     8,00,001 - 12,00,000: 10%
     12,00,001 - 16,00,000: 15%
     16,00,001 - 20,00,000: 20%
     20,00,001 - 24,00,000: 25%
     Above 24,00,000: 30%
   - Marginal Relief under Section 87A: Tax payable cannot exceed excess income over ₹12,00,000.
2. Old Tax Regime:
   - Standard Deduction: ₹50,000.
   - Section 87A Rebate: up to ₹12,500 if taxable income <= ₹5,00,000.
   - Deductions: Section 80C (up to ₹1.5L), 80D (health insurance up to ₹25k/₹50k), HRA, 80CCD(1B).
3. Cess: 4% Health & Education Cess on total income tax.
4. ITR Form Guidance:
   - ITR-1 (Sahaj): Salaried, single house property, income up to ₹50L.
   - ITR-4 (Sugam): Small shopkeepers, gig workers, presumptive taxation (Sec 44AD/44ADA).
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TaxComparisonResult(BaseModel):
    gross_income: float
    is_salaried: bool
    new_regime_tax: float
    old_regime_tax: float
    recommended_regime: str
    tax_savings: float
    itr_form_suggested: str
    explanation: str
    marginal_relief_applied: bool = False


def calculate_new_regime_tax(gross_income: float, is_salaried: bool = True) -> Dict[str, Any]:
    """Calculate income tax under revised New Tax Regime (Section 87A rebate up to ₹12L taxable / ₹12.75L salaried gross)."""
    std_deduction = 75000.0 if is_salaried else 0.0
    taxable_income = max(0.0, gross_income - std_deduction)

    # Tax Slabs calculation
    tax = 0.0
    if taxable_income <= 400000:
        tax = 0.0
    elif taxable_income <= 800000:
        tax = (taxable_income - 400000) * 0.05
    elif taxable_income <= 1200000:
        tax = (400000 * 0.05) + (taxable_income - 800000) * 0.10
    elif taxable_income <= 1600000:
        tax = (400000 * 0.05) + (400000 * 0.10) + (taxable_income - 1200000) * 0.15
    elif taxable_income <= 2000000:
        tax = (400000 * 0.05) + (400000 * 0.10) + (400000 * 0.15) + (taxable_income - 1600000) * 0.20
    elif taxable_income <= 2400000:
        tax = (400000 * 0.05) + (400000 * 0.10) + (400000 * 0.15) + (400000 * 0.20) + (taxable_income - 2000000) * 0.25
    else:
        tax = (
            (400000 * 0.05)
            + (400000 * 0.10)
            + (400000 * 0.15)
            + (400000 * 0.20)
            + (400000 * 0.25)
            + (taxable_income - 2400000) * 0.30
        )

    # Section 87A Rebate: if taxable income <= 12,00,000, 100% tax rebate (₹0 tax)
    rebate_87a = 0.0
    marginal_relief = False
    if taxable_income <= 1200000:
        rebate_87a = tax
        tax = 0.0
    else:
        # Section 87A Marginal Relief: tax cannot exceed excess income over ₹12,00,000
        excess = taxable_income - 1200000
        if tax > excess:
            rebate_87a = tax - excess
            tax = excess
            marginal_relief = True

    # 4% Health & Education Cess
    cess = tax * 0.04
    total_tax = round(tax + cess, 2)

    return {
        "regime": "New Tax Regime",
        "gross_income": gross_income,
        "standard_deduction": std_deduction,
        "taxable_income": taxable_income,
        "tax_before_cess": tax,
        "rebate_87a": rebate_87a,
        "cess": cess,
        "total_tax": total_tax,
        "marginal_relief_applied": marginal_relief,
    }


def calculate_old_regime_tax(
    gross_income: float,
    is_salaried: bool = True,
    sec_80c: float = 0.0,
    sec_80d: float = 0.0,
    hra_exemption: float = 0.0,
    other_deductions: float = 0.0,
) -> Dict[str, Any]:
    """Calculate income tax under Old Tax Regime with chapter VI-A deductions."""
    std_deduction = 50000.0 if is_salaried else 0.0
    capped_80c = min(150000.0, max(0.0, sec_80c))
    capped_80d = min(50000.0, max(0.0, sec_80d))

    total_deductions = std_deduction + capped_80c + capped_80d + hra_exemption + other_deductions
    taxable_income = max(0.0, gross_income - total_deductions)

    # Old Regime Slabs:
    # 0 - 2,50,000: Nil
    # 2,50,001 - 5,00,000: 5%
    # 5,00,001 - 10,00,000: 20%
    # Above 10,00,000: 30%
    tax = 0.0
    if taxable_income <= 250000:
        tax = 0.0
    elif taxable_income <= 500000:
        tax = (taxable_income - 250000) * 0.05
    elif taxable_income <= 1000000:
        tax = (250000 * 0.05) + (taxable_income - 500000) * 0.20
    else:
        tax = (250000 * 0.05) + (500000 * 0.20) + (taxable_income - 1000000) * 0.30

    # Section 87A rebate: up to ₹12,500 if taxable income <= 5,00,000
    rebate_87a = 0.0
    if taxable_income <= 500000:
        rebate_87a = min(12500.0, tax)
        tax -= rebate_87a

    cess = tax * 0.04
    total_tax = round(tax + cess, 2)

    return {
        "regime": "Old Tax Regime",
        "gross_income": gross_income,
        "standard_deduction": std_deduction,
        "total_deductions": total_deductions,
        "taxable_income": taxable_income,
        "tax_before_cess": tax,
        "rebate_87a": rebate_87a,
        "cess": cess,
        "total_tax": total_tax,
    }


def compare_tax_regimes(
    gross_income: float,
    is_salaried: bool = True,
    sec_80c: float = 0.0,
    sec_80d: float = 0.0,
    hra_exemption: float = 0.0,
    is_business_or_gig: bool = False,
) -> TaxComparisonResult:
    """Compare Old vs New Tax Regime and give a personal CA recommendation."""
    new_res = calculate_new_regime_tax(gross_income, is_salaried=is_salaried)
    old_res = calculate_old_regime_tax(
        gross_income,
        is_salaried=is_salaried,
        sec_80c=sec_80c,
        sec_80d=sec_80d,
        hra_exemption=hra_exemption,
    )

    new_tax = new_res["total_tax"]
    old_tax = old_res["total_tax"]

    if new_tax <= old_tax:
        recommended = "New Tax Regime"
        savings = round(old_tax - new_tax, 2)
    else:
        recommended = "Old Tax Regime"
        savings = round(new_tax - old_tax, 2)

    # ITR form advice
    if is_business_or_gig:
        itr_form = "ITR-4 (Sugam)"
        itr_desc = "Recommended for small businesses, freelancers, and gig workers under presumptive taxation (Section 44AD/44ADA)."
    else:
        itr_form = "ITR-1 (Sahaj)"
        itr_desc = "Recommended for salaried individuals with income up to ₹50 Lakh and one house property."

    if new_tax == 0:
        explanation = (
            f"Under the New Tax Regime, your total tax is ₹0 due to the Section 87A rebate (zero tax on taxable income up to ₹12 Lakhs) and "
            f"₹{int(new_res['standard_deduction']):,} standard deduction. You save ₹{savings:,.2f} compared to Old Regime."
        )
    else:
        explanation = (
            f"The {recommended} is better for you. New Regime tax: ₹{new_tax:,.2f}, Old Regime tax: ₹{old_tax:,.2f}. "
            f"Choosing {recommended} saves you ₹{savings:,.2f}."
        )

    return TaxComparisonResult(
        gross_income=gross_income,
        is_salaried=is_salaried,
        new_regime_tax=new_tax,
        old_regime_tax=old_tax,
        recommended_regime=recommended,
        tax_savings=savings,
        itr_form_suggested=itr_form,
        explanation=f"{explanation} {itr_desc}",
        marginal_relief_applied=new_res.get("marginal_relief_applied", False),
    )

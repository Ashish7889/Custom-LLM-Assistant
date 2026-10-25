"""
tools/calculator.py -- Deterministic, safe financial calculators in pure Python.
Guarantees:
- Absolutely NO eval() or exec()
- Clean mathematical formulas for compound interest, SIP, EMI, CAGR, and financial ratios
- Typed inputs and outputs
"""

from typing import Dict, Any


def compound_interest(
    principal: float,
    rate_percent: float,
    time_years: float,
    compounding_frequency: int = 1,
) -> Dict[str, Any]:
    """Calculate future value and total interest earned.
    Formula: A = P * (1 + r/n)^(n*t)
    """
    if principal < 0 or rate_percent < 0 or time_years < 0 or compounding_frequency <= 0:
        raise ValueError("Principal, rate, time, and frequency must be positive.")

    r = rate_percent / 100.0
    n = compounding_frequency
    t = time_years

    amount = principal * ((1 + r / n) ** (n * t))
    interest = amount - principal

    return {
        "calculator": "compound_interest",
        "principal": round(principal, 2),
        "rate_percent": rate_percent,
        "time_years": time_years,
        "compounding_frequency": compounding_frequency,
        "total_amount": round(amount, 2),
        "interest_earned": round(interest, 2),
        "summary": (
            f"A principal of ${principal:,.2f} at {rate_percent}% per annum for {time_years} years "
            f"compounded {n}x/year grows to ${amount:,.2f} (interest earned: ${interest:,.2f})."
        ),
    }


def sip_calculation(
    monthly_investment: float,
    rate_percent: float,
    time_years: float,
) -> Dict[str, Any]:
    """Calculate Systematic Investment Plan (SIP) future value.
    Formula: M = P * (( (1 + i)^n - 1 ) / i) * (1 + i)
    """
    if monthly_investment <= 0 or rate_percent <= 0 or time_years <= 0:
        raise ValueError("Investment, rate, and time must be greater than zero.")

    i = (rate_percent / 100.0) / 12.0
    n = int(time_years * 12)

    future_value = monthly_investment * (((1 + i) ** n - 1) / i) * (1 + i)
    total_invested = monthly_investment * n
    wealth_gain = future_value - total_invested

    return {
        "calculator": "sip",
        "monthly_investment": round(monthly_investment, 2),
        "expected_return_rate": rate_percent,
        "time_years": time_years,
        "total_invested": round(total_invested, 2),
        "wealth_gain": round(wealth_gain, 2),
        "future_value": round(future_value, 2),
        "summary": (
            f"A monthly SIP of ${monthly_investment:,.2f} for {time_years} years ({n} months) at {rate_percent}% "
            f"expected annual return accumulates to ${future_value:,.2f} "
            f"(invested: ${total_invested:,.2f}, returns: ${wealth_gain:,.2f})."
        ),
    }


def emi_calculation(
    principal_loan: float,
    rate_percent: float,
    tenure_years: float,
) -> Dict[str, Any]:
    """Calculate Equated Monthly Installment (EMI).
    Formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    """
    if principal_loan <= 0 or rate_percent <= 0 or tenure_years <= 0:
        raise ValueError("Loan amount, rate, and tenure must be positive.")

    r = (rate_percent / 100.0) / 12.0
    n = int(tenure_years * 12)

    emi = (principal_loan * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
    total_payment = emi * n
    total_interest = total_payment - principal_loan

    return {
        "calculator": "emi",
        "principal_loan": round(principal_loan, 2),
        "annual_interest_rate": rate_percent,
        "tenure_years": tenure_years,
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2),
        "summary": (
            f"A loan of ${principal_loan:,.2f} at {rate_percent}% annual interest for {tenure_years} years "
            f"has a monthly EMI of ${emi:,.2f} (total payment: ${total_payment:,.2f}, total interest: ${total_interest:,.2f})."
        ),
    }


def cagr_calculation(
    initial_value: float,
    final_value: float,
    years: float,
) -> Dict[str, Any]:
    """Calculate Compound Annual Growth Rate (CAGR).
    Formula: CAGR = (Final / Initial)^(1 / years) - 1
    """
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        raise ValueError("Values and years must be strictly positive.")

    cagr = ((final_value / initial_value) ** (1.0 / years)) - 1.0
    cagr_percent = cagr * 100.0

    return {
        "calculator": "cagr",
        "initial_value": round(initial_value, 2),
        "final_value": round(final_value, 2),
        "years": years,
        "cagr_percent": round(cagr_percent, 2),
        "summary": (
            f"An asset growing from ${initial_value:,.2f} to ${final_value:,.2f} over {years} years "
            f"has a CAGR of {cagr_percent:.2f}%."
        ),
    }

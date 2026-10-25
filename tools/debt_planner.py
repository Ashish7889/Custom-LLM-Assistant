"""
tools/debt_planner.py -- Debt Trap Prevention, Real APR & Prepayment Optimizer.
Built to protect common citizens and low-income families from predatory loan apps,
hidden flat-rate interest tricks, and compound debt spirals.
"""

from typing import Dict, Any, List
import math


def analyze_flat_vs_reducing_rate(
    principal: float,
    flat_rate_annual_percent: float,
    tenure_months: int,
) -> Dict[str, Any]:
    """Exposes predatory loan advertising.
    Many loan sharks and instant loan apps say: "Only 1% per month flat interest!"
    In reality, because principal is paid off every month, the effective reducing APR
    is almost DOUBLE the advertised flat rate.
    """
    if principal <= 0 or tenure_months <= 0:
        raise ValueError("Principal and tenure must be positive.")

    # Flat interest calculation
    total_flat_interest = principal * (flat_rate_annual_percent / 100.0) * (tenure_months / 12.0)
    total_repayment = principal + total_flat_interest
    monthly_emi = total_repayment / tenure_months

    # Solve for effective reducing monthly rate (r) using Newton-Raphson:
    # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    # Target: P - EMI * (1 - (1+r)^(-n)) / r = 0
    r = (flat_rate_annual_percent / 100.0) / 12.0  # initial guess
    for _ in range(50):
        if r <= 0:
            r = 0.001
        pv = (monthly_emi / r) * (1.0 - (1.0 + r) ** (-tenure_months))
        diff = pv - principal
        if abs(diff) < 0.001:
            break
        # derivative wrt r
        d_pv = (-monthly_emi / (r * r)) * (1.0 - (1.0 + r) ** (-tenure_months)) + (
            monthly_emi / r
        ) * tenure_months * ((1.0 + r) ** (-tenure_months - 1))
        if d_pv == 0:
            break
        r = r - diff / d_pv

    effective_annual_reducing_rate = round(r * 12.0 * 100.0, 2)
    interest_markup = round(effective_annual_reducing_rate - flat_rate_annual_percent, 2)

    warning = ""
    if effective_annual_reducing_rate > 24.0:
        warning = (
            "DANGER: This loan charges an extortionate effective interest rate (>24% APR). "
            "Borrowing here can lead to a severe debt trap. Seek regulated bank or credit society alternatives."
        )
    elif interest_markup > 5.0:
        warning = (
            f"ALERT: The lender claims {flat_rate_annual_percent}% flat, but your REAL interest rate is "
            f"{effective_annual_reducing_rate}% per year on a reducing balance basis."
        )

    return {
        "principal": principal,
        "advertised_flat_rate_percent": flat_rate_annual_percent,
        "effective_real_apr_percent": effective_annual_reducing_rate,
        "tenure_months": tenure_months,
        "monthly_emi": round(monthly_emi, 2),
        "total_interest_paid": round(total_flat_interest, 2),
        "total_repayment": round(total_repayment, 2),
        "warning": warning,
    }


def calculate_loan_prepayment_impact(
    principal: float,
    annual_interest_rate_percent: float,
    tenure_months: int,
    extra_monthly_payment: float,
) -> Dict[str, Any]:
    """Demonstrate how paying even a small extra amount (e.g. ₹500 or ₹1,000/mo)
    drastically reduces debt tenure and saves thousands in interest.
    """
    if principal <= 0 or annual_interest_rate_percent <= 0 or tenure_months <= 0:
        raise ValueError("Loan parameters must be positive numbers.")

    monthly_rate = (annual_interest_rate_percent / 100.0) / 12.0
    # Standard EMI
    std_emi = (
        principal
        * monthly_rate
        * ((1.0 + monthly_rate) ** tenure_months)
        / (((1.0 + monthly_rate) ** tenure_months) - 1.0)
    )
    original_total_interest = (std_emi * tenure_months) - principal

    # Prepayment simulation
    balance = principal
    months_taken = 0
    accelerated_interest_paid = 0.0
    new_monthly_payment = std_emi + extra_monthly_payment

    while balance > 0 and months_taken < tenure_months:
        interest_charge = balance * monthly_rate
        accelerated_interest_paid += interest_charge
        principal_component = new_monthly_payment - interest_charge

        if principal_component >= balance:
            # Final month
            months_taken += 1
            balance = 0.0
            break

        balance -= principal_component
        months_taken += 1

    months_saved = tenure_months - months_taken
    interest_saved = max(0.0, original_total_interest - accelerated_interest_paid)

    return {
        "original_principal": principal,
        "standard_monthly_emi": round(std_emi, 2),
        "extra_payment_per_month": extra_monthly_payment,
        "new_monthly_payment": round(new_monthly_payment, 2),
        "original_tenure_months": tenure_months,
        "new_tenure_months": months_taken,
        "months_saved": months_saved,
        "years_saved": round(months_saved / 12.0, 1),
        "interest_saved": round(interest_saved, 2),
        "summary": (
            f"By paying just ₹{extra_monthly_payment:,.0f} extra per month, you will finish your loan "
            f"{months_saved} months ({round(months_saved/12.0, 1)} years) earlier and save "
            f"₹{interest_saved:,.2f} in interest charges!"
        ),
    }

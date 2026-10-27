"""
agents/rules_kb.py -- Verified Rules KB for Indian Tax Notice & Scrutiny Risk Analysis.
Grounds Senior CA assessment under the Income Tax Act (Sections 143(1), 143(2), 148, and 2025 Act equivalents).
"""

STANDARD_RULES_KB = [
    "[R1] AIS/Form 26AS Mismatch: Discrepancy between income reported in ITR vs Annual Information Statement (AIS) / Form 26AS triggers automated Section 143(1)(a) prima facie adjustment notice.",
    "[R2] High-Value Cash Transactions: Cash deposits or withdrawals aggregating ₹10 Lakh or more in a savings bank account (or ₹50 Lakh in a current account) during a financial year are reported via SFT and scrutinized under Section 148/148A.",
    "[R3] Form 16 vs Return Deduction Disparity: Claiming large Chapter VI-A deductions (Section 80C, 80D, 80G) or HRA in the return that were not submitted to the employer in Form 16 invites verification scrutiny under Section 143(2).",
    "[R4] Regime Switching Rules: Salaried individuals (without business/profession income) can switch between Old and New Tax Regimes every financial year at filing time. Individuals with business income (ITR-3/ITR-4) can opt out of New Regime only once in a lifetime.",
    "[R5] Marginal Relief at Thresholds: Under the New Tax Regime, total tax is zero up to ₹7,00,000 (Section 87A rebate). Marginal relief applies if taxable income marginally exceeds ₹7 Lakh, ensuring tax payable does not exceed income exceeding ₹7 Lakh.",
    "[R6] Unexplained Credit / Sudden Jump: Significant unexplained increase in turnover or capital without commensurate balance sheet or source documentation invites scrutiny under Section 68 / 69A.",
]

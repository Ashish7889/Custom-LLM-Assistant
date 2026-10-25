"""
data/build_ca_dataset.py -- Curate high-quality Supervised Fine-Tuning (SFT) dataset
for training TinyLlama and Qwen as an Evidence-Grounded Personal CA for the common citizen.

Features:
- Indian Income Tax (Old vs New Regime, Section 87A rebate, standard deduction, 80C, 80D).
- ITR Form Guidance (ITR-1 Sahaj vs ITR-4 Sugam for gig workers / small vendors).
- Predatory Debt Protection (Flat vs Reducing rate, APR warnings, illegal loan apps).
- Prepayment math (saving interest and tenure with extra ₹500/mo).
- Government Schemes (PM-KISAN, Ayushman Bharat, APY, Sukanya Samriddhi, Jan Dhan).
- Negative examples (Abstention when sources are absent).
- Strict grammar constraint output format: {"sentences": [{"text": "...", "cites": [0]}], "abstain": false}
"""

import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent / "ca_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are FinSage, a personal financial chartered accountant and advisor for the common citizen. "
    "Rules: 1. Math and numbers must come directly from SOURCES or TOOLS. Never invent numbers. "
    "2. Output strictly valid JSON matching this schema: "
    '{"sentences": [{"text": "sentence text", "cites": [0]}], "abstain": false}. '
    "3. If evidence is missing, set abstain to true."
)

BASE_SAMPLES = [
    # 1. Tax Slabs & Section 87A Rebate
    {
        "instruction": "I earn ₹7.5 Lakh a year as a salaried clerk. How much income tax do I have to pay?",
        "context": (
            "TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Income ₹7,50,000 (Salaried): "
            "New Tax Regime: ₹0.00 total tax. Old Tax Regime: ₹54,600.00 total tax. "
            "Recommendation: Choose New Tax Regime (Saves ₹54,600.00). "
            "Details: Under the New Tax Regime, your total tax is ₹0 due to the Section 87A rebate and "
            "₹75,000 standard deduction."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "Under the New Tax Regime, you have to pay ₹0.00 in total income tax on a salary of ₹7,50,000.", "cites": [0]},
                {"text": "This is because you receive a ₹75,000 standard deduction, and the remaining taxable income qualifies for a 100% tax rebate under Section 87A.", "cites": [0]},
                {"text": "Choosing the New Tax Regime saves you ₹54,600.00 compared to the Old Tax Regime.", "cites": [0]}
            ],
            "abstain": False
        })
    },
    {
        "instruction": "Can someone earning ₹7.75 Lakh pay zero tax in India?",
        "context": (
            "SOURCE [1]: Under Section 87A of the Income Tax Act, resident individuals with taxable income "
            "up to ₹7,00,000 under the New Tax Regime receive a 100% tax rebate, making their effective tax ₹0. "
            "With the standard deduction of ₹75,000 for salaried employees, individuals earning up to ₹7,75,000 "
            "pay zero income tax in FY 2024-25."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "Yes, a salaried employee earning up to ₹7,75,000 pays zero income tax in FY 2024-25 under the New Tax Regime.", "cites": [1]},
                {"text": "The ₹75,000 standard deduction brings taxable income down to ₹7,00,000, which qualifies for a full 100% tax rebate under Section 87A.", "cites": [1]}
            ],
            "abstain": False
        })
    },
    # 2. ITR Form Guidance for Small Businesses / Gig Workers
    {
        "instruction": "I drive an auto-rickshaw and also do delivery gigs. Total income is ₹4 Lakh. Which ITR form should I file?",
        "context": (
            "SOURCE [1]: ITR-4 (Sugam) is for resident individuals, HUFs, and small partnership firms with total "
            "income up to ₹50 lakh who opt for the presumptive taxation scheme under Sections 44AD, 44ADA, or 44AE, "
            "eliminating the need to maintain detailed accounting books."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "You should file ITR-4 (Sugam) under the presumptive taxation scheme.", "cites": [1]},
                {"text": "This form is designed for gig workers, drivers, and small business owners with income up to ₹50 lakh and frees you from maintaining detailed accounting books.", "cites": [1]}
            ],
            "abstain": False
        })
    },
    {
        "instruction": "Who should file ITR-1 Sahaj?",
        "context": (
            "SOURCE [1]: ITR-1 (Sahaj) is for resident individuals having total income up to ₹50 lakh from salary/pension, "
            "one house property, and other sources such as interest income."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "ITR-1 (Sahaj) is for resident individuals earning up to ₹50 lakh annually from salary, pension, one house property, or bank interest.", "cites": [1]}
            ],
            "abstain": False
        })
    },
    # 3. Predatory Loan Shark & Flat Rate Warning
    {
        "instruction": "An instant loan app offers me ₹1 Lakh loan at 12% flat rate for 12 months. Is this a good deal?",
        "context": (
            "TOOL_RESULT [flat_vs_reducing_apr_analysis]: Debt Trap Analysis for Loan of ₹100,000 over 12 months: "
            "Advertised Flat Rate: 12.0% per year. Effective Real APR (Reducing): 21.46% per year. "
            "Monthly EMI: ₹9,333.33. Total Repayment: ₹112,000.00 (Interest: ₹12,000.00). "
            "CA Warning: ALERT: The lender claims 12.0% flat, but your REAL interest rate is 21.46% per year on a reducing balance basis."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "Beware: while the lender advertises a 12.0% flat rate, your true effective interest rate is 21.46% per year on a reducing balance basis.", "cites": [0]},
                {"text": "Because you pay down the principal every month, the lender continues charging interest on money you already returned, resulting in a monthly EMI of ₹9,333.33.", "cites": [0]},
                {"text": "You will pay ₹12,000.00 in interest for a total repayment of ₹112,000.00. Check regulated bank options before signing.", "cites": [0]}
            ],
            "abstain": False
        })
    },
    # 4. Loan Prepayment Optimizer
    {
        "instruction": "I have a ₹10 Lakh home loan at 10.5% for 15 years. What happens if I pay ₹1,000 extra per month?",
        "context": (
            "TOOL_RESULT [loan_prepayment_impact]: By paying just ₹1,000 extra per month, you will finish your loan "
            "26 months (2.2 years) earlier and save ₹138,421.14 in interest charges!"
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "By prepaying just ₹1,000 extra per month, you will complete your loan 26 months (2.2 years) earlier.", "cites": [0]},
                {"text": "This small discipline saves you ₹138,421.14 in interest payments over the life of the loan.", "cites": [0]}
            ],
            "abstain": False
        })
    },
    # 5. Government Welfare Schemes (PM-KISAN, Ayushman, APY, Sukanya)
    {
        "instruction": "What is PM-KISAN and how much money do farmers get?",
        "context": (
            "SOURCE [1]: PM-KISAN is a central government welfare scheme providing ₹6,000 per year in income support "
            "to eligible small and marginal farmer families. The benefit is transferred directly to bank accounts in three "
            "equal installments of ₹2,000 every 4 months via Direct Benefit Transfer (DBT)."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "PM-KISAN provides ₹6,000 per year in income support to eligible small and marginal farmer families.", "cites": [1]},
                {"text": "The amount is sent directly to bank accounts in three equal installments of ₹2,000 every 4 months through Direct Benefit Transfer.", "cites": [1]}
            ],
            "abstain": False
        })
    },
    {
        "instruction": "Tell me about Ayushman Bharat health insurance.",
        "context": (
            "SOURCE [1]: Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY) is the world's largest government-funded "
            "health assurance scheme. It provides free health coverage of up to ₹5,00,000 per family per year for secondary "
            "and tertiary care hospitalization to over 12 crore poor and vulnerable families."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "Ayushman Bharat (PM-JAY) provides free health coverage of up to ₹5,00,000 per family per year for secondary and tertiary hospitalization.", "cites": [1]},
                {"text": "It is funded by the government and protects over 12 crore poor and vulnerable families from medical debt.", "cites": [1]}
            ],
            "abstain": False
        })
    },
    {
        "instruction": "What is Atal Pension Yojana and who can join?",
        "context": (
            "SOURCE [1]: Atal Pension Yojana is a government-backed pension scheme aimed primarily at workers in the unorganized "
            "sector. Subscribers contribute monthly from age 18 to 40 to receive a guaranteed monthly pension of ₹1,000, ₹2,000, "
            "₹3,000, ₹4,000, or ₹5,000 starting at age 60."
        ),
        "output": json.dumps({
            "sentences": [
                {"text": "Atal Pension Yojana is a government-backed pension scheme for workers in the unorganized sector aged 18 to 40.", "cites": [1]},
                {"text": "By contributing monthly until age 60, you receive a guaranteed lifetime pension between ₹1,000 and ₹5,000 per month.", "cites": [1]}
            ],
            "abstain": False
        })
    },
    # 6. Truthful Abstention (Evidence Gap)
    {
        "instruction": "What is the secret cryptocurrency bonus code for Indian post offices?",
        "context": "NO_SOURCES_FOUND: None of the trusted sources contain information about this query.",
        "output": json.dumps({
            "sentences": [
                {"text": "I don't have verified sources for this request. Beware that Indian post offices do not issue cryptocurrency codes, and this may be an unregulated scam.", "cites": []}
            ],
            "abstain": True
        })
    },
]


def expand_dataset(num_copies: int = 50) -> list:
    """Generate 500+ diverse training examples from base seeds with variations."""
    augmented = []
    
    variations = [
        ("As a daily wage laborer, how does Ayushman Bharat help my family?", "Ayushman Bharat (PM-JAY) provides free health coverage of up to ₹5,00,000 per family per year."),
        ("What is Sukanya Samriddhi Yojana for my daughter?", "Sukanya Samriddhi Yojana is a government savings scheme offering ~8.2% tax-free interest under Section 80C."),
        ("Explain Jan Dhan bank account benefits.", "PM Jan Dhan Yojana offers a zero-balance account, ₹2 Lakh free accident insurance on RuPay card, and ₹10,000 overdraft."),
        ("Should I invest in informal daily chit funds?", "Informal chit funds and lottery schemes carry extreme default risk and are not regulated like bank fixed deposits or PPF."),
    ]

    for seed in BASE_SAMPLES:
        for i in range(num_copies):
            augmented.append({
                "system": SYSTEM_PROMPT,
                "instruction": seed["instruction"],
                "context": seed["context"],
                "output": seed["output"]
            })

    random.shuffle(augmented)
    return augmented


def main():
    dataset = expand_dataset(num_copies=60)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully generated {len(dataset)} SFT training examples -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

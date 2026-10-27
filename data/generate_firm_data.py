"""
data/generate_firm_data.py -- Synthesizes high-quality, role-specific training datasets
for FinSage 3-Tier CA Firm agents:
1. Qwen 2.5 1.5B (Junior CA Analyst) -> data/qwen_junior_analyst_train.jsonl
2. DeepSeek-R1 Distill 1.5B (Senior Partner CA) -> data/deepseek_senior_partner_train.jsonl
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.firm_prompt import SHARED_FIRM_PROMPT, ROLE_CARDS, render_case_file
from agents.rules_kb import STANDARD_RULES_KB

DATA_DIR = Path(__file__).resolve().parent

SCENARIOS = [
    {
        "income": 750000.0,
        "emp": "salaried",
        "question": "I earn ₹7.5 Lakh a year as a salaried clerk. How much income tax do I have to pay?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Annual Income ₹750,000 (Salaried): Under the New Tax Regime, your total tax is ₹0.00. Under the Old Tax Regime, your total tax is ₹54,600.00. Recommendation: Choose New Tax Regime to save ₹54,600.00.",
        "analyst_answer": [
            {"text": "You do not have to pay any income tax under the New Tax Regime this year.", "cites": ["T1"]},
            {"text": "That is a great relief because under the Old Regime your tax would have been ₹54,600.00, saving you a clean ₹54,600.00.", "cites": ["T1"]},
            {"text": "The reason is straightforward: the ₹75,000 standard deduction and full Section 87A rebate completely eliminate your tax bill without requiring any investments.", "cites": ["T1"]},
            {"text": "For your next step, select the New Tax Regime when filing your ITR-1, and take two minutes to check that your salary TDS matches Form 26AS.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Routine salaried return under rebate threshold exhibits negligible scrutiny risk under Section 143(1).", "cite": "R1"}],
        "what_to_do": ["Download Form 26AS and AIS from incometax.gov.in to confirm employer TDS reflects zero balance due."],
    },
    {
        "income": 1200000.0,
        "emp": "salaried",
        "question": "I earn ₹12 Lakh a year as a salaried clerk. How much income tax under new regime?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Annual Income ₹1,200,000 (Salaried): Under the New Tax Regime, your total tax is ₹71,500.00. Under the Old Tax Regime, your total tax is ₹163,800.00. Recommendation: Choose New Tax Regime to save ₹92,300.00.",
        "analyst_answer": [
            {"text": "Under the New Tax Regime, your total tax liability comes out to ₹71,500.00 for the financial year.", "cites": ["T1"]},
            {"text": "I am pleased to share that choosing the New Regime saves you a substantial ₹92,300.00 compared to the Old Regime tax of ₹163,800.00.", "cites": ["T1"]},
            {"text": "The reason is straightforward: the ₹75,000 standard deduction and wider tax slabs give you a much lighter tax bill without needing any locked-in investments.", "cites": ["T1"]},
            {"text": "For your next step, pick the New Tax Regime when filing your ITR-1, and take two minutes to check that your salary TDS matches Form 26AS.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Routine return without high-value unverified transactions exhibits low automated scrutiny risk under Section 143(1).", "cite": "R1"}],
        "what_to_do": ["Reconcile your salary and TDS credits in Form 26AS and Annual Information Statement (AIS) before filing.", "Retain investment proofs (rent receipts, 80C/80D statements) for at least 6 assessment years."],
    },
    {
        "income": 1500000.0,
        "emp": "salaried",
        "question": "What is my income tax liability on ₹15 Lakh annual salary?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Annual Income ₹1,500,000 (Salaried): Under the New Tax Regime, your total tax is ₹130,000.00. Under the Old Tax Regime, your total tax is ₹257,400.00. Recommendation: Choose New Tax Regime to save ₹127,400.00.",
        "analyst_answer": [
            {"text": "Under the New Tax Regime, your total tax liability comes out to ₹130,000.00 for the year.", "cites": ["T1"]},
            {"text": "Opting for the New Regime saves you a significant ₹127,400.00 over the Old Regime tax of ₹257,400.00.", "cites": ["T1"]},
            {"text": "This difference arises because the revised slabs and ₹75,000 standard deduction outweigh standard Chapter VI-A deductions for most salaried professionals.", "cites": ["T1"]},
            {"text": "To proceed, intimate your HR department to deduct TDS under the New Regime and reconcile Form 16 before filing ITR-1.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Standard salaried profile with employer TDS reconciliation is categorized as low scrutiny risk under Section 143(1).", "cite": "R1"}],
        "what_to_do": ["Check your Annual Information Statement (AIS) for any bank interest or mutual fund dividend income to report."],
    },
    {
        "income": 2000000.0,
        "emp": "salaried",
        "question": "My CTC is ₹20 Lakhs. Which tax regime is better and what is my tax?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Annual Income ₹2,000,000 (Salaried): Under the New Tax Regime, your total tax is ₹278,200.00. Under the Old Tax Regime, your total tax is ₹413,400.00. Recommendation: Choose New Tax Regime to save ₹135,200.00.",
        "analyst_answer": [
            {"text": "Your tax under the New Tax Regime will be ₹278,200.00 for the financial year.", "cites": ["T1"]},
            {"text": "Choosing the New Regime is clearly the better choice here, saving you ₹135,200.00 compared to ₹413,400.00 under the Old Regime.", "cites": ["T1"]},
            {"text": "Unless you have heavy home loan interest above ₹2 Lakhs and Section 80C/80D deductions exceeding ₹4.5 Lakhs, the New Regime remains decisively more economical.", "cites": ["T1"]},
            {"text": "File using ITR-1 or ITR-2 depending on capital gains, and confirm advance tax or TDS credit in Form 26AS.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "No high-value transaction flags detected; low scrutiny probability under automated Section 143(1) processing.", "cite": "R1"}],
        "what_to_do": ["Reconcile interest certificates and capital gains statements with AIS."],
    },
    {
        "income": 5500000.0,
        "emp": "salaried",
        "question": "My annual salary is ₹55 Lakhs. What is my tax and is there any scrutiny risk?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Personal CA Tax Analysis for Gross Annual Income ₹5,500,000 (Salaried): Under the New Tax Regime, your total tax including 10% surcharge is ₹1,465,100.00. Under the Old Tax Regime, total tax is ₹1,617,000.00. Recommendation: Choose New Tax Regime to save ₹151,900.00.",
        "analyst_answer": [
            {"text": "Under the New Tax Regime, your total tax liability is ₹1,465,100.00 including the applicable 10% surcharge.", "cites": ["T1"]},
            {"text": "The New Regime still delivers a substantial benefit, saving you ₹151,900.00 over the Old Regime liability of ₹1,617,000.00.", "cites": ["T1"]},
            {"text": "Because income exceeds ₹50 Lakh, remember that surcharge applies and comprehensive asset-liability disclosure is mandated.", "cites": ["T1"]},
            {"text": "Please prepare your Schedule AL disclosures diligently before filing your ITR-2.", "cites": ["T1"]},
        ],
        "risk_level": "medium",
        "risk_factors": [{"factor": "Income exceeding ₹50 Lakh triggers mandatory Schedule AL (Assets and Liabilities) reporting and enhanced scrutiny review under Section 143(2).", "cite": "R1"}],
        "what_to_do": ["Prepare cost basis records for immovable property and jewelry for Schedule AL.", "Reconcile all bank balances reported in AIS as of March 31."],
    },
    {
        "income": 650000.0,
        "emp": "salaried",
        "question": "Can I open Sukanya Samriddhi Yojana for my daughter and what are the benefits?",
        "tool_result": "TOOL_RESULT [scheme_lookup]: Sukanya Samriddhi Account (SSY) is an Indian government-backed savings scheme for girl children before age 10. Offers 8.2% sovereign interest, minimum deposit ₹250, maximum ₹1.5 Lakh per FY, and complete EEE tax exemption under Section 80C.",
        "analyst_answer": [
            {"text": "Yes, you can open a Sukanya Samriddhi Yojana account for your daughter any time before she turns 10 years old.", "cites": ["T1"]},
            {"text": "This is one of the safest and highest-yielding government schemes available, offering an 8.2% sovereign interest rate with full EEE tax exemption.", "cites": ["T1"]},
            {"text": "You can start with a minimum annual deposit of ₹250 up to a maximum limit of ₹1.5 Lakh per financial year.", "cites": ["T1"]},
            {"text": "To get started, visit any authorized post office or bank branch with your daughter's birth certificate and your KYC documents.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Statutory government savings scheme with sovereign guarantee carries zero tax scrutiny risk.", "cite": "R1"}],
        "what_to_do": ["Retain annual deposit receipts and interest passbook for Section 80C proof if choosing Old Tax Regime."],
    },
    {
        "income": 800000.0,
        "emp": "salaried",
        "question": "What is PM-KISAN scheme and who is eligible?",
        "tool_result": "TOOL_RESULT [scheme_lookup]: Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a central sector scheme providing income support of ₹6,000 per year in three equal installments of ₹2,000 each to eligible farmer families across India directly via DBT.",
        "analyst_answer": [
            {"text": "PM-KISAN is an Indian government income support initiative that provides eligible farming families with ₹6,000 each year.", "cites": ["T1"]},
            {"text": "The amount is deposited directly into the beneficiary's bank account in three equal installments of ₹2,000 every four months.", "cites": ["T1"]},
            {"text": "All landholding farmer families are eligible, subject to statutory exclusions such as institutional landholders or income-tax payers.", "cites": ["T1"]},
            {"text": "You can register directly on the official pmkisan.gov.in portal or via your nearest Common Service Centre (CSC) with your Aadhaar and land record.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Direct benefit transfer under Central Sector Scheme verified via Aadhaar DBT.", "cite": "R1"}],
        "what_to_do": ["Complete e-KYC on pmkisan.gov.in using Aadhaar OTP to prevent installment pauses."],
    },
]


def build_datasets():
    qwen_samples = []
    deepseek_samples = []

    for sc in SCENARIOS:
        client_msg = sc["question"]
        facts = {
            "gross_income_rupees": sc["income"],
            "employment_type": sc["emp"],
            "regime_preference": "new",
            "financial_year": "2024-25",
            "deductions_rupees": {},
            "loan": {},
            "ticker": None,
        }
        facts_memo = {
            "from": "intake",
            "to": "analyst",
            "type": "FACTS",
            "content": {
                "intent": "tax_calc",
                "facts": facts,
                "open_questions": [],
                "needs_web": False,
                "ack": "Understood. The firm is gathering official sources and calculation sheets for you now.",
            },
            "cites": [],
            "flags": [],
        }
        case = {
            "client_message": client_msg,
            "client_facts": facts,
            "open_questions": [],
            "tool_results": [sc["tool_result"]],
            "sources": [],
            "rules": STANDARD_RULES_KB,
            "memos": [facts_memo],
            "draft": None,
            "client_history": "none",
        }
        case_text = render_case_file(case)

        # 1. Qwen Junior Analyst Training Sample
        qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
        qwen_user = "Prepare your memo now."
        qwen_draft_memo = {
            "from": "analyst",
            "to": "senior",
            "type": "DRAFT",
            "content": {
                "answer": sc["analyst_answer"],
                "assumptions": ["Standard deduction of ₹75,000 applied for salaried under New Tax Regime."],
                "uncertain": [],
            },
            "cites": ["T1"],
            "flags": [],
        }
        qwen_samples.append({
            "system": qwen_system,
            "instruction": qwen_user,
            "context": case_text,
            "output": json.dumps(qwen_draft_memo),
        })

        # 2. DeepSeek Senior Partner Training Sample
        case["draft"] = qwen_draft_memo
        case["memos"].append(qwen_draft_memo)
        deepseek_case_text = render_case_file(case)

        deepseek_user = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + deepseek_case_text + "\nPost your memo now."
        think_trace = (
            f"<think>\n"
            f"Senior Partner Critical Audit:\n"
            f"- Client Facts: Annual Income {sc['income']} ({sc['emp']}).\n"
            f"- Arithmetic Grounding: Verified exact figures against [T1] Tool Result.\n"
            f"- Scrutiny Risk Analysis: Section 143(1) automated processing risk evaluated as {sc['risk_level']}.\n"
            f"- Statutory Compliance: Official portal incometax.gov.in verified.\n"
            f"- Verdict: Approve without challenges.\n"
            f"</think>"
        )
        deepseek_final_memo = {
            "from": "senior",
            "to": "verifier",
            "type": "FINAL",
            "content": {
                "final_answer": sc["analyst_answer"],
                "notice_risk": {
                    "level": sc["risk_level"],
                    "factors": sc["risk_factors"],
                    "what_to_do": sc["what_to_do"],
                },
                "needs_user_input": [],
                "official_proof": [{"claim": "Official Income Tax e-Filing & Rules Portal", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                "closing_note": "This assessment is educational guidance based on official rates and does not constitute a statutory audit certificate.",
                "think": think_trace,
            },
            "cites": ["T1"],
            "flags": [],
        }
        deepseek_output = f"{think_trace}\n{json.dumps(deepseek_final_memo)}"

        deepseek_samples.append({
            "instruction": deepseek_user,
            "context": deepseek_case_text,
            "output": deepseek_output,
        })

    # Save datasets
    qwen_path = DATA_DIR / "qwen_junior_analyst_train.jsonl"
    with open(qwen_path, "w", encoding="utf-8") as f:
        for s in qwen_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Generated {len(qwen_samples)} samples in {qwen_path}")

    deepseek_path = DATA_DIR / "deepseek_senior_partner_train.jsonl"
    with open(deepseek_path, "w", encoding="utf-8") as f:
        for s in deepseek_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Generated {len(deepseek_samples)} samples in {deepseek_path}")


if __name__ == "__main__":
    build_datasets()

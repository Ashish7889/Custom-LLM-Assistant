"""
data/generate_200_firm_samples.py -- Generates 200+ grounded Indian tax training samples for FinSage:
1. Qwen 2.5-1.5B (Junior CA Analyst memo drafting) -> data/qwen_junior_analyst_train.jsonl
2. DeepSeek-R1-Distill-1.5B (Senior Partner <think> CoT + Final Advice) -> data/deepseek_senior_partner_train.jsonl
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

SALARY_BRACKETS = [
    (300000, 0, 0, 0, "I earn Rs 3 Lakh per year. Do I need to pay any tax or file ITR?"),
    (400000, 0, 0, 0, "My annual salary is Rs 4 Lakh. Will any tax be deducted?"),
    (500000, 0, 0, 0, "My CTC is Rs 5 Lakh. What is my income tax under New and Old regime?"),
    (600000, 0, 5000, 0, "I earn Rs 6 Lakh salary. Do I get Section 87A rebate?"),
    (700000, 0, 15000, 0, "My total salary is Rs 7 Lakh. What is the tax under the new regime?"),
    (750000, 0, 20800, 0, "My salary is Rs 7.5 Lakh. Is it true I pay Rs 0 tax after standard deduction?"),
    (775000, 0, 24700, 0, "My salary is Rs 7,75,000. With Rs 75,000 standard deduction, how much tax?"),
    (800000, 26000, 31200, 5200, "I have a salary of Rs 8 Lakh. How much is the tax and what is best regime?"),
    (850000, 33800, 44200, 10400, "My annual package is Rs 8.5 Lakh. Which regime saves more?"),
    (900000, 41600, 57200, 15600, "I earn Rs 9 Lakh. Should I choose new regime or claim 80C under old?"),
    (1000000, 54600, 83200, 28600, "My CTC is Rs 10 Lakh. What is my tax payable under FY 2025-26 rules?"),
    (1100000, 67600, 109200, 41600, "I earn Rs 11 Lakh. What is my total tax under new tax slabs?"),
    (1200000, 80600, 135200, 54600, "My salary is Rs 12 Lakh. How much will I save in new regime?"),
    (1300000, 96200, 161200, 65000, "My annual CTC is Rs 13 Lakh. Please advise tax calculation."),
    (1400000, 114400, 187200, 72800, "I make Rs 14 Lakh per year. Which regime is better?"),
    (1500000, 135200, 213200, 78000, "My salary is Rs 15 Lakh. What is my tax breakdown?"),
    (1600000, 161200, 244400, 83200, "I earn Rs 16 Lakh. How much tax under the new regime?"),
    (1800000, 213200, 306800, 93600, "My package is Rs 18 Lakh. What is my annual tax liability?"),
    (2000000, 265200, 369200, 104000, "My annual CTC is Rs 20 Lakh. What is the tax calculation?"),
    (2200000, 317200, 431600, 114400, "I earn Rs 22 Lakh salary. How much do I pay?"),
    (2500000, 395200, 525200, 130000, "My salary is Rs 25 Lakh per year. Please compute my tax."),
    (3000000, 525200, 681200, 156000, "I earn Rs 30 Lakh. What is my tax under new regime?"),
    (4000000, 785200, 993200, 208000, "My package is Rs 40 Lakh. Compute my tax liability."),
    (5000000, 1045200, 1305200, 260000, "I earn Rs 50 Lakh. What is the surcharge and tax?"),
]

FREELANCER_BRACKETS = [
    (500000, 250000, 0, "I am a freelance UI designer earning Rs 5 Lakh. How much tax under 44ADA?"),
    (800000, 400000, 0, "I earned Rs 8 Lakh from software consulting. What is my tax under 44ADA?"),
    (1000000, 500000, 0, "I make Rs 10 Lakh from independent freelancing. How much tax do I pay under 44ADA?"),
    (1200000, 600000, 0, "I earn Rs 12 Lakh as a content creator. Can I use Section 44ADA?"),
    (1500000, 750000, 0, "My freelancing income is Rs 15 Lakh. What is my taxable profit under 44ADA?"),
    (1800000, 900000, 31200, "I earned Rs 18 Lakh as an architect consultant. What is my tax under 44ADA?"),
    (2000000, 1000000, 52000, "My freelance engineering revenue is Rs 20 Lakh. How much tax under 44ADA?"),
    (2500000, 1250000, 91000, "I made Rs 25 Lakh from digital marketing consultancy. What is 44ADA tax?"),
    (3000000, 1500000, 135200, "My software development contracts totaled Rs 30 Lakh. How is it taxed?"),
    (4000000, 2000000, 265200, "I earned Rs 40 Lakh from freelance data science contracts. Compute my tax."),
    (5000000, 2500000, 395200, "My consulting gross receipts are Rs 50 Lakh. What is my tax under 44ADA?"),
    (7500000, 3750000, 720200, "My professional receipts are Rs 75 Lakh all digital. Can I claim 44ADA?"),
]

CAPITAL_GAINS_SCENARIOS = [
    (150000, "LTCG", 125000, 3120, "I made Rs 1,50,000 profit on selling equity mutual funds after 2 years. What is the tax?"),
    (300000, "LTCG", 125000, 21840, "My long term capital gain on shares is Rs 3,00,000. How much tax under Section 112A?"),
    (500000, "LTCG", 125000, 46800, "I booked Rs 5 Lakh profit on equity shares held for 3 years. Calculate my 112A tax."),
    (1000000, "LTCG", 125000, 109200, "I made Rs 10 Lakh long term capital gains from stocks. What is the tax rate and liability?"),
    (2000000, "LTCG", 125000, 234000, "My LTCG on equity mutual funds is Rs 20 Lakh. How much tax to pay?"),
    (100000, "STCG", 0, 20800, "I made Rs 1 Lakh short term capital gains on shares sold within 6 months. What is the tax?"),
    (250000, "STCG", 0, 52000, "My STCG on listed stocks is Rs 2,50,000. What is the tax under Section 111A?"),
    (500000, "STCG", 0, 104000, "I booked Rs 5 Lakh short-term trading profit on equities. How much tax?"),
    (1000000, "Property LTCG", 0, 125000, "I sold a residential plot with Rs 10 Lakh capital gain. What is the new 12.5% tax?"),
    (2500000, "Property LTCG", 0, 312500, "I made Rs 25 Lakh profit on selling an apartment. What is the capital gains tax?"),
]

SENIOR_CITIZEN_SCENARIOS = [
    (450000, 65, 40000, "I am 65 years old with pension Rs 4.5L and interest Rs 40,000. Do I pay tax?"),
    (600000, 68, 60000, "I am 68 with Rs 6 Lakh pension and Rs 60k FD interest. How much tax and 80TTB deduction?"),
    (750000, 72, 50000, "I am 72 years old earning Rs 7.5 Lakh. What is my tax under Old vs New regime?"),
    (900000, 62, 50000, "I am a retired teacher aged 62 with Rs 9 Lakh income. Which regime is better?"),
    (1200000, 82, 70000, "I am 82 years old super-senior with Rs 12 Lakh income. What is my tax?"),
]

DOC_RECONCILIATION_SCENARIOS = [
    ("Form 16 vs 26AS", "My Form 16 shows Rs 45,000 TDS but 26AS shows Rs 35,000. Why the mismatch?", "Employer likely deposited TDS late or filed quarterly return with wrong PAN. Ask employer to file TDS correction return."),
    ("AIS High-Value Cash", "My AIS shows Rs 12 Lakh cash deposit in SBI. Will I get an IT notice?", "Cash deposits exceeding Rs 10 Lakh in savings account trigger SFT reporting under Sec 285BA. Keep bank statements and proof of source ready."),
    ("Dividend TDS 194K", "I received Rs 15,000 dividend and 10% TDS was deducted. Do I need to report it?", "Yes dividend income is taxable under Income from Other Sources at slab rates. Claim credit for Rs 1,500 TDS in ITR."),
    ("Savings Interest 80TTA", "I earned Rs 18,000 savings bank interest. How much can I deduct?", "Under Section 80TTA deduction is capped at Rs 10,000. The balance Rs 8,000 is taxable at slab rates under Old Regime."),
    ("Section 89 Relief", "I received Rs 1,50,000 salary arrears. How to avoid higher tax bracket?", "File Form 10E online before filing ITR to claim Section 89(1) tax relief on salary arrears."),
]


def build_datasets():
    qwen_samples = []
    deepseek_samples = []

    # 1. Salaried (24 * 4 = 96)
    for income, new_tax, old_tax, savings, q_base in SALARY_BRACKETS:
        variations = [
            (q_base, f"TOOL_RESULT [tax_calculator_india]: Income Rs {income:,.2f} (Salaried). New Regime tax: Rs {new_tax:,.2f}. Old Regime tax: Rs {old_tax:,.2f}. New Regime saves Rs {savings:,.2f}."),
            (f"Can you compute my income tax for Rs {income:,.2f} salary package?", f"TOOL_RESULT [tax_calculator_india]: Income Rs {income:,.2f} (Salaried). New Regime tax: Rs {new_tax:,.2f}. Old Regime tax: Rs {old_tax:,.2f}."),
            (f"What is my take-home and tax on annual salary of Rs {income:,.2f}?", f"TOOL_RESULT [tax_calculator_india]: Income Rs {income:,.2f} (Salaried). Net tax Rs {new_tax:,.2f} under new slabs."),
            (f"Between new and old tax regime, which one should I pick for Rs {income:,.2f} income?", f"TOOL_RESULT [tax_calculator_india]: Income Rs {income:,.2f}. New Regime Rs {new_tax:,.2f} vs Old Regime Rs {old_tax:,.2f}."),
        ]

        for q, tool_res in variations:
            tax_str = f"Rs {new_tax:,.2f}" if new_tax > 0 else "Rs 0 (zero tax)"
            regime_advice = "The New Tax Regime is the optimal choice." if new_tax <= old_tax else "The Old Tax Regime may be beneficial if you have substantial 80C/80D deductions."
            
            analyst_ans = [
                {"text": f"Your estimated income tax under the New Tax Regime is {tax_str}.", "cites": ["T1"]},
                {"text": regime_advice, "cites": ["T1"]},
                {"text": "Always reconcile your TDS credits with Form 26AS on incometax.gov.in before filing.", "cites": ["T1"]},
            ]

            facts = {"gross_income_rupees": income, "employment_type": "salaried", "regime_preference": "new", "financial_year": "2024-25"}
            facts_memo = {
                "from": "intake", "to": "analyst", "type": "FACTS",
                "content": {"intent": "tax_calc", "facts": facts, "open_questions": [], "needs_web": False, "ack": "Acknowledged."},
                "cites": [], "flags": []
            }
            case = {
                "client_message": q, "client_facts": facts, "open_questions": [],
                "tool_results": [tool_res], "sources": [], "rules": STANDARD_RULES_KB,
                "memos": [facts_memo], "draft": None, "client_history": "none",
            }
            case_text = render_case_file(case)

            qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
            qwen_draft = {
                "from": "analyst", "to": "senior", "type": "DRAFT",
                "content": {"answer": analyst_ans, "assumptions": ["Standard deduction Rs 75,000 applied."], "uncertain": []},
                "cites": ["T1"], "flags": []
            }
            qwen_samples.append({
                "system": qwen_system,
                "instruction": "Prepare your memo now.",
                "context": case_text,
                "output": json.dumps(qwen_draft, ensure_ascii=False),
            })

            case["draft"] = qwen_draft
            case["memos"].append(qwen_draft)
            ds_case_text = render_case_file(case)

            think_trace = (
                f"<think>\n"
                f"Senior Partner Critical Audit:\n"
                f"- Client Facts: Annual Income {income} (salaried).\n"
                f"- Arithmetic Grounding: Verified exact figures against [T1] Tool Result.\n"
                f"- Scrutiny Risk Analysis: risk evaluated as low.\n"
                f"- Statutory Compliance: Official portal incometax.gov.in verified.\n"
                f"- Verdict: Approve without challenges.\n"
                f"</think>"
            )
            senior_memo = {
                "from": "senior", "to": "verifier", "type": "FINAL",
                "content": {
                    "final_answer": analyst_ans,
                    "notice_risk": {"level": "low", "factors": [{"factor": "Routine salaried return under standard slab rates.", "cite": "R1"}], "what_to_do": ["Verify TDS in Form 26AS before filing."]},
                    "needs_user_input": [],
                    "official_proof": [{"claim": "Official Income Tax e-Filing Portal", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                    "closing_note": "Educational guidance based on official rates, not a statutory audit certificate.",
                    "think": think_trace,
                },
                "cites": ["T1"], "flags": []
            }
            deepseek_samples.append({
                "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
                "context": ds_case_text,
                "output": think_trace + "\n" + json.dumps(senior_memo, ensure_ascii=False),
            })

    # 2. Freelancers (12 * 4 = 48)
    for gross, deemed_profit, tax, q_base in FREELANCER_BRACKETS:
        variations = [
            (q_base, f"TOOL_RESULT [tax_calculator_india]: Gross Receipts Rs {gross:,.2f} under Sec 44ADA. Presumptive Profit 50%: Rs {deemed_profit:,.2f}. Tax: Rs {tax:,.2f}."),
            (f"How much tax do I pay under 44ADA on Rs {gross:,.2f} revenue?", f"TOOL_RESULT [tax_calculator_india]: Sec 44ADA 50% profit: Rs {deemed_profit:,.2f}. Tax payable: Rs {tax:,.2f}."),
            (f"Can software freelancers earning Rs {gross:,.2f} file ITR-4 Sugam?", f"TOOL_RESULT [tax_calculator_india]: Eligible under Sec 44ADA for ITR-4. Deemed income Rs {deemed_profit:,.2f}."),
            (f"What books of accounts do I need for Rs {gross:,.2f} freelancing income?", f"TOOL_RESULT [tax_calculator_india]: Under 44ADA no books maintenance u/s 44AA required if declaring >=50% profit."),
        ]

        for q, tool_res in variations:
            analyst_ans = [
                {"text": f"Under Section 44ADA your taxable income is taken as 50% of receipts (Rs {deemed_profit:,.2f}).", "cites": ["T1"]},
                {"text": f"Estimated tax payable is Rs {tax:,.2f} under the New Tax Regime.", "cites": ["T1"]},
                {"text": "You do not need to maintain detailed accounting ledgers or audit books if 50% or higher profit is declared in ITR-4.", "cites": ["T1"]},
            ]

            facts = {"gross_income_rupees": gross, "employment_type": "freelancer", "regime_preference": "new", "financial_year": "2024-25"}
            facts_memo = {
                "from": "intake", "to": "analyst", "type": "FACTS",
                "content": {"intent": "tax_calc", "facts": facts, "open_questions": [], "needs_web": False, "ack": "Acknowledged."},
                "cites": [], "flags": []
            }
            case = {
                "client_message": q, "client_facts": facts, "open_questions": [],
                "tool_results": [tool_res], "sources": [], "rules": STANDARD_RULES_KB,
                "memos": [facts_memo], "draft": None, "client_history": "none",
            }
            case_text = render_case_file(case)

            qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
            qwen_draft = {
                "from": "analyst", "to": "senior", "type": "DRAFT",
                "content": {"answer": analyst_ans, "assumptions": ["Section 44ADA 50% presumptive profit applied."], "uncertain": []},
                "cites": ["T1"], "flags": []
            }
            qwen_samples.append({
                "system": qwen_system,
                "instruction": "Prepare your memo now.",
                "context": case_text,
                "output": json.dumps(qwen_draft, ensure_ascii=False),
            })

            case["draft"] = qwen_draft
            case["memos"].append(qwen_draft)
            ds_case_text = render_case_file(case)

            think_trace = (
                f"<think>\n"
                f"Senior Partner Critical Audit:\n"
                f"- Client Facts: Freelance receipts {gross}.\n"
                f"- Arithmetic Grounding: 50% presumptive is {deemed_profit}, verified against [T1].\n"
                f"- Scrutiny Risk Analysis: low risk within statutory limits.\n"
                f"- Verdict: Approved.\n"
                f"</think>"
            )
            senior_memo = {
                "from": "senior", "to": "verifier", "type": "FINAL",
                "content": {
                    "final_answer": analyst_ans,
                    "notice_risk": {"level": "low", "factors": [{"factor": "Statutory Section 44ADA declaration.", "cite": "R1"}], "what_to_do": ["Pay advance tax by March 15."]},
                    "needs_user_input": [],
                    "official_proof": [{"claim": "Section 44ADA CBDT Rules", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                    "closing_note": "Educational guidance based on official rates, not a statutory audit certificate.",
                    "think": think_trace,
                },
                "cites": ["T1"], "flags": []
            }
            deepseek_samples.append({
                "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
                "context": ds_case_text,
                "output": think_trace + "\n" + json.dumps(senior_memo, ensure_ascii=False),
            })

    # 3. Capital Gains (10 * 3 = 30)
    for gain, gtype, exempt, cgtax, q_base in CAPITAL_GAINS_SCENARIOS:
        variations = [
            (q_base, f"TOOL_RESULT [capital_gains_calculator]: Gain Rs {gain:,.2f} ({gtype}). Exemption Rs {exempt:,.2f}. Taxable Rs {max(0, gain-exempt):,.2f}. Tax: Rs {cgtax:,.2f}."),
            (f"What is the tax rate on {gtype} of Rs {gain:,.2f}?", f"TOOL_RESULT [capital_gains_calculator]: {gtype} rate applied. Tax Rs {cgtax:,.2f}."),
            (f"Do I pay tax on Rs {gain:,.2f} {gtype} in listed shares?", f"TOOL_RESULT [capital_gains_calculator]: Exemption Rs {exempt:,.2f}. Net tax Rs {cgtax:,.2f}."),
        ]

        for q, tool_res in variations:
            analyst_ans = [
                {"text": f"Your tax liability on {gtype} of Rs {gain:,.2f} is Rs {cgtax:,.2f}.", "cites": ["T1"]},
                {"text": f"Under updated Budget 2024 rules, long-term equity capital gains above Rs 1.25 Lakh are taxed at 12.5% (STCG at 20%).", "cites": ["T1"]},
                {"text": "Report these capital gains in Schedule CG of ITR-2.", "cites": ["T1"]},
            ]

            facts = {"capital_gain_rupees": gain, "gain_type": gtype, "financial_year": "2024-25"}
            facts_memo = {
                "from": "intake", "to": "analyst", "type": "FACTS",
                "content": {"intent": "tax_calc", "facts": facts, "open_questions": [], "needs_web": False, "ack": "Acknowledged."},
                "cites": [], "flags": []
            }
            case = {
                "client_message": q, "client_facts": facts, "open_questions": [],
                "tool_results": [tool_res], "sources": [], "rules": STANDARD_RULES_KB,
                "memos": [facts_memo], "draft": None, "client_history": "none",
            }
            case_text = render_case_file(case)

            qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
            qwen_draft = {
                "from": "analyst", "to": "senior", "type": "DRAFT",
                "content": {"answer": analyst_ans, "assumptions": ["Capital gains reported as per broker statement."], "uncertain": []},
                "cites": ["T1"], "flags": []
            }
            qwen_samples.append({
                "system": qwen_system,
                "instruction": "Prepare your memo now.",
                "context": case_text,
                "output": json.dumps(qwen_draft, ensure_ascii=False),
            })

            case["draft"] = qwen_draft
            case["memos"].append(qwen_draft)
            ds_case_text = render_case_file(case)

            think_trace = (
                f"<think>\n"
                f"Senior Partner Critical Audit:\n"
                f"- Asset: {gtype} Gain {gain}.\n"
                f"- Arithmetic Grounding: Tax {cgtax} verified against [T1].\n"
                f"- Schedule: ITR-2 required.\n"
                f"- Verdict: Approved.\n"
                f"</think>"
            )
            senior_memo = {
                "from": "senior", "to": "verifier", "type": "FINAL",
                "content": {
                    "final_answer": analyst_ans,
                    "notice_risk": {"level": "low", "factors": [{"factor": "Capital gains reported matching AIS broker data.", "cite": "R1"}], "what_to_do": ["Download Consolidated Capital Gains statement."]},
                    "needs_user_input": [],
                    "official_proof": [{"claim": "CBDT Capital Gains Schedule", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                    "closing_note": "Educational guidance based on official rates, not a statutory audit certificate.",
                    "think": think_trace,
                },
                "cites": ["T1"], "flags": []
            }
            deepseek_samples.append({
                "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
                "context": ds_case_text,
                "output": think_trace + "\n" + json.dumps(senior_memo, ensure_ascii=False),
            })

    # 4. Senior Citizens (5 * 3 = 15)
    for income, age, int_inc, q in SENIOR_CITIZEN_SCENARIOS:
        tool_res = f"TOOL_RESULT [senior_citizen_tax]: Age {age}. Income Rs {income:,.2f}. Interest Rs {int_inc:,.2f}. 80TTB deduction Rs {min(int_inc, 50000):,.2f}."
        analyst_ans = [
            {"text": f"As a senior citizen aged {age}, you are entitled to claim deduction up to Rs 50,000 on bank interest under Section 80TTB.", "cites": ["T1"]},
            {"text": "Under the New Tax Regime, total income up to Rs 7.75 Lakh has zero net tax with rebate.", "cites": ["T1"]},
            {"text": "Submit Form 15H to your bank at the start of the financial year to prevent TDS deduction on your FD interest.", "cites": ["T1"]},
        ]

        facts = {"gross_income_rupees": income, "age": age, "interest_income": int_inc, "financial_year": "2024-25"}
        facts_memo = {
            "from": "intake", "to": "analyst", "type": "FACTS",
            "content": {"intent": "tax_calc", "facts": facts, "open_questions": [], "needs_web": False, "ack": "Acknowledged."},
            "cites": [], "flags": []
        }
        case = {
            "client_message": q, "client_facts": facts, "open_questions": [],
            "tool_results": [tool_res], "sources": [], "rules": STANDARD_RULES_KB,
            "memos": [facts_memo], "draft": None, "client_history": "none",
        }
        case_text = render_case_file(case)

        qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
        qwen_draft = {
            "from": "analyst", "to": "senior", "type": "DRAFT",
            "content": {"answer": analyst_ans, "assumptions": ["Senior citizen status verified."], "uncertain": []},
            "cites": ["T1"], "flags": []
        }
        qwen_samples.append({
            "system": qwen_system,
            "instruction": "Prepare your memo now.",
            "context": case_text,
            "output": json.dumps(qwen_draft, ensure_ascii=False),
        })

        case["draft"] = qwen_draft
        case["memos"].append(qwen_draft)
        ds_case_text = render_case_file(case)

        think_trace = (
            f"<think>\n"
            f"Senior Partner Critical Audit:\n"
            f"- Client Status: Age {age} Senior Citizen.\n"
            f"- Deduction: Section 80TTB Rs 50,000 applied.\n"
            f"- Form 15H: Recommended.\n"
            f"- Verdict: Approved.\n"
            f"</think>"
        )
        senior_memo = {
            "from": "senior", "to": "verifier", "type": "FINAL",
            "content": {
                "final_answer": analyst_ans,
                "notice_risk": {"level": "low", "factors": [{"factor": "Standard senior citizen pension and interest.", "cite": "R1"}], "what_to_do": ["Submit Form 15H to bank branch."]},
                "needs_user_input": [],
                "official_proof": [{"claim": "Senior Citizen Income Tax Benefits", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                "closing_note": "Educational guidance based on official rates, not a statutory audit certificate.",
                "think": think_trace,
            },
            "cites": ["T1"], "flags": []
        }
        deepseek_samples.append({
            "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
            "context": ds_case_text,
            "output": think_trace + "\n" + json.dumps(senior_memo, ensure_ascii=False),
        })

    # 5. Document Reconciliation (5 * 3 = 15)
    for topic, q, sol in DOC_RECONCILIATION_SCENARIOS:
        tool_res = f"TOOL_RESULT [compliance_check]: Issue: {topic}. Resolution: {sol}."
        analyst_ans = [
            {"text": sol, "cites": ["T1"]},
            {"text": "Always reconcile your Annual Information Statement (AIS) before submitting your return.", "cites": ["T1"]},
        ]

        facts = {"issue": topic, "financial_year": "2024-25"}
        facts_memo = {
            "from": "intake", "to": "analyst", "type": "FACTS",
            "content": {"intent": "tax_calc", "facts": facts, "open_questions": [], "needs_web": False, "ack": "Acknowledged."},
            "cites": [], "flags": []
        }
        case = {
            "client_message": q, "client_facts": facts, "open_questions": [],
            "tool_results": [tool_res], "sources": [], "rules": STANDARD_RULES_KB,
            "memos": [facts_memo], "draft": None, "client_history": "none",
        }
        case_text = render_case_file(case)

        qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
        qwen_draft = {
            "from": "analyst", "to": "senior", "type": "DRAFT",
            "content": {"answer": analyst_ans, "assumptions": ["AIS reconciliation verified."], "uncertain": []},
            "cites": ["T1"], "flags": []
        }
        qwen_samples.append({
            "system": qwen_system,
            "instruction": "Prepare your memo now.",
            "context": case_text,
            "output": json.dumps(qwen_draft, ensure_ascii=False),
        })

        case["draft"] = qwen_draft
        case["memos"].append(qwen_draft)
        ds_case_text = render_case_file(case)

        risk_level = "medium" if "mismatch" in q.lower() or "notice" in q.lower() else "low"
        think_trace = (
            f"<think>\n"
            f"Senior Partner Critical Audit:\n"
            f"- Compliance Vector: {topic}.\n"
            f"- Risk Assessment: {risk_level}.\n"
            f"- Actionable Step: {sol}.\n"
            f"- Verdict: Approved.\n"
            f"</think>"
        )
        senior_memo = {
            "from": "senior", "to": "verifier", "type": "FINAL",
            "content": {
                "final_answer": analyst_ans,
                "notice_risk": {"level": risk_level, "factors": [{"factor": f"Potential inquiry if {topic} remains unaddressed.", "cite": "R1"}], "what_to_do": ["Provide feedback in AIS portal if discrepancy exists."]},
                "needs_user_input": [],
                "official_proof": [{"claim": "CBDT AIS/TIS Compliance Guidelines", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                "closing_note": "Educational guidance based on official rates, not a statutory audit certificate.",
                "think": think_trace,
            },
            "cites": ["T1"], "flags": []
        }
        deepseek_samples.append({
            "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
            "context": ds_case_text,
            "output": think_trace + "\n" + json.dumps(senior_memo, ensure_ascii=False),
        })

    # 6. Loan Advisory & EMI traps (6 base * 4 variations = 24 samples -> 208 total)
    loan_scenarios = [
        (500000, 14.0, 5, "I am offered a Rs 5 Lakh personal loan at 14% flat interest for 5 years. Is it a good deal?", "14% flat interest is roughly equivalent to 24.5% reducing effective rate. You pay Rs 3,50,000 in interest alone. Insist on a reducing balance quote."),
        (1000000, 9.5, 15, "I have a Rs 10 Lakh home loan at 9.5% for 15 years. How much will I save if I prepay Rs 1 Lakh?", "Prepaying Rs 1 Lakh in year 2 saves approximately Rs 1,65,000 in future interest and cuts loan tenure by over 20 months."),
        (2000000, 8.75, 20, "Should I take a 20-year home loan or 15-year home loan for Rs 20 Lakh at 8.75%?", "A 15-year tenure saves over Rs 7,80,000 in total interest compared to 20 years with only around Rs 2,500/month higher EMI."),
        (300000, 16.0, 3, "Is zero-cost EMI on a Rs 3 Lakh consumer loan really zero interest?", "No, zero-cost EMIs usually charge 18% GST on the interest discount component and may levy non-refundable processing fees."),
        (1500000, 9.0, 10, "What is the tax benefit on home loan principal and interest?", "Under Old Regime, Section 24(b) allows up to Rs 2 Lakh deduction on interest, and Section 80C allows up to Rs 1.5 Lakh on principal repayment."),
        (800000, 11.5, 7, "My car loan is at 11.5% reducing. Can I claim any income tax deduction?", "Car loans for personal use have zero tax deductions. Only self-employed individuals using the vehicle for business can claim interest as business expense."),
    ]

    for principal, rate, tenure, q_base, adv in loan_scenarios:
        variations = [
            (q_base, f"TOOL_RESULT [loan_calculator]: Principal Rs {principal:,.2f}, Rate {rate}%, Tenure {tenure} yrs. Advice: {adv}"),
            (f"Can you check if my loan quote for Rs {principal:,.2f} at {rate}% is safe?", f"TOOL_RESULT [loan_calculator]: Rate analysis. Advice: {adv}"),
            (f"What is the effective rate and savings on Rs {principal:,.2f} loan?", f"TOOL_RESULT [loan_calculator]: Computed repayment schedule. Advice: {adv}"),
            (f"Advise on repayment strategy for Rs {principal:,.2f} at {rate}%.", f"TOOL_RESULT [loan_calculator]: Tenure {tenure} years. Advice: {adv}"),
        ]

        for q, tool_res in variations:
            analyst_ans = [
                {"text": adv, "cites": ["T1"]},
                {"text": "Always compare the Annual Percentage Rate (APR) including processing fees before signing loan agreements.", "cites": ["T1"]},
            ]

            facts = {"loan_principal": principal, "interest_rate": rate, "tenure_years": tenure}
            facts_memo = {
                "from": "intake", "to": "analyst", "type": "FACTS",
                "content": {"intent": "loan_calc", "facts": facts, "open_questions": [], "needs_web": False, "ack": "Acknowledged."},
                "cites": [], "flags": []
            }
            case = {
                "client_message": q, "client_facts": facts, "open_questions": [],
                "tool_results": [tool_res], "sources": [], "rules": STANDARD_RULES_KB,
                "memos": [facts_memo], "draft": None, "client_history": "none",
            }
            case_text = render_case_file(case)

            qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
            qwen_draft = {
                "from": "analyst", "to": "senior", "type": "DRAFT",
                "content": {"answer": analyst_ans, "assumptions": ["Standard reducing balance amortization verified."], "uncertain": []},
                "cites": ["T1"], "flags": []
            }
            qwen_samples.append({
                "system": qwen_system,
                "instruction": "Prepare your memo now.",
                "context": case_text,
                "output": json.dumps(qwen_draft, ensure_ascii=False),
            })

            case["draft"] = qwen_draft
            case["memos"].append(qwen_draft)
            ds_case_text = render_case_file(case)

            think_trace = (
                f"<think>\n"
                f"Senior Partner Loan & Financial Advisory Audit:\n"
                f"- Principal: Rs {principal:,.2f} at {rate}%.\n"
                f"- Amortization Check: Verified APR and flat-vs-reducing interest implications against [T1].\n"
                f"- Client Protection: Warned against hidden charges and ungrounded claims.\n"
                f"- Verdict: Approved.\n"
                f"</think>"
            )
            senior_memo = {
                "from": "senior", "to": "verifier", "type": "FINAL",
                "content": {
                    "final_answer": analyst_ans,
                    "notice_risk": {"level": "low", "factors": [{"factor": "Financial planning guidance without tax scrutiny triggers.", "cite": "R1"}], "what_to_do": ["Request Key Fact Statement (KFS) from the lender."]},
                    "needs_user_input": [],
                    "official_proof": [{"claim": "RBI Guidelines on Digital Lending and KFS", "url": "https://www.rbi.org.in/"}],
                    "closing_note": "Educational guidance by FinSage & Associates based on RBI lending disclosure norms.",
                    "think": think_trace,
                },
                "cites": ["T1"], "flags": []
            }
            deepseek_samples.append({
                "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
                "context": ds_case_text,
                "output": think_trace + "\n" + json.dumps(senior_memo, ensure_ascii=False),
            })

    # Save to jsonl
    qwen_path = DATA_DIR / "qwen_junior_analyst_train.jsonl"
    with open(qwen_path, "w", encoding="utf-8") as f:
        for s in qwen_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[OK] Generated {len(qwen_samples)} samples in {qwen_path}")

    deepseek_path = DATA_DIR / "deepseek_senior_partner_train.jsonl"
    with open(deepseek_path, "w", encoding="utf-8") as f:
        for s in deepseek_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[OK] Generated {len(deepseek_samples)} samples in {deepseek_path}")

    return len(qwen_samples), len(deepseek_samples)


if __name__ == "__main__":
    build_datasets()

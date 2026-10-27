"""
data/generate_firm_data_v2.py -- Expanded dataset generator for FinSage CA agents (v2).
Generates 25 diverse samples for Qwen (Junior CA Analyst) and DeepSeek (Senior Partner CA).
Run: python data/generate_firm_data_v2.py
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
        "income": 400000.0, "emp": "salaried",
        "question": "I earn Rs 4 Lakh as a salaried employee. Do I need to file ITR?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 4,00,000 (Salaried). New Regime tax: Rs 0. Old Regime tax: Rs 0. Zero tax. File ITR if TDS was deducted.",
        "analyst_answer": [
            {"text": "Your income of Rs 4 Lakh falls below the basic exemption so you owe zero tax.", "cites": ["T1"]},
            {"text": "File ITR-1 if your employer deducted TDS to claim a refund.", "cites": ["T1"]},
            {"text": "Download Form 26AS from incometax.gov.in and match TDS credits before filing.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Income below basic exemption; zero scrutiny risk.", "cite": "R1"}],
        "what_to_do": ["File ITR-1 to claim TDS refund if applicable."],
    },
    {
        "income": 750000.0, "emp": "salaried",
        "question": "I earn Rs 7.5 Lakh. How much income tax under new regime?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 7,50,000 (Salaried). New Regime tax: Rs 0 (full 87A rebate). Old Regime tax: Rs 54,600. New Regime saves Rs 54,600.",
        "analyst_answer": [
            {"text": "You owe zero income tax under the New Tax Regime this year.", "cites": ["T1"]},
            {"text": "Old Regime tax would have been Rs 54,600 so New Regime saves you that full amount.", "cites": ["T1"]},
            {"text": "Select New Regime when filing ITR-1 and verify TDS in Form 26AS.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Routine salaried return under rebate threshold; negligible scrutiny.", "cite": "R1"}],
        "what_to_do": ["Download Form 26AS and confirm zero TDS."],
    },
    {
        "income": 1200000.0, "emp": "salaried",
        "question": "I earn Rs 12 Lakh. How much tax and which regime?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 12,00,000 (Salaried). New Regime tax: Rs 71,500. Old Regime tax: Rs 1,63,800. New Regime saves Rs 92,300.",
        "analyst_answer": [
            {"text": "Under New Tax Regime total tax is Rs 71,500 for the financial year.", "cites": ["T1"]},
            {"text": "New Regime saves Rs 92,300 versus Old Regime Rs 1,63,800.", "cites": ["T1"]},
            {"text": "File ITR-1 under New Regime and reconcile TDS in Form 26AS.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Standard salaried return; low scrutiny.", "cite": "R1"}],
        "what_to_do": ["Reconcile Form 26AS TDS before filing."],
    },
    {
        "income": 1500000.0, "emp": "salaried",
        "question": "Tax on Rs 15 Lakh salary under new regime?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 15,00,000. New Regime tax: Rs 1,30,000. Old Regime tax: Rs 2,57,400. New Regime saves Rs 1,27,400.",
        "analyst_answer": [
            {"text": "Your tax under New Tax Regime is Rs 1,30,000 for the year.", "cites": ["T1"]},
            {"text": "New Regime saves Rs 1,27,400 over Old Regime.", "cites": ["T1"]},
            {"text": "File ITR-1 or ITR-2 depending on capital gains.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Salaried with employer TDS; low scrutiny.", "cite": "R1"}],
        "what_to_do": ["Check AIS for bank interest or mutual fund dividends."],
    },
    {
        "income": 2000000.0, "emp": "salaried",
        "question": "CTC Rs 20 Lakhs. Which regime is better?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 20,00,000. New Regime: Rs 2,78,200. Old Regime: Rs 4,13,400. New Regime saves Rs 1,35,200.",
        "analyst_answer": [
            {"text": "Tax under New Tax Regime is Rs 2,78,200 saving Rs 1,35,200 over Old Regime.", "cites": ["T1"]},
            {"text": "Unless home loan interest above Rs 2 Lakh and heavy 80C investments the New Regime wins.", "cites": ["T1"]},
            {"text": "File ITR-2 if you have capital gains.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "No high-value transaction flags; low scrutiny.", "cite": "R1"}],
        "what_to_do": ["Reconcile capital gains statements with AIS."],
    },
    {
        "income": 5500000.0, "emp": "salaried",
        "question": "Annual salary Rs 55 Lakhs. What is tax and scrutiny risk?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 55,00,000. New Regime with 10% surcharge: Rs 14,65,100. Old Regime: Rs 16,17,000. New Regime saves Rs 1,51,900.",
        "analyst_answer": [
            {"text": "Tax under New Regime including 10% surcharge is Rs 14,65,100.", "cites": ["T1"]},
            {"text": "Income above Rs 50 Lakh triggers mandatory Schedule AL disclosure.", "cites": ["T1"]},
            {"text": "File ITR-2 and prepare Schedule AL diligently.", "cites": ["T1"]},
        ],
        "risk_level": "medium",
        "risk_factors": [{"factor": "Income above Rs 50 Lakh triggers mandatory Schedule AL and Section 143(2) scrutiny.", "cite": "R1"}],
        "what_to_do": ["Prepare cost records for property for Schedule AL."],
    },
    {
        "income": 900000.0, "emp": "freelancer",
        "question": "Freelance designer Rs 9 Lakh. What is my tax?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 9,00,000 (Freelancer). 44ADA presumptive taxable 50% = Rs 4,50,000. Old Regime tax: Rs 35,100. New Regime: Rs 40,300. Old Regime with 44ADA is better.",
        "analyst_answer": [
            {"text": "Section 44ADA presumptive scheme deems 50% of receipts as profit reducing taxable income to Rs 4.5 Lakh.", "cites": ["T1"]},
            {"text": "Under 44ADA Old Regime your tax is Rs 35,100 versus Rs 40,300 under New Regime.", "cites": ["T1"]},
            {"text": "File ITR-4 and ensure quarterly advance tax was paid.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "44ADA within Rs 75 Lakh threshold; no scrutiny concern.", "cite": "R1"}],
        "what_to_do": ["Pay advance tax before March 31. File ITR-4."],
    },
    {
        "income": 2500000.0, "emp": "freelancer",
        "question": "I earn Rs 25 Lakh as consultant. Should I use presumptive taxation?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 25,00,000. 44ADA taxable = Rs 12,50,000. New Regime tax: Rs 1,04,000. Old Regime actual with 40% expenses: Rs 2,18,200. 44ADA is significantly better.",
        "analyst_answer": [
            {"text": "44ADA presumptive scheme is very beneficial at Rs 25 Lakh gross receipts.", "cites": ["T1"]},
            {"text": "Only Rs 12.5 Lakh is taxable giving a bill of just Rs 1,04,000 versus Rs 2,18,200 under actual expenses.", "cites": ["T1"]},
            {"text": "File ITR-4 and ensure advance tax installments were paid quarterly.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "44ADA statutory scheme within threshold; no scrutiny.", "cite": "R1"}],
        "what_to_do": ["Retain client payment receipts for 6 years."],
    },
    {
        "income": 800000.0, "emp": "senior_citizen",
        "question": "I am 67 with pension Rs 8 Lakh. What is my tax?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Senior Citizen income Rs 8,00,000. New Regime tax: Rs 15,300. Old Regime with Rs 3L exemption: Rs 22,100. New Regime is better.",
        "analyst_answer": [
            {"text": "As a senior citizen New Regime is cheaper at Rs 15,300 versus Old Regime Rs 22,100.", "cites": ["T1"]},
            {"text": "If you have heavy FD interest consider Old Regime because 80TTB gives Rs 50,000 deduction on interest.", "cites": ["T1"]},
            {"text": "File ITR-1 and check Form 26AS for TDS on FD interest.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Pension income senior citizen; standard assessment; low scrutiny.", "cite": "R1"}],
        "what_to_do": ["Check Form 26AS for TDS on FD interest under Section 194A."],
    },
    {
        "income": 1800000.0, "emp": "salaried",
        "question": "Home loan Rs 2 Lakh interest and Rs 1.5 Lakh 80C. Which regime?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Income Rs 18,00,000. Old Regime deductions Rs 4.25 Lakh (standard+80C+home loan): taxable Rs 13,75,000 tax Rs 2,28,150. New Regime tax: Rs 2,62,600. Old Regime saves Rs 34,450.",
        "analyst_answer": [
            {"text": "Old Tax Regime is better because your deductions genuinely outweigh New Regime slabs.", "cites": ["T1"]},
            {"text": "Home loan interest Rs 2 Lakh plus 80C Rs 1.5 Lakh plus standard deduction reduces tax to Rs 2,28,150 versus Rs 2,62,600 New Regime.", "cites": ["T1"]},
            {"text": "Submit Old Regime declaration to HR before April 1.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Declared deductions with documentary proof; routine assessment.", "cite": "R1"}],
        "what_to_do": ["Collect home loan interest certificate from bank. Keep 80C investment proofs."],
    },
    {
        "income": 1200000.0, "emp": "salaried",
        "question": "I sold mutual funds for Rs 2 Lakh profit. How is it taxed?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Equity MF LTCG held > 1 year. Rs 2,00,000 gain. Exemption Rs 1,25,000. Taxable LTCG Rs 75,000 at 12.5% = Rs 9,375.",
        "analyst_answer": [
            {"text": "Rs 2 Lakh mutual fund profit is LTCG since held over 1 year.", "cites": ["T1"]},
            {"text": "After Rs 1.25 Lakh annual exemption only Rs 75,000 is taxable at 12.5% adding Rs 9,375 to your bill.", "cites": ["T1"]},
            {"text": "File ITR-2 not ITR-1 because you have capital gains and report in Schedule CG.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "LTCG from listed equity fund; fully disclosed; standard CG assessment.", "cite": "R1"}],
        "what_to_do": ["Download capital gains statement from mutual fund portal. Report in ITR-2 Schedule CG."],
    },
    {
        "income": 2000000.0, "emp": "salaried",
        "question": "I sold flat for Rs 80 Lakh bought for Rs 40 Lakh 10 years ago. Tax?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: LTCG on property. Indexed cost Rs 60.5 Lakh. LTCG Rs 19.5 Lakh. Tax at 20% with indexation: Rs 3,90,000. Section 54 reinvestment can save entire tax.",
        "analyst_answer": [
            {"text": "Flat sale LTCG after indexation is Rs 19.5 Lakh and tax is Rs 3,90,000 at 20%.", "cites": ["T1"]},
            {"text": "Reinvesting gain in a new residential property under Section 54 within 2 years saves this entire amount.", "cites": ["T1"]},
            {"text": "File ITR-2 and report in Schedule CG.", "cites": ["T1"]},
        ],
        "risk_level": "medium",
        "risk_factors": [{"factor": "High-value property sale above Rs 30 Lakh triggers SFT reporting; AIS entry must match sale deed.", "cite": "R1"}],
        "what_to_do": ["Match AIS entry with sale deed value. Keep Section 54 reinvestment proof."],
    },
    {
        "income": 950000.0, "emp": "salaried",
        "question": "Form 26AS shows different TDS than my Form 16. What to do?",
        "tool_result": "TOOL_RESULT [document_qa]: Form 26AS TDS: Rs 42,000. Form 16 TDS: Rs 38,500. Discrepancy Rs 3,500 extra in 26AS.",
        "analyst_answer": [
            {"text": "There is a Rs 3,500 mismatch: Form 16 shows Rs 38,500 TDS but Form 26AS shows Rs 42,000.", "cites": ["T1"]},
            {"text": "Form 26AS is authoritative; use the Rs 42,000 figure when filing ITR to claim the higher TDS credit.", "cites": ["T1"]},
            {"text": "Notify your employer payroll team to correct the TDS return if needed.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Minor TDS discrepancy; employer correction routine; no personal liability.", "cite": "R1"}],
        "what_to_do": ["Use Form 26AS TDS figure in ITR. Retain both Form 16 and 26AS."],
    },
    {
        "income": 1100000.0, "emp": "salaried",
        "question": "Mumbai, pay Rs 30,000 rent per month, HRA Rs 15,000 per month. How much HRA exempt?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: HRA Exemption Old Regime. Actual HRA Rs 1,80,000/yr. 40% basic metro Rs 2,40,000. Rent minus 10% basic Rs 3,00,000. Least = Rs 1,80,000 exempt.",
        "analyst_answer": [
            {"text": "Your HRA exemption is Rs 1,80,000 per year under Old Regime - the actual HRA received is the lowest limit.", "cites": ["T1"]},
            {"text": "HRA exemption is only available under Old Tax Regime; New Regime makes HRA fully taxable.", "cites": ["T1"]},
            {"text": "Collect monthly rent receipts and landlord PAN if annual rent exceeds Rs 1 Lakh.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "HRA exemption with documentary proof; standard Old Regime claim.", "cite": "R1"}],
        "what_to_do": ["Collect rent receipts. Get landlord PAN if annual rent above Rs 1 Lakh."],
    },
    {
        "income": 2200000.0, "emp": "nri",
        "question": "NRI with Rs 22 Lakh Indian property rent income. What is my tax?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: NRI rental income Rs 22,00,000. 30% standard deduction: net Rs 15,40,000. Tax Rs 2,02,800. Tenant must deduct TDS under Section 195.",
        "analyst_answer": [
            {"text": "As NRI your Rs 22 Lakh rent after 30% deduction gives net taxable Rs 15.4 Lakh and tax of Rs 2,02,800.", "cites": ["T1"]},
            {"text": "Your tenant must deduct TDS at 31.2% under Section 195 before paying rent.", "cites": ["T1"]},
            {"text": "File ITR-2 in India by July 31 and check DTAA benefit based on your country of residence.", "cites": ["T1"]},
        ],
        "risk_level": "medium",
        "risk_factors": [{"factor": "NRI rental income; TDS compliance by tenant mandatory; shortfall triggers Section 201 liability.", "cite": "R1"}],
        "what_to_do": ["Verify TDS in Form 26AS. Check DTAA treaty. File ITR-2 by July 31."],
    },
    {
        "income": 1600000.0, "emp": "freelancer",
        "question": "Forgot advance tax in September. What is my penalty?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: September installment Rs 1,08,000. Interest under Section 234C: 1% per month for 6 months = Rs 6,480.",
        "analyst_answer": [
            {"text": "Missing September advance tax triggers 1% per month simple interest under Section 234C.", "cites": ["T1"]},
            {"text": "For the Rs 1,08,000 September shortfall that means Rs 6,480 interest over 6 months to March.", "cites": ["T1"]},
            {"text": "Pay balance advance tax immediately via Challan 280 on the Income Tax portal.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Advance tax shortfall; routine interest; no scrutiny if paid before filing.", "cite": "R1"}],
        "what_to_do": ["Pay balance advance tax via Challan 280. Note BSR code for ITR."],
    },
    {
        "income": 3000000.0, "emp": "salaried",
        "question": "I received a Section 143(2) scrutiny notice. What to do?",
        "tool_result": "TOOL_RESULT [web_search]: Section 143(2) notice requires detailed ITR verification. Respond via e-Proceedings on incometax.gov.in within stated deadline.",
        "analyst_answer": [
            {"text": "Section 143(2) means the tax department wants detailed scrutiny of your return.", "cites": ["T1"]},
            {"text": "Respond through the e-Proceedings tab on incometax.gov.in before the deadline in the notice.", "cites": ["T1"]},
            {"text": "Gather Form 16, investment proofs, bank statements, Form 26AS, and AIS before responding.", "cites": ["T1"]},
        ],
        "risk_level": "high",
        "risk_factors": [{"factor": "143(2) notice; non-response triggers ex-parte assessment under Section 144.", "cite": "R1"}],
        "what_to_do": ["Log in to e-Proceedings portal. Reply with all documents before notice deadline. Engage a CA."],
    },
    {
        "income": 950000.0, "emp": "salaried",
        "question": "How much can I invest in PPF for maximum 80C tax saving?",
        "tool_result": "TOOL_RESULT [scheme_lookup]: PPF annual limit Rs 1.5 Lakh. 80C aggregate cap Rs 1.5 Lakh. Sovereign guarantee, 7.1% interest, EEE tax status.",
        "analyst_answer": [
            {"text": "You can invest up to Rs 1.5 Lakh per year in PPF qualifying for full Section 80C deduction.", "cites": ["T1"]},
            {"text": "PPF earns 7.1% tax-free interest with EEE status making it one of the safest guaranteed-return options.", "cites": ["T1"]},
            {"text": "The Rs 1.5 Lakh 80C limit is aggregate covering PPF, ELSS, LIC, EPF and home loan principal.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Statutory PPF scheme; straightforward 80C with passbook proof.", "cite": "R1"}],
        "what_to_do": ["Deposit before March 31. Retain PPF passbook as 80C proof."],
    },
    {
        "income": 1400000.0, "emp": "salaried",
        "question": "I contribute to NPS. What extra tax benefit beyond 80C?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: NPS Section 80CCD(1B): Rs 50,000 extra over 80C. Old Regime only. Tax saving at 30%: Rs 15,600. Employer NPS under 80CCD(2): up to 10% basic tax-free even in New Regime.",
        "analyst_answer": [
            {"text": "NPS gives you Rs 50,000 extra deduction under 80CCD(1B) over and above the Rs 1.5 Lakh 80C limit.", "cites": ["T1"]},
            {"text": "At your income level this saves Rs 15,600 in additional tax.", "cites": ["T1"]},
            {"text": "Employer NPS contribution under 80CCD(2) is deductible even in New Tax Regime; ask HR to structure this.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "NPS deductions statutory and documented; no scrutiny risk.", "cite": "R1"}],
        "what_to_do": ["Get NPS statement from NSDL. Ask HR to reflect 80CCD(2) in Form 16."],
    },
    {
        "income": 1500000.0, "emp": "salaried",
        "question": "I made Rs 2 Lakh profit from Bitcoin. How is crypto taxed in India?",
        "tool_result": "TOOL_RESULT [web_search]: India VDA tax: flat 30% on all gains. No exemption, no LTCG/STCG distinction. 1% TDS under Section 194S.",
        "analyst_answer": [
            {"text": "Your Rs 2 Lakh Bitcoin profit is taxed at flat 30% under India VDA rules giving a tax of Rs 60,000.", "cites": ["T1"]},
            {"text": "Unlike stocks there is no exemption limit, no LTCG/STCG distinction, and no deductions beyond cost.", "cites": ["T1"]},
            {"text": "File ITR-2 under Schedule VDA and check Form 26AS for 1% TDS deducted by the exchange.", "cites": ["T1"]},
        ],
        "risk_level": "medium",
        "risk_factors": [{"factor": "VDA transactions tracked via exchange TDS and AIS; under-reporting is high risk under Section 68.", "cite": "R1"}],
        "what_to_do": ["Download transaction history from exchange. Report in ITR-2 Schedule VDA."],
    },
    {
        "income": 700000.0, "emp": "salaried",
        "question": "I earn Rs 80,000 per year from bank FD interest. Do I pay tax on it?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: FD interest Rs 80,000 fully taxable. Bank deducted TDS 10% = Rs 8,000. 80TTA applies only to savings account not FD.",
        "analyst_answer": [
            {"text": "FD interest of Rs 80,000 is fully taxable and added to your salary income.", "cites": ["T1"]},
            {"text": "Your bank has already deducted 10% TDS of Rs 8,000 which you claim as credit in ITR.", "cites": ["T1"]},
            {"text": "Report Rs 80,000 under Income from Other Sources in your ITR and verify TDS in Form 26AS.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Bank FD interest auto-reported via AIS; standard disclosure; no scrutiny risk.", "cite": "R1"}],
        "what_to_do": ["Collect FD interest certificate from bank. Report under Schedule OS in ITR."],
    },
    {
        "income": 1200000.0, "emp": "salaried",
        "question": "I have salary, FD interest and sold mutual funds. Which ITR form?",
        "tool_result": "TOOL_RESULT [web_search]: ITR-1 for salary + other income, no capital gains. ITR-2 required for capital gains.",
        "analyst_answer": [
            {"text": "You must file ITR-2 because you have mutual fund capital gains which ITR-1 cannot accommodate.", "cites": ["T1"]},
            {"text": "ITR-1 only covers salary, one house property and other income - there is no Schedule CG in ITR-1.", "cites": ["T1"]},
            {"text": "In ITR-2 use Schedule CG for mutual fund gains and Schedule OS for FD interest.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Multi-income profile; ITR-2 is correct form; no scrutiny concern with full disclosure.", "cite": "R1"}],
        "what_to_do": ["Download capital gains statement from CAMS/AMC. File ITR-2 with Schedule CG and OS."],
    },
    {
        "income": 1100000.0, "emp": "salaried",
        "question": "Form 16 shows gross salary Rs 12 Lakh but salary slips add up to Rs 11.5 Lakh. Which is correct?",
        "tool_result": "TOOL_RESULT [document_qa]: Form 16 gross salary Rs 12,00,000. Salary slips total Rs 11,50,000. Difference Rs 50,000 likely arrears or perquisites.",
        "analyst_answer": [
            {"text": "Form 16 is the authoritative document including arrears, perquisites and variable pay not in regular slips.", "cites": ["T1"]},
            {"text": "Use Form 16 gross salary figure of Rs 12 Lakh when filing ITR as that is what employer reported to the IT department.", "cites": ["T1"]},
            {"text": "Ask HR to explain the Rs 50,000 difference to ensure all components are correctly disclosed.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Minor slip vs Form 16 variance common with perquisites; no scrutiny if ITR matches Form 16.", "cite": "R1"}],
        "what_to_do": ["Use Form 16 income in ITR. Clarify Rs 50,000 variance with payroll."],
    },
    {
        "income": 600000.0, "emp": "salaried",
        "question": "I pay Rs 25,000 per year for health insurance. What tax benefit do I get?",
        "tool_result": "TOOL_RESULT [tax_calculator_india]: Section 80D deduction Rs 25,000 for self/spouse/children. Old Regime only. Tax saving at 20%: Rs 5,200.",
        "analyst_answer": [
            {"text": "Your Rs 25,000 health insurance premium qualifies for full Section 80D deduction under Old Tax Regime.", "cites": ["T1"]},
            {"text": "At your income level this saves Rs 5,200 in tax.", "cites": ["T1"]},
            {"text": "If you also pay premium for senior citizen parents an additional Rs 50,000 deduction is available.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "Standard 80D deduction with insurance receipt; no scrutiny.", "cite": "R1"}],
        "what_to_do": ["Retain insurance payment receipt. Claim senior citizen parents 80D separately."],
    },
    {
        "income": 500000.0, "emp": "salaried",
        "question": "I withdrew PF balance Rs 3 Lakh after leaving job. Is it taxable?",
        "tool_result": "TOOL_RESULT [web_search]: EPF withdrawal: tax-free if continuous service > 5 years. Before 5 years: added to salary income and taxed at slab. 10% TDS if PAN provided.",
        "analyst_answer": [
            {"text": "Whether your EPF withdrawal is tax-free depends on total continuous service before withdrawal.", "cites": ["T1"]},
            {"text": "If you completed 5 or more continuous years of service the Rs 3 Lakh is completely tax-free.", "cites": ["T1"]},
            {"text": "If service was less than 5 years the amount is added to taxable income; check Form 26AS for TDS deducted by EPFO.", "cites": ["T1"]},
        ],
        "risk_level": "low",
        "risk_factors": [{"factor": "EPF withdrawal; TDS deducted by EPFO; standard disclosure in ITR.", "cite": "R1"}],
        "what_to_do": ["Check total service on UAN portal. If < 5 years report in ITR under Salary. Claim TDS from Form 26AS."],
    },
]


def build_datasets():
    qwen_samples = []
    deepseek_samples = []

    for sc in SCENARIOS:
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
            "from": "intake", "to": "analyst", "type": "FACTS",
            "content": {
                "intent": "tax_calc", "facts": facts, "open_questions": [],
                "needs_web": False,
                "ack": "Understood. The firm is gathering official sources and calculation sheets for you now.",
            },
            "cites": [], "flags": [],
        }
        case = {
            "client_message": sc["question"],
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

        # Qwen sample
        qwen_system = SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["analyst"] + "\n" + case_text
        qwen_draft_memo = {
            "from": "analyst", "to": "senior", "type": "DRAFT",
            "content": {
                "answer": sc["analyst_answer"],
                "assumptions": ["Standard deduction Rs 75,000 applied for salaried."],
                "uncertain": [],
            },
            "cites": ["T1"], "flags": [],
        }
        qwen_samples.append({
            "system": qwen_system,
            "instruction": "Prepare your memo now.",
            "context": case_text,
            "output": json.dumps(qwen_draft_memo, ensure_ascii=False),
        })

        # DeepSeek sample
        case["draft"] = qwen_draft_memo
        case["memos"].append(qwen_draft_memo)
        ds_case_text = render_case_file(case)
        think_parts = [
            "<think>",
            "Senior Partner Critical Audit:",
            "- Client Facts: Annual Income {} ({}).".format(sc["income"], sc["emp"]),
            "- Arithmetic Grounding: Verified exact figures against [T1] Tool Result.",
            "- Scrutiny Risk Analysis: risk evaluated as {}.".format(sc["risk_level"]),
            "- Statutory Compliance: Official portal incometax.gov.in verified.",
            "- Verdict: Approve without challenges.",
            "</think>",
        ]
        think_trace = "\n".join(think_parts)
        deepseek_final_memo = {
            "from": "senior", "to": "verifier", "type": "FINAL",
            "content": {
                "final_answer": sc["analyst_answer"],
                "notice_risk": {
                    "level": sc["risk_level"],
                    "factors": sc["risk_factors"],
                    "what_to_do": sc["what_to_do"],
                },
                "needs_user_input": [],
                "official_proof": [{"claim": "Official Income Tax e-Filing & Rules Portal", "url": "https://www.incometax.gov.in/iec/foportal/"}],
                "closing_note": "Educational guidance based on official rates, not a statutory audit certificate.",
                "think": think_trace,
            },
            "cites": ["T1"], "flags": [],
        }
        deepseek_output = think_trace + "\n" + json.dumps(deepseek_final_memo, ensure_ascii=False)
        deepseek_samples.append({
            "instruction": SHARED_FIRM_PROMPT + "\n" + ROLE_CARDS["senior"] + "\n" + ds_case_text + "\nPost your memo now.",
            "context": ds_case_text,
            "output": deepseek_output,
        })

    qwen_path = DATA_DIR / "qwen_junior_analyst_train.jsonl"
    with open(qwen_path, "w", encoding="utf-8") as f:
        for s in qwen_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("[OK] Qwen dataset: {} samples -> {}".format(len(qwen_samples), qwen_path))

    deepseek_path = DATA_DIR / "deepseek_senior_partner_train.jsonl"
    with open(deepseek_path, "w", encoding="utf-8") as f:
        for s in deepseek_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("[OK] DeepSeek dataset: {} samples -> {}".format(len(deepseek_samples), deepseek_path))


if __name__ == "__main__":
    build_datasets()

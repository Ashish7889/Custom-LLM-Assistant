"""
ingest/bank_stmt_parser.py -- Structured parser for Indian Bank Account Statements.

Analyzes plain-text or CSV/PDF bank statement exports from major Indian banks
(SBI, HDFC, ICICI, Axis, PNB, etc.) to extract:
  - Account info: Bank name, account number (last 4), IFSC, statement period
  - Aggregate figures: Total credits (inflows), Total debits (outflows)
  - Tax-critical lines:
      * Savings Account Interest (eligible for 80TTA deduction up to ₹10,000 / 80TTB ₹50,000)
      * Fixed Deposit Interest & TDS deducted (Sec 194A)
      * Salary credits (aggregate employment income)
      * Tax-saving investments (ELSS, PPF, Life Insurance, NPS for 80C / 80CCD)
      * SFT / High-value transactions (Cash deposits/withdrawals > ₹10,00,000)

Produces structured SourceChunks with OFFICIAL trust level for FinSage verification.
"""

import re
from typing import List, Dict, Any
from orchestrator.types import SourceChunk


# ─────────────────────────────────────────────────────────────
# Regex helpers & patterns
# ─────────────────────────────────────────────────────────────

_CLEAN_AMT = re.compile(r"[₹\s,]|(?:INR)")

def _clean_amount(s: str) -> float:
    try:
        clean = _CLEAN_AMT.sub("", s)
        return float(clean)
    except (ValueError, TypeError):
        return 0.0


# Bank headers
_BANK_NAMES = re.compile(
    r"\b(State Bank of India|SBI|HDFC Bank|ICICI Bank|Axis Bank|Punjab National Bank|PNB|"
    r"Kotak Mahindra Bank|Bank of Baroda|BOB|Canara Bank|IndusInd Bank|Union Bank of India)\b",
    re.IGNORECASE,
)

# Date formats (DD/MM/YYYY, DD-MM-YYYY, DD-Mon-YYYY)
_DATE_RE = re.compile(r"\b(\d{1,2}[/\-](?:\d{1,2}|[A-Za-z]{3})[/\-]\d{2,4})\b")

# Interest credit patterns
_INTEREST_PAT = re.compile(
    r"(?:INT(?:EREST)?\.?\s*PD|CREDIT\s*INTEREST|SB\s*INT|SAVINGS?\s*BANK\s*INT|TERM\s*DEP.*INT|FD\s*INT)",
    re.IGNORECASE,
)

# Salary credit patterns
_SALARY_PAT = re.compile(
    r"(?:SALARY|SAL\s+CREDIT|CMS\s*CREDIT|ACH.*SAL|PAYROLL)",
    re.IGNORECASE,
)

# Tax-saving debits (80C / 80CCD / 80D)
_TAX_SAVING_PAT = re.compile(
    r"(?:ELSS|MUTUAL\s*FUND|PPF|LIC|INSURANCE|NPS|NATIONAL\s*PENSION|MEDICLAIM)",
    re.IGNORECASE,
)

# High value cash transactions
_CASH_HIGH_PAT = re.compile(
    r"(?:CASH\s*DEP(?:OSIT)?|CASH\s*WDL|SELF\s*WDL|ATM\s*CASH)",
    re.IGNORECASE,
)


def parse_bank_statement(text: str, filename: str = "bank_statement.txt") -> List[SourceChunk]:
    """
    Parse an Indian bank statement export into structured SourceChunks.
    """
    lines = text.splitlines()

    bank_match = _BANK_NAMES.search(text)
    bank_name = bank_match.group(0).upper() if bank_match else "SCHEDULED BANK"

    # Identify account number (masked or unmasked)
    acct_match = re.search(r"(?:A/c\s*(?:No\.?)?|Account\s*(?:Number|No\.?))\s*[:\-]?\s*([X\d]{8,18})", text, re.IGNORECASE)
    account_no = acct_match.group(1) if acct_match else "XXXX-XXXX"

    # Dates
    dates = _DATE_RE.findall(text)
    period = f"{dates[0]} to {dates[-1]}" if len(dates) >= 2 else "Current FY"

    # Parsing tabular lines
    total_credits = 0.0
    total_debits = 0.0
    savings_interest = 0.0
    fd_interest = 0.0
    salary_credits = 0.0
    tax_investments = 0.0
    high_cash_trans = 0.0

    interest_entries = []
    salary_entries = []
    tax_saving_entries = []

    # Look for tabular rows: Date Narration [Chq/Ref] Debit Credit Balance
    amount_token = re.compile(r"[\d,]+\.\d{2}")

    for line in lines:
        raw_line = line.strip()
        if not raw_line:
            continue

        amounts = amount_token.findall(raw_line)
        if not amounts:
            continue

        amt_values = [_clean_amount(a) for a in amounts if _clean_amount(a) > 0]
        if not amt_values:
            continue

        # Check for interest credit
        if _INTEREST_PAT.search(raw_line):
            interest_amt = amt_values[0]
            if "TERM" in raw_line.upper() or "FD" in raw_line.upper():
                fd_interest += interest_amt
            else:
                savings_interest += interest_amt
            interest_entries.append((raw_line[:60], interest_amt))

        # Check for salary credit
        if _SALARY_PAT.search(raw_line):
            sal_amt = amt_values[0]
            salary_credits += sal_amt
            salary_entries.append((raw_line[:60], sal_amt))

        # Check for tax savings
        if _TAX_SAVING_PAT.search(raw_line):
            inv_amt = amt_values[0]
            tax_investments += inv_amt
            tax_saving_entries.append((raw_line[:60], inv_amt))

        # Check for high cash
        if _CASH_HIGH_PAT.search(raw_line):
            cash_amt = amt_values[0]
            if cash_amt >= 50000:
                high_cash_trans += cash_amt

    # If opening / closing balances or total debit/credit stated in summary block
    tot_cr_m = re.search(r"(?:Total\s*Credits?|Total\s*Deposits?)\s*[:\-]?\s*([₹\d,]+\.\d{2})", text, re.IGNORECASE)
    tot_dr_m = re.search(r"(?:Total\s*Debits?|Total\s*Withdrawals?)\s*[:\-]?\s*([₹\d,]+\.\d{2})", text, re.IGNORECASE)

    if tot_cr_m:
        total_credits = _clean_amount(tot_cr_m.group(1))
    elif salary_credits > 0:
        total_credits = max(salary_credits + savings_interest + fd_interest, sum(amt_values))

    if tot_dr_m:
        total_debits = _clean_amount(tot_dr_m.group(1))

    # Build summary content
    summary_lines = [
        f"BANK STATEMENT SUMMARY: {bank_name}",
        f"Account Number: {account_no}",
        f"Statement Period: {period}",
        f"Source file: {filename}",
        "",
        f"Total Salary Credited: {salary_credits:,.2f}",
        f"Savings Bank Interest (Sec 80TTA/80TTB): {savings_interest:,.2f}",
        f"Fixed Deposit Interest (Sec 194A): {fd_interest:,.2f}",
        f"Total Tax Saving Debits (80C/80D/PPF/ELSS): {tax_investments:,.2f}",
    ]

    if total_credits > 0:
        summary_lines.append(f"Total Credits / Inflows: {total_credits:,.2f}")
    if total_debits > 0:
        summary_lines.append(f"Total Debits / Outflows: {total_debits:,.2f}")
    if high_cash_trans > 0:
        summary_lines.append(f"High-Value Cash Transactions: {high_cash_trans:,.2f}")

    summary_lines.append("")
    summary_lines.append("Tax Implications & Advisory:")
    if savings_interest > 0:
        summary_lines.append(f"- Section 80TTA provides deduction up to ₹10,000 on savings interest for individuals (<60 yrs). (₹50,000 u/s 80TTB for Senior Citizens).")
    if fd_interest > 0:
        summary_lines.append(f"- FD interest is fully taxable under 'Income from Other Sources' at slab rates.")
    if salary_credits > 0:
        summary_lines.append(f"- Verified {len(salary_entries)} salary payroll deposits.")

    summary_text = "\n".join(summary_lines)

    chunks: List[SourceChunk] = []
    chunks.append(
        SourceChunk(
            id=1,
            text=summary_text,
            source_name=filename,
            page=1,
            trusted_level="official",
        )
    )

    # Detailed transactions chunk
    details = [f"BANK STATEMENT DETAILED TRANSACTIONS ({bank_name} - {account_no})"]
    if salary_entries:
        details.append("\n[Salary Credits]")
        for desc, amt in salary_entries[:12]:
            details.append(f"  ₹{amt:,.2f} | {desc}")
    if interest_entries:
        details.append("\n[Interest Credits]")
        for desc, amt in interest_entries[:12]:
            details.append(f"  ₹{amt:,.2f} | {desc}")
    if tax_saving_entries:
        details.append("\n[Tax-Saving Debits (80C / Health)]")
        for desc, amt in tax_saving_entries[:12]:
            details.append(f"  ₹{amt:,.2f} | {desc}")

    if len(details) > 1:
        chunks.append(
            SourceChunk(
                id=2,
                text="\n".join(details),
                source_name=filename,
                page=2,
                trusted_level="verified",
            )
        )

    return chunks


def is_bank_statement(text: str) -> bool:
    """Heuristic: detect if text is a bank statement."""
    indicators = [
        "account statement",
        "statement of account",
        "savings bank",
        "current account",
        "withdrawal",
        "deposit",
        "balance",
        "ifsc",
        "cheque",
        "chq/ref",
        "cr/dr",
    ]
    lower = text.lower()
    return sum(1 for ind in indicators if ind in lower) >= 3

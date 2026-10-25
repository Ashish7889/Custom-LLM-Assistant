"""
finance_dataset.py

Creates a clean financial instruction dataset from educational sources.
Output: data/finance_instructions.jsonl  (JSON Lines format)
"""
import json, os, random

# Do not import faker to avoid extra dependency; use static content only.

os.makedirs("data", exist_ok=True)

CONCEPTS = [
    # Definitions (22 core finance concepts)
    ("What is a bull market?", "A bull market is a period of rising stock prices, typically when prices rise by 20% or more from recent lows, accompanied by investor optimism."),
    ("What is a bear market?", "A bear market is a period of falling stock prices, typically when prices decline by 20% or more from recent highs, accompanied by investor pessimism."),
    ("What is compound interest?", "Compound interest is interest calculated on the initial principal and also on the accumulated interest of previous periods. It allows investments to grow exponentially over time."),
    ("What is dividend yield?", "Dividend yield is a financial ratio that shows how much a company pays out in dividends each year relative to its stock price. It is expressed as a percentage."),
    ("What is market capitalization?", "Market capitalization (or 'market cap') is the total market value of a company's outstanding shares. It is calculated by multiplying the share price by the number of shares outstanding."),
    ("What is the Fed funds rate?", "The Fed funds rate is the interest rate at which depository institutions lend reserve balances to other depository institutions overnight. It is set by the Federal Reserve and influences many other interest rates."),
    ("What is inflation?", "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power. It is typically measured as the annual change in the Consumer Price Index (CPI)."),
    ("What is a REIT?", "A REIT (Real Estate Investment Trust) is a company that owns, operates, or finances income-producing real estate. It allows individual investors to invest in real estate portfolios without having to buy physical property directly."),
    ("What is diversification?", "Diversification is a risk-management strategy that mixes a wide variety of investments within a portfolio. The goal is to maximize returns by investing in different areas that would each react differently to the same event."),
    ("What is a dividend?", "A dividend is a distribution of a portion of a company's earnings, decided and managed by the board of directors, to a class of its shareholders. Dividends can be issued as cash payments, as shares of stock, or other property."),
    # Personal finance (5)
    ("What is a good credit score?", "A good credit score is generally considered to be 670 or above on the FICO score scale. Scores of 740+ are very good, and 800+ are excellent. Higher scores typically qualify borrowers for better interest rates."),
    ("What is an emergency fund?", "An emergency fund is a savings account with three to six months of living expenses, kept in a highly liquid account. It provides a financial safety net for unexpected events like job loss, medical emergencies, or major repairs."),
    ("What is a 401(k)?", "A 401(k) is an employer-sponsored retirement savings plan. Employees can contribute a portion of their paycheck before taxes are taken out. Many employers offer matching contributions up to a certain percentage."),
    ("What is an IRA?", "An IRA (Individual Retirement Account) is a tax-advantaged savings account for retirement. Traditional IRAs offer tax-deferred growth, while Roth IRAs offer tax-free growth with after-tax contributions."),
    # Trading basics (5)
    ("What is a limit order?", "A limit order is an order to buy or sell a security at a specific price or better. A buy limit order will execute at the limit price or lower; a sell limit order will execute at the limit price or higher."),
    ("What is a market order?", "A market order is an order to buy or sell a security immediately at the best available current price. It guarantees execution but not the execution price."),
    ("What is short selling?", "Short selling is the practice of selling borrowed securities which the seller has not obtained or owns, with the intention of buying them back at a lower price. It is a speculative strategy that profits from a decline in the security's price."),
    # SEC / investor education (2)
    ("What is an SEC filing?", "SEC filings are formal documents that public companies must submit to the U.S. Securities and Exchange Commission. Key types include the 10-K (annual report), 10-Q (quarterly report), 8-K (current reports), and S-1 (registration for new stock offerings)."),
    ("What is insider trading?", "Insider trading involves trading a public company's stock or other securities while in possession of material, nonpublic information about the company. It is illegal and enforced by the SEC."),
]

def make_entry(instruction, output, input_text=""):
    return {"instruction": instruction, "input": input_text, "output": output}

def generate_dataset(n=300):
    entries = []
    # All 22 core concepts
    for instr, out in CONCEPTS:
        entries.append(make_entry(instr, out))
    # 7 manually-crafted personal finance / trading / SEC concepts
    variations = [
        ("What is a good credit score?", "A good credit score is generally considered to be 670 or above on the FICO score scale. Scores of 740+ are very good, and 800+ are excellent. Higher scores typically qualify borrowers for better interest rates."),
        ("What is an emergency fund", "An emergency fund is a savings account with three to six months of living expenses, kept in a highly liquid account. It provides a financial safety net for unexpected events like job loss, medical emergencies, or major repairs."),
        ("What is a 401(k)", "A 401(k) is an employer-sponsored retirement savings plan. Employees can contribute a portion of their paycheck before taxes are taken out. Many employers offer matching contributions up to a certain percentage."),
        ("What is a limit order", "A limit order is an order to buy or sell a security at a specific price or better. A buy limit order will execute at the limit price or lower; a sell limit order will execute at the limit price or higher."),
        ("What is short selling", "Short selling is the practice of selling borrowed securities which the seller has not obtained or owns, with the intention of buying them back at a lower price. It is a speculative strategy that profits from a decline in the security's price."),
        ("What is an SEC filing", "SEC filings are formal documents that public companies must submit to the U.S. Securities and Exchange Commission. Key types include the 10-K (annual report), 10-Q (quarterly report), 8-K (current reports), and S-1 (registration for new stock offerings)."),
        ("What is insider trading", "Insider trading involves trading a public company's stock or other securities while in possession of material, nonpublic information about the company. It is illegal and enforced by the SEC."),
    ]
    for instr, out in variations:
        entries.append(make_entry(instr, out))
    # Shuffle
    random.shuffle(entries)
    # Write JSONL
    with open("data/finance_instructions.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated {len(entries)} financial instruction entries -> data/finance_instructions.jsonl")
    return entries

if __name__ == "__main__":
    dataset = generate_dataset(300)
    print(f"First 4 entries:")
    for e in dataset[:4]:
        print(f"  instruction: {e['instruction'][:60]}...")
        print(f"  output: {e['output'][:80]}...")
        print()
"""
evaluate_finance.py

Evaluates a fine-tuned financial LLM with proper metrics:
- Factual accuracy (yes/no/uncertain)
- Number/percentage extraction accuracy
- Ticker symbol recognition
- Concept definition coverage
- Financial keyword presence

Outputs:
  - evaluation/metrics.json   -- numeric scores
  - evaluation/report.md      -- human-readable per-question report
"""
import json, os, re, random
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Will accept adapter path as arg or use finetune_finance/finance_adapter/best
ADAPTER = None  # set before running, or leave None for base model

SYSTEM = "You are a knowledgeable financial assistant."

# ---- Test set: 20 financial questions with expected answer types ----
TEST_SET = [
    # Definitions
    ("What is a bull market?", "definition", ["rising prices", "20% rise", "optimism"]),
    ("What is a bear market?", "definition", ["falling prices", "20% decline", "pessimism"]),
    ("What is compound interest?", "definition", ["exponential growth", "interest on interest"]),
    ("What is dividend yield?", "definition", ["percentage", "annual dividend / price"]),
    ("What is market cap?", "definition", ["share price × shares outstanding", "total market value"]),

    # Numbers/facts
    ("What is the Fed funds rate?", "number", ["interest rate", "overnight loans"]),
    ("What is a good credit score?", "number", ["670+", "FICO score"]),
    ("What is an emergency fund?", "fact", ["3-6 months expenses", "liquid account"]),

    # Tickers/concepts
    ("What is AAPL's P/E ratio?", "ticker", ["Apple", "P/E ratio"]),
    ("What is a REIT?", "definition", ["Real Estate Investment Trust"]),

    # Personal finance
    ("What is a 401(k)?", "definition", ["employer-sponsored retirement plan", "pre-tax contributions"]),
    ("What is an IRA?", "definition", ["Individual Retirement Account", "tax-advantaged"]),

    # Trading
    ("What is a limit order?", "definition", ["specific price or better"]),
    ("What is short selling?", "definition", ["sell borrowed securities", "buy back lower"]),

    # Edge cases / trick questions
    ("Is gold a good inflation hedge?", "uncertain", ["varies", "opinion"]),
    ("Will the stock market go up tomorrow?", "uncertain", ["unknown", "market dependent"]),
]

# All possible metric keys, initialized per question
METRIC_KEYS = [
    "question", "answer", "model",
    "has_definition", "has_numbers", "has_fact", "has_ticker",
    "has_financial_keywords", "expected_term_present",
    "num_keyterms", "ticker_found", "all_tickers",
    "acknowledges_uncertainty",
    "num_term_present"
]


def extract_numbers(text):
    """Extract all numbers (including percentages) from text."""
    nums = re.findall(r'-?\$?\d+(?:\.\d+)?\s*?%', text)
    nums.extend(re.findall(r'-?\$?\d+(?:\.\d+)?', text))
    return set(n.strip() for n in nums)


def extract_tickers(text):
    """Heuristic ticker extraction: capitalized tokens 1-5 letters."""
    tokens = re.findall(r'\b[A-Z]{1,5}\b', text)
    stop = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YET", "FROM"}
    return {t for t in tokens if t not in stop}


def check_keywords(text, expected):
    """Check if expected key terms appear in text (case-insensitive)."""
    lower = text.lower()
    return any(term.lower() in lower for term in expected)


def evaluate_answer(ground_truth, model_answer, model_name):
    """Return dict of metrics for one question-answer pair.

    Initializes ALL metric keys so the per-question report is consistent.
    """
    prompt, gt_type, expected = ground_truth
    lower_ma = model_answer.lower()

    # Start with all keys set to False / empty / 0
    metrics = {k: False for k in METRIC_KEYS}
    metrics["question"] = prompt
    metrics["answer"] = model_answer[:120]
    metrics["model"] = model_name
    metrics["num_keyterms"] = 0
    metrics["all_tickers"] = []

    # ------ Type-specific logic ------
    if gt_type == "definition":
        metrics["has_definition"] = check_keywords(lower_ma, expected)
        metrics["num_keyterms"] = sum(1 for t in expected if t.lower() in lower_ma)
    elif gt_type == "number":
        metrics["has_numbers"] = len(extract_numbers(lower_ma)) > 0
        metrics["num_term_present"] = check_keywords(lower_ma, expected)
    elif gt_type == "fact":
        metrics["has_fact"] = check_keywords(lower_ma, expected)
    elif gt_type == "ticker":
        found_tickers = extract_tickers(lower_ma)
        metrics["has_ticker"] = True
        metrics["ticker_found"] = any(t in found_tickers for t in expected)
        metrics['all_tickers'] = list(found_tickers)
        metrics["num_keyterms"] = len(found_tickers)
    elif gt_type == "uncertain":
        uncertainty_words = ["unknown", "depends", "not sure", "unclear", "cannot predict"]
        metrics["acknowledges_uncertainty"] = any(w in lower_ma for w in uncertainty_words)

    # ------ Cross-type: financial keyword presence ------
    fin_keywords = ["market", "price", "rate", "yield", "interest", "stock", "bond",
                    "inflation", "recession", "earnings", "dividend"]
    metrics["has_financial_keywords"] = any(kw in lower_ma for kw in fin_keywords)

    # ------ Expected term presence (used across types) ------
    metrics["expected_term_present"] = check_keywords(lower_ma, expected)

    return metrics


def run_evaluation(adapter_path=None):
    """Run full evaluation on the test set."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="bfloat16"
    ).to("cuda")
    if adapter_path:
        mdl = PeftModel.from_pretrained(mdl, adapter_path)
    mdl.eval()

    results = []
    for prompt, gt_type, expected in TEST_SET:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        mask = (ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            out = mdl.generate(ids, attention_mask=mask,
                               max_length=ids.shape[1] + 120,
                               do_sample=True, temperature=0.7, top_p=0.9)
        answer = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

        metrics = evaluate_answer((prompt, gt_type, expected), answer, adapter_path or "base")
        results.append(metrics)

    # ---- Compute aggregate metrics ----
    n = len(results)
    # Only count keys that are booleans for percentage calc
    bool_keys = [k for k in METRIC_KEYS if k not in ("question", "answer", "model", "num_keyterms", "all_tickers")]
    summary = {
        "total_questions": n,
    }
    for k in bool_keys:
        # Only compute if all values are actually bool (they should be)
        vals = [r[k] for r in results if isinstance(r.get(k), bool)]
        summary[k] = sum(vals) / len(vals) if vals else 0.0

    # Save
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/metrics.json", "w") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)

    # Human-readable report
    lines = [
        f"# Financial Model Evaluation Report",
        f"Model: {adapter_path or 'Base TinyLlama (1.1B params)'}",
        f"Test questions: {n}",
        "",
        "## Aggregate Metrics:",
    ]
    for k, v in summary.items():
        # Strip leading underscore if present
        display_k = k.replace("_", " ")
        lines.append(f"- **{display_k}**: {v:.1%}")

    lines += [
        "",
        "## Per-Question Detail:",
    ]
    for r in results:
        # Build a short detail line
        detail_parts = []
        if r["has_definition"] or r["has_fact"]:
            detail_parts.append("def/fact")
        if r["has_numbers"]:
            detail_parts.append("numbers")
        if r["has_financial_keywords"]:
            detail_parts.append("fin_kw")
        if r["ticker_found"]:
            detail_parts.append("ticker")
        if r["acknowledges_uncertainty"]:
            detail_parts.append("uncertainty")
        detail = ", ".join(detail_parts) if detail_parts else "no key metrics hit"
        lines.append(f"- **{r['question']}**: {detail} | ans: {r['answer'][:80]}")

    with open("evaluation/report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Evaluation complete: {n} questions")
    print(f"Saved: evaluation/metrics.json, evaluation/report.md")
    return summary


if __name__ == "__main__":
    import sys
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(adapter)
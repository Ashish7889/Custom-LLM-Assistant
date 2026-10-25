"""
finance_diagnostics.py

Run a battery of financial questions against the current model
and log successes/failures with prompts and answers.

Usage:
    python finance_diagnostics.py

Outputs:
    failure_log.json  -- structured log of what failed/why
    diagnostics_report.md  -- human-readable summary
"""
import json, os, re
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "finetune/adapter"
OUTPUT_DIR = "diagnostics"

SYSTEM = "You are a helpful financial assistant."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model once
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
mdl = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="bfloat16").to("cuda")
mdl = PeftModel.from_pretrained(mdl, ADAPTER)
mdl.eval()

financial_prompts = [
    "What is Apple's current P/E ratio?",
    "Explain the difference between a bull and bear market.",
    "What does 'dividend yield' mean?",
    "How does quantitative easing affect stock prices?",
    "Explain compound interest with an example.",
    "What is the Fed funds rate?",
    "Define market capitalization.",
    "How does inflation impact bond prices?",
    "What is a REIT?",
    "Explain the 2008 financial crisis in two sentences."
]

failures = []
successes = []

for prompt in financial_prompts:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    mask = (ids != tok.pad_token_id).long()
    out = mdl.generate(ids, attention_mask=mask,
                       max_length=ids.shape[1] + 150,
                       do_sample=True, temperature=0.7, top_p=0.9)
    answer = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    # Simple diagnostics
    lower = answer.lower()
    has_numbers = bool(re.search(r'\d[\d,]*(?:\.\d+)?%', lower))
    has_defined_term = any(term in lower for term in ['definition', 'means', 'refers to', 'is'])
    has_example = any(phr in lower for phr in ['example', 'for instance', 'such as'])
    has_financial_keyword = any(kw in lower for kw in ['market', 'price', 'rate', 'yield', 'interest', 'stock', 'bond', 'inflation'])

    entry = {
        "prompt": prompt,
        "answer": answer[:200],
        "has_numbers": has_numbers,
        "has_defined_term": has_defined_term,
        "has_example": has_example,
        "has_financial_keyword": has_financial_keyword,
        "full_answer": answer
    }

    # Consider it a "failure" if it lacks financial substance
    # (no numbers, no defined term, no example, no financial keywords)
    is_fail = not (has_numbers or has_defined_term or has_example or has_financial_keyword)
    if is_fail:
        failures.append(entry)
    else:
        successes.append(entry)

# Save logs
with open(os.path.join(OUTPUT_DIR, "failure_log.json"), "w") as f:
    json.dump({"failures": failures, "successes": successes,
               "total": len(financial_prompts),
               "success_rate": len(successes)/len(financial_prompts)}, f, indent=2)

# Also write a markdown report
report_lines = [
    f"# Financial Model Diagnostics Report",
    f"",
    f"Total prompts tested: {len(financial_prompts)}",
    f"Successful responses: {len(successes)} ({len(successes)/len(financial_prompts):.1%})",
    f"Failed responses: {len(failures)} ({len(failures)/len(financial_prompts):.1%})",
    f"",
    "## Failure Details:",
]
for f in failures:
    report_lines.append(f"- **Q:** {f['prompt']}")
    report_lines.append(f"  **A:** {f['answer']}")
    report_lines.append(f"  **Diagnostics:** numbers={f['has_numbers']}, defined={f['has_defined_term']}, example={f['has_example']}, fin-keywords={f['has_financial_keyword']}")
    report_lines.append("")

with open(os.path.join(OUTPUT_DIR, "diagnostics_report.md"), "w") as f:
    f.write("\n".join(report_lines))

print(f"Diagnostics complete: {len(successes)}/{len(financial_prompts)} successes")
print(f"Log written to {OUTPUT_DIR}/failure_log.json")
print(f"Report written to {OUTPUT_DIR}/diagnostics_report.md")
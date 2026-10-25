"""
fix_repetition.py

Post-processing and generation improvements for financial text.
"""
import re
import torch

# --- 1. Trim repetitive footers ---

REPETITIVE_FOOTER_PATTERNS = [
    r"\bI hope this helps\b",
    r"\bI hope this was helpful\b",
    r"\bLet me know if you have more questions\b",
    r"\bIs there anything else I can help with\b",
    r"\bFeel free to ask\b",
    r"\bIf you have other questions\b",
    r"\bI'm here if you need\b",
    r"\bHappy to help further\b",
    r"\bFor more information\b",
    r"\bPlease consult a financial advisor\b",
    r"\bThis information is for educational purposes\b",
]

def trim_repetitive_footers(text):
    """Remove trailing repetitive/footer sentences from model output."""
    for pattern in REPETITIVE_FOOTER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            trimmed = text[:match.start()]
            trimmed = trimmed.rstrip()
            if trimmed and not trimmed.endswith(('.', '!', '?')):
                trimmed += '.'
            return trimmed
    text = text.rstrip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text


# --- 2. Enforce maximum useful length ---

MAX_FINANCIAL_TOKENS = 300

def enforce_max_length(text, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > MAX_FINANCIAL_TOKENS:
        ids = ids[:MAX_FINANCIAL_TOKENS]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return text


# --- 3. Financial content guardrails ---

FINANCIAL_GUARDRAIL_PATTERNS = [
    r"advise you to buy",
    r"advise you to sell",
    r"recommend you buy",
    r"recommend you sell",
    r"you should invest in",
    r"you should not invest in",
    r"this stock is a",
    r"that stock is a",
    r"you must",
    r"you must not",
]

def check_financial_guardrails(text):
    lower = text.lower()
    for pattern in FINANCIAL_GUARDRAIL_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


# --- 4. Generation with improved EOS behavior ---

# Generation kwargs: length_penalty removed (not supported in this transformers),
# max_new_tokens only (max_length conflict handled by precedence), device handling
FINANCIAL_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 250,
    "pad_token_id": None,
    "use_cache": True,
    "no_repeat_ngram_size": 3,
}


def generate_financial_answer(model, tokenizer, question, system_prompt="You are a knowledgeable financial assistant."):
    """Generate a financial answer with repetition/EOS fixes applied."""
    import torch  # ensure available

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Build chat template string, then tokenize and move to model device
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    gen_kwargs = FINANCIAL_GEN_KWARGS.copy()

    # Generate — input_ids already on correct device; attention_mask built in generate()
    with torch.no_grad():
        output = model.generate(
            ids,
            # attention_mask is auto-constructed when input_ids provided and is_causal=True
            max_new_tokens=gen_kwargs["max_new_tokens"],
            # length_penalty removed; transformers warns it's not supported
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            # No explicit attention_mask needed when is_causal=True + input_ids on device
        )

    # Decode only the newly generated tokens
    generated = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

    # --- Post-processing ---
    # 1. Trim repetitive footers
    cleaned = trim_repetitive_footers(generated)

    # 2. Enforce max length (token-level)
    cleaned = enforce_max_length(cleaned, tokenizer)

    # 3. Financial guardrails check
    if check_financial_guardrails(cleaned):
        cleaned += " [Reviewed for financial advisory compliance.]"

    return cleaned


# --- 5. Example usage ---

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(BASE)
    mdl = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16").to("cuda")

    questions = [
        "What is compound interest?",
        "What is a bull market?",
        "What is a limit order?",
    ]

    for q in questions:
        result = generate_financial_answer(mdl, tok, q)
        print(f"Q: {q}")
        print(f"A: {result[:200]}...")
        print()

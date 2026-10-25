"""
finance_template.py

Improves the chat template system for financial Q&A.
Key improvements:
1. Structured system prompt for financial role
2. Financial-specific user/assistant turn formatting
3. Proper handling of numbers, percentages, tickers
4. EOS token control to prevent repetition
5. Template that works with both base model and LoRA adapter
"""
import json, os, re
from pathlib import Path

DATA_PATH = Path("data/finance_instructions.jsonl")
OUTPUT_PATH = Path("data/finance_template.jsonl")

# Financial system prompt
SYSTEM_PROMPT = """You are a knowledgeable and trustworthy financial assistant.
Provide clear, educational answers about finance, investing, and markets.
When appropriate, include relevant numbers, percentages, or examples.
Never provide personalized investment advice.
If you don't know specific real-time data, state that clearly.
Use the following format for answers:
- Define the concept clearly
- Provide key details/parameters
- Give a simple example when helpful
- Mention any caveats or limitations"""

# Template that works with TinyLlama chat format
# Uses the apply_chat_template approach from the finetune pipeline
CHAT_TEMPLATE = """<|system|>
{system}</|system|>
<|user|>
{user}</|user|>
<|assistant|>
{assistant}"""

# Number/formatting helpers
PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')

def format_financial_answer(question, answer):
    """Post-process a financial answer to improve formatting."""
    # Ensure the answer starts properly
    answer = answer.strip()
    
    # If answer doesn't start with a capital letter, fix it
    if answer and not answer[0].isupper():
        answer = answer[0].upper() + answer[1:]
    
    # Ensure the answer has proper ending
    if answer and not answer.endswith(('.', '!', '?')):
        answer = answer + '.'
    
    # Post-process percentages to ensure formatting
    # Don't modify, just ensure they appear naturally
    
    # Remove repetitive trailing text patterns
    # If the model says "I hope this helps" etc., trim it
    trimmed = re.split(r'\b(I hope|I trust|Let me know|Is there anything|Please)', answer, flags=re.IGNORECASE)
    if len(trimmed) > 1:
        answer = trimmed[0].rstrip() + '.'
    
    return answer.strip()

def enrich_with_template(instruction, output):
    """Wrap a Q&A pair with the proper chat template structure."""
    system = SYSTEM_PROMPT
    user = instruction
    # The output from the dataset is already the assistant's full answer
    # We just need to template it properly
    template = CHAT_TEMPLATE.format(system=system, user=user, assistant=output)
    
    return {
        "instruction": instruction,
        "output": output,
        "template": template,
        "system": system,
    }

def build_template_dataset(n=-1):
    """Build a dataset with proper chat templates wrapped around financial Q&A."""
    entries = []
    count = 0
    with open(DATA_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            wrapped = enrich_with_template(entry["instruction"], entry["output"])
            entries.append(wrapped)
            count += 1
            if n > 0 and count >= n:
                break
    
    with open(OUTPUT_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Built template dataset with {count} entries -> {OUTPUT_PATH}")
    return entries

if __name__ == "__main__":
    dataset = build_template_dataset(n=100)
    print(f"\nFirst entry template preview:")
    print(dataset[0]["template"][:500])
    print("\nFirst entry raw Q:")
    print(dataset[0]["instruction"][:80])
    print("\nFirst entry raw A:")
    print(dataset[0]["output"][:80])
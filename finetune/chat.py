"""
chat.py -- Talk to YOUR fine-tuned assistant.

Loads TinyLlama (the frozen base brain) + the LoRA adapter you trained
(the tiny "personality patch"), and chats with proper question answering.

Run (from the finetune/ folder):
    python chat.py
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "adapter"

SYSTEM = "You are a helpful, friendly assistant."


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(ADAPTER):
        raise SystemExit("No adapter found. Run:  python train.py")

    print("Loading TinyLlama + LoRA adapter ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    print("Assistant ready. Type your questions; 'quit' to exit.")
    while True:
        try:
            query = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
        # Template -> string -> explicit tensor (API-safe across transformers versions)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        mask = (ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=mask, max_length=ids.shape[1] + 300, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\nassistant > {answer}")


if __name__ == "__main__":
    main()

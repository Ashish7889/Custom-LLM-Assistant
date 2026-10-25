"""
train.py -- Turn a pre-trained LLM into an assistant with LoRA.

The BIG difference from your from-scratch GPT:
  * My-project/model.py started from RANDOM weights -- it knew nothing, so it
    had to read millions of characters just to learn spelling.
  * TinyLlama-1.1B-Chat already read ~1 trillion tokens in its early life.
    It already "knows" language, facts, grammar, style. Fine-tuning simply
    steers that knowledge toward Q&A behavior, using a tiny slice of the steps.

What is LoRA?
  Training all 1.1B parameters (full fine-tuning) would need ~30+ GB of
  memory. LoRA freezes ALL original weights and injects two small matrices
  (a rank-16 "shortcut") into each attention layer; only ~13 million
  parameters are updated. That's why it fits in your 6 GB GPU -- and why the
  saved result (called an "adapter") is only ~60 MB instead of 2+ GB.

Data: `yahma/alpaca-cleaned` (~52k instruction->response pairs). We use the
  first 2000 and hide the prompt tokens from the loss, so the model learns
  "what the assistant should SAY", not "how users write".

Run:
    python train.py           (from the finetune/ folder)
"""
import math
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_EXAMPLES = 2000
MAX_LEN = 512
OUTPUT_DIR = "adapter"

BATCH_SIZE = 2        # sequences per step (padded together)
GRAD_ACCUM = 8        # accumulate 8 steps -> effective batch of 16
EPOCHS = 2
LR = 2e-4
WARMUP = 50
LOG_EVERY = 10


def build_texts(n=N_EXAMPLES):
    """Alpaca rows -> (prompt token ids, full token ids)."""
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = []
    for row in list(ds)[:n]:
        user = row["instruction"] + (("\n\n" + row["input"]) if row["input"] else "")
        if not row["output"].strip():
            continue
        messages = [{"role": "user", "content": user},
                    {"role": "assistant", "content": row["output"]}]
        # Build the exact chat-template STRINGS, then encode. The template
        # already includes <bos>/<eos> special tokens, so add_special_tokens=False.
        prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        ids = full_ids[:MAX_LEN]
        labels = list(ids)
        for k in range(min(len(prompt_ids), len(ids))):
            labels[k] = -100          # hide the prompt: no loss on it
        rows.append((ids, labels))
    return rows


def collate(batch):
    """Pad examples to the same length so they can form a tensor."""
    maxlen = max(len(ids) for ids, _ in batch)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (ids, lbl) in enumerate(batch):
        t = torch.tensor(ids)
        input_ids[i, : len(ids)] = t
        labels[i, : len(lbl)] = torch.tensor(lbl)
        attn[i, : len(ids)] = 1
    return input_ids, labels, attn


def main():
    global tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = False          # required while training
    model.gradient_checkpointing_enable()   # trade a bit of speed for VRAM

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],   # classic LoRA choice
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(f"Tokenizing {N_EXAMPLES} instruction examples ...")
    rows = build_texts()
    print(f"  {len(rows)} examples ready")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps_per_epoch = math.ceil(len(rows) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    print(f"Plan: {steps_per_epoch} optimizer steps/epoch x {EPOCHS} epochs = {total_steps} total\n")

    model.train()
    step = micro = 0
    micro_losses = []
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(rows)).tolist()
        micros = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
        for m in micros:
            input_ids, labels, attn = collate(m)
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            micro_losses.append(out.loss.item())
            micro += 1
            if micro % GRAD_ACCUM == 0:
                step += 1
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                # warmup then gentle decay
                lr = LR * min((step + 1) / WARMUP, 1.0) \
                     if step < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % LOG_EVERY == 0 or step == total_steps:
                    avg = sum(micro_losses[-LOG_EVERY * GRAD_ACCUM:]) / (LOG_EVERY * GRAD_ACCUM)
                    print(f"step {step:4d}/{total_steps} | epoch {epoch+1} | loss {avg:.4f}", flush=True)

    model.save_pretrained(OUTPUT_DIR)
    n_files = sum(len(fs) for _, _, fs in os.walk(OUTPUT_DIR))
    print(f"\nSaved LoRA adapter to {OUTPUT_DIR}/ ({n_files} files)")
    print("Now run:  python chat.py")


if __name__ == "__main__":
    main()

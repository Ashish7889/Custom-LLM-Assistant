"""
finetune/train_personal_ca.py -- Ultra-Fast LoRA fine-tuning for TinyLlama and Qwen 2.5.
Trains models to serve as an Evidence-Grounded Personal CA for the common citizen.

Optimized for NVIDIA RTX 3060 Laptop GPU:
- Direct GPU VRAM allocation (model.to("cuda")) with bfloat16/float16.
- Compact LoRA target modules: ["q_proj", "v_proj"].
- Dynamic batch padding (no wasted compute on 512 static tokens).
- Completes 100 steps in ~2 minutes.
"""

import argparse
import json
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ca_dataset.jsonl"
ADAPTERS_DIR = BASE_DIR / "adapters"
os.makedirs(ADAPTERS_DIR, exist_ok=True)


def train_model(model_key: str = "tinyllama", max_steps: int = 100, batch_size: int = 2, grad_accum: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f"Personal CA Fine-Tuning: Model={model_key} on {device}")
    print(f"============================================================")

    if model_key == "qwen":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        out_dir = ADAPTERS_DIR / "qwen_ca"
    else:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        out_dir = ADAPTERS_DIR / "tinyllama_ca"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Tokenizing {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f if line.strip()]

    encoded_samples = []
    for item in raw_items:
        user_content = f"SOURCES:\n{item['context']}\n\nQUESTION: {item['instruction']}"
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(full_text, truncation=True, max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        encoded_samples.append((input_ids, mask, labels))

    print(f"Loaded {len(encoded_samples)} training samples.")
    split = int(0.9 * len(encoded_samples))
    train_data = encoded_samples[:split]
    val_data = encoded_samples[split:]

    print(f"Loading model: {model_name}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    # Peak-level LoRA: All attention and MLP projections adapted at rank 32
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    tr_params, tot_params = model.get_nb_trainable_parameters()
    print(f"Trainable Peak LoRA Parameters: {tr_params:,} / {tot_params:,} ({100*tr_params/tot_params:.2f}%)", flush=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=15, num_training_steps=max_steps)

    model.train()
    optimizer.zero_grad()
    step = 0
    running_loss = 0.0

    print("Beginning fast GPU training loop...", flush=True)
    while step < max_steps:
        random.shuffle(train_data)
        for i in range(0, len(train_data) - batch_size, batch_size):
            if step >= max_steps:
                break

            batch = train_data[i : i + batch_size]
            max_len = max(x[0].shape[0] for x in batch)
            
            b_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
            b_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
            b_lbl = torch.full((batch_size, max_len), -100, dtype=torch.long)

            for j, (ids, msk, lbl) in enumerate(batch):
                l = ids.shape[0]
                b_ids[j, :l] = ids
                b_mask[j, :l] = msk
                b_lbl[j, :l] = lbl

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_lbl = b_lbl.to(device)

            outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbl)
            loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == max_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % 10 == 0 or step == max_steps:
                divisor = 10 if step % 10 == 0 else (step % 10)
                avg_l = running_loss / divisor
                print(f"Step {step}/{max_steps} | Loss: {avg_l:.4f}", flush=True)
                running_loss = 0.0

    print(f"Saving fine-tuned adapter to {out_dir}...", flush=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Successfully trained and saved Personal CA adapter for {model_key}!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast LoRA fine-tuning for Personal CA")
    parser.add_argument("--model", type=str, choices=["tinyllama", "qwen"], default="tinyllama")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    train_model(model_key=args.model, max_steps=args.max_steps)

"""
finetune/train_firm_agents.py -- Unified LoRA Fine-Tuning Script for FinSage CA Agents:
- Agent 1: Qwen 2.5 1.5B (Junior CA Analyst) -> adapters/qwen_ca
- Agent 2: DeepSeek-R1 Distill 1.5B (Senior Partner CA) -> adapters/deepseek_ca

Optimized for NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM):
- PEFT LoRA (rank=16, alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
- Gradient checkpointing enabled
- bfloat16 / float16 mixed precision
- Dynamic token padding with loss masking
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Prevent Windows Python 3.13 pyarrow crash
sys.modules["pyarrow"] = None
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

DATA_DIR = BASE_DIR / "data"
ADAPTERS_DIR = BASE_DIR / "adapters"
os.makedirs(ADAPTERS_DIR, exist_ok=True)


def train_agent(
    agent_type: str = "analyst",
    max_steps: int = 50,
    batch_size: int = 1,
    grad_accum: int = 4,
    lr: float = 2e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65, flush=True)
    print(f"FinSage Multi-Agent Fine-Tuning: Agent={agent_type.upper()} on {device}", flush=True)
    print("=" * 65, flush=True)

    if agent_type == "senior":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        data_path = DATA_DIR / "deepseek_senior_partner_train.jsonl"
        out_dir = ADAPTERS_DIR / "deepseek_ca"
    else:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        data_path = DATA_DIR / "qwen_junior_analyst_train.jsonl"
        out_dir = ADAPTERS_DIR / "qwen_ca"

    print(f"Loading tokenizer: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and encode data
    print(f"Loading training data from {data_path}...", flush=True)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f if line.strip()]

    encoded_samples = []
    for item in raw_items:
        if agent_type == "senior":
            # DeepSeek-R1 distill uses user-turn prompt without system prompt
            messages = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ]
        else:
            # Qwen uses system + user + assistant
            messages = [
                {"role": "system", "content": item.get("system", "")},
                {"role": "user", "content": item.get("instruction", "Prepare your memo now.")},
                {"role": "assistant", "content": item["output"]},
            ]

        try:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            full_text = f"{item.get('instruction', '')}\n\n{item.get('output', '')}"

        enc = tokenizer(full_text, truncation=True, max_length=1024, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        encoded_samples.append((input_ids, mask, labels))

    print(f"Total training samples: {len(encoded_samples)}", flush=True)

    # Clear GPU VRAM before model load
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading base model: {model_name}...", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    tr_params, tot_params = model.get_nb_trainable_parameters()
    print(f"Trainable LoRA Parameters: {tr_params:,} / {tot_params:,} ({100*tr_params/tot_params:.2f}%)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=max_steps)

    model.train()
    step = 0
    running_loss = 0.0

    print("\nStarting Training Loop...", flush=True)
    while step < max_steps:
        for idx in range(0, len(encoded_samples), batch_size):
            batch = encoded_samples[idx : idx + batch_size]
            max_len = max(x[0].size(0) for x in batch)

            # Collate batch with padding
            b_ids, b_mask, b_labels = [], [], []
            for input_ids, mask, labels in batch:
                pad_len = max_len - input_ids.size(0)
                if pad_len > 0:
                    input_ids = torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                    labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                b_ids.append(input_ids)
                b_mask.append(mask)
                b_labels.append(labels)

            input_ids_t = torch.stack(b_ids).to(device)
            attention_mask_t = torch.stack(b_mask).to(device)
            labels_t = torch.stack(b_labels).to(device)

            outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t, labels=labels_t)
            loss = outputs.loss / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum

            if (idx // batch_size + 1) % grad_accum == 0 or (idx + batch_size >= len(encoded_samples)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 5 == 0 or step == max_steps:
                    avg_loss = running_loss / 5.0
                    print(f"Step [{step}/{max_steps}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
                    running_loss = 0.0

                if step >= max_steps:
                    break

    print(f"\nTraining completed! Saving LoRA adapter to {out_dir}...", flush=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[OK] Adapter successfully saved to {out_dir}", flush=True)

    # Clean VRAM
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSage CA Multi-Agent LoRA Fine-Tuner")
    parser.add_argument("--agent", choices=["analyst", "senior", "all"], default="all", help="Which agent to train")
    parser.add_argument("--steps", type=int, default=30, help="Number of training steps")
    args = parser.parse_args()

    if args.agent in ["analyst", "all"]:
        train_agent(agent_type="analyst", max_steps=args.steps)

    if args.agent in ["senior", "all"]:
        train_agent(agent_type="senior", max_steps=args.steps)

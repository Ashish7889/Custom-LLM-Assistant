"""
finetune_finance/train.py

Minimal LoRA fine-tuning of TinyLlama on financial instructions.
Dataset: data/finance_template.jsonl (chat-formatted Q&A).
"""
import json, os, random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = Path("data/finance_template.jsonl")
OUTPUT_DIR = Path("./finance_adapter")
MAX_STEPS = 500
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_STEPS = 50
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Global tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_dataset():
    with open(DATA_PATH, "r") as f:
        raw_lines = [json.loads(l) for l in f.readlines()]

    def tokenize_one(ex):
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["instruction"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": ex["output"]}],
            tokenize=False,
        )
        enc = tokenizer(full_text, truncation=True, max_length=MAX_LEN, padding="max_length")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        assist_idx = text.find("<|assistant|>")

        labels = list(input_ids)
        if assist_idx >= 0:
            prefix_ids = tokenizer.encode(text[:assist_idx], add_special_tokens=False)
            mask_len = len(prefix_ids)
        else:
            mask_len = max(1, int(MAX_LEN * 0.35))

        for i in range(min(mask_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    tokenized = [tokenize_one(ex) for ex in raw_lines]
    split_idx = int(0.9 * len(tokenized))
    train_tensors = [
        torch.stack([d[k] for d in tokenized[:split_idx]]) for k in ["input_ids", "attention_mask", "labels"]
    ]
    val_tensors = [
        torch.stack([d[k] for d in tokenized[split_idx:]]) for k in ["input_ids", "attention_mask", "labels"]
    ]
    return train_tensors, val_tensors


def train():
    train_tensors, val_tensors = tokenize_dataset()
    n_train = train_tensors[0].shape[0]
    n_val = val_tensors[0].shape[0]
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    tr, tot = model.get_nb_trainable_parameters()
    print(f"LoRA params: {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = MAX_STEPS
    warmup_steps = WARMUP_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model.train()
    optimizer.zero_grad()
    step = 0
    best_val = float("inf")
    no_improve = 0

    for epoch in range(3):
        indices = list(range(n_train))
        random.shuffle(indices)

        for idx_start in range(0, n_train, BATCH_SIZE * GRAD_ACCUM):
            if step >= MAX_STEPS:
                break

            # Collect batch items
            batch_item_indices = indices[idx_start: idx_start + BATCH_SIZE]
            # Gather raw tensor slices
            raw_inputs = [train_tensors[0][i] for i in batch_item_indices]
            raw_masks = [train_tensors[1][i] for i in batch_item_indices]
            raw_labels = [train_tensors[2][i] for i in batch_item_indices]

            # Pad to same length within batch
            max_len = max(inp.shape[0] for inp in raw_inputs)
            padded_input = torch.full((BATCH_SIZE, max_len), tokenizer.pad_token_id, dtype=torch.long)
            padded_mask = torch.zeros((BATCH_SIZE, max_len), dtype=torch.long)
            padded_label = torch.full((BATCH_SIZE, max_len), -100, dtype=torch.long)
            for i, idx in enumerate(batch_item_indices):
                l = raw_inputs[i].shape[0]
                padded_input[i, :l] = raw_inputs[i]
                padded_mask[i, :l] = raw_masks[i]
                padded_label[i, :l] = raw_labels[i]

            # DataCollator expects a list of dicts with python lists or tensors
            features = []
            for i in range(BATCH_SIZE):
                features.append({
                    "input_ids": padded_input[i].tolist(),
                    "attention_mask": padded_mask[i].tolist(),
                    "labels": padded_label[i].tolist(),
                })

            collated = data_collator(features)
            collated = {k: v.to(DEVICE) for k, v in collated.items()}

            outputs = model(**collated)
            loss = outputs.loss
            loss_backward = loss / GRAD_ACCUM
            loss_backward.backward()

            step += 1
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Validation every 50 steps
                if step % 50 == 0 or step == MAX_STEPS:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vi in range(min(20, n_val)):
                            # Get one val example as a list dict
                            ve = {"input_ids": val_tensors[0][vi].tolist(),
                                  "attention_mask": val_tensors[1][vi].tolist(),
                                  "labels": val_tensors[2][vi].tolist()}
                            c = data_collator([ve])
                            c = {k: v.to(DEVICE) for k, v in c.items()}
                            out = model(**c)
                            val_losses.append(out.loss.item())
                    model.train()

                    avg_val = sum(val_losses) / len(val_losses) if val_losses else float("inf")
                    print(f"Step {step:5d}/{MAX_STEPS} | train {loss.item()*GRAD_ACCUM:.4f} | val {avg_val:.4f}")

                    if avg_val < best_val:
                        best_val = avg_val
                        no_improve = 0
                        model.save_pretrained(OUTPUT_DIR / "best")
                        print(f"  -> new best saved")
                    else:
                        no_improve += 1
                        if no_improve >= 3:
                            print("Early stopping.")
                            break

                if step >= MAX_STEPS:
                    break

        if step >= MAX_STEPS:
            break

    model.save_pretrained(OUTPUT_DIR / "final")
    print(f"\nTraining done. Model in {OUTPUT_DIR}/ (best val: {best_val:.4f})")


if __name__ == "__main__":
    train()
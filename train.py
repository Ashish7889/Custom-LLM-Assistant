"""
train.py -- Teach the model to predict the next character.

How training actually works (the loop below, in plain English):

  Repeat thousands of times:
  1. Grab a random window of 256 characters from the corpus (the "context").
     The "targets" are the SAME data shifted one step to the left -- because
     if you see characters [1..256], the model should predict [2..257]:
     the character after each character it has read so far.
  2. Feed the context through the transformer -> get predictions.
  3. Cross-entropy measures how wrong the predictions were ("the loss").
  4. Backpropagation computes, for every one of the ~10.7 million parameters,
     "if I were nudged THIS direction, the error would shrink".
  5. AdamW takes a tiny step in that direction for every parameter.

  After a few thousand steps: the model can write Shakespeare.
  Learning rate scheduling: start gentle, learn fast at peak, then fade out
  -- like wiggling a key less and less as the lock starts to turn.

Run it:
    python train.py

Then watch the two loss numbers drop. When val_loss stops falling for many
evals in a row, you're done -- stop training and chat (python chat.py).
"""
import math
import os

import numpy as np
import torch

from model import Config, GPT
from tokenizer import CharTokenizer

# ----------------------------------------------------------------------------
# Training hyperparameters (the "recipe"). Try tweaking these later!
# ----------------------------------------------------------------------------
BATCH_SIZE = 48       # how many independent character-windows per step
BLOCK_SIZE = 256      # window length the model sees (must match model.py Config)
MAX_ITERS = 5000      # total training steps; 5000 -> decent Shakespeare style
EVAL_INTERVAL = 250   # every N steps, pause and measure on train/val data
EVAL_ITERS = 40       # batches used for a fair loss measurement
LEARNING_RATE = 6e-4  # size of each parameter step
WEIGHT_DECAY = 0.1    # keeps parameters small -> less overfitting
DROPOUT = 0.2
WARMUP_STEPS = 100    # gentle start so the first steps don't destroy the weights
GRAD_CLIP = 1.0       # safety rail: cap how far one step can move the weights
CHECKPOINT_PATH = "checkpoints/tinyclaude.pt"
BEST_PATH = "checkpoints/tinyclaude-best.pt"


def get_batch(data, device):
    """Pick BATCH_SIZE random windows from the corpus and their shifted targets."""
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i : i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, train_data, val_data, device):
    """Average the loss over EVAL_ITERS windows, separately for train and val."""
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # Load the flat binaries; memory-map so Python never copies the full file.
    train_data = np.memmap("data/train.bin", dtype=np.uint8, mode="r")
    val_data = np.memmap("data/val.bin", dtype=np.uint8, mode="r")

    tokenizer = CharTokenizer.load("checkpoints/tokenizer.pt")
    config = Config(vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE, dropout=DROPOUT)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Plan: {MAX_ITERS} steps, batch={BATCH_SIZE}x{BLOCK_SIZE} characters\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    # Cosine learning-rate schedule with warmup (see docstring).
    def lr_for_step(it):
        if it < WARMUP_STEPS:
            return LEARNING_RATE * (it + 1) / WARMUP_STEPS
        if it > MAX_ITERS:
            return LEARNING_RATE / 10
        decay_ratio = (it - WARMUP_STEPS) / (MAX_ITERS - WARMUP_STEPS)
        return LEARNING_RATE * 0.1 + 0.9 * LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * decay_ratio))

    model.train()
    best_val = float("inf")
    for it in range(1, MAX_ITERS + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_for_step(it)

        X, Y = get_batch(train_data, device)
        with torch.autocast(device, dtype=torch.bfloat16):
            _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # safety rail
        optimizer.step()

        if it % 25 == 0 or it == 1:
            print(f"step {it:5d}/{MAX_ITERS} | training loss {loss.item():.4f}")
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = evaluate(model, train_data, val_data, device)
            print(f"-- eval @ step {it}: train {losses['train']:.4f} | val {losses['val']:.4f} --")
            # Early stopping: keep whichever checkpoint did best on UNSEEN data.
            # Once val loss turns up, further training is memorization.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model_state": model.state_dict(), "config": config}, BEST_PATH)
                print(f"   (new best val {best_val:.4f} -> saved {BEST_PATH})")
            # Sanity numbers for Shakespeare: random init ~4.4, done ~1.5.

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, CHECKPOINT_PATH)
    # prefer the best-val checkpoint for chat.py
    if best_val < float("inf"):
        import shutil
        shutil.copyfile(BEST_PATH, CHECKPOINT_PATH)
        print(f"\nBest val {best_val:.4f} kept; final overfit copy also saved to {CHECKPOINT_PATH}")
    else:
        torch.save({"model_state": model.state_dict(), "config": config}, CHECKPOINT_PATH)
    print(f"Saved model to {CHECKPOINT_PATH}. Now run: python chat.py")


if __name__ == "__main__":
    main()

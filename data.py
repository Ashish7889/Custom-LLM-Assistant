"""
data.py -- Prepare the training data.

Run this BEFORE training. It:
  1. Downloads a training corpus (default: Shakespeare -- ~1 MB, perfect for
     a first model. Every famous "tiny GPT" tutorial uses it because results
     converge fast and are fun to read: the model learns to *write like
     Shakespeare*).
     Later, swap in your own text file and re-run to teach the model your
     style instead.
  2. Tokenizes all of it into one long list of character IDs.
  3. Splits it 95/5 into training data and "validation" data (data the model
     never trains on -- comparing loss on both tells us if the model is
     LEARNING or just MEMORIZING).
  4. Saves everything as flat binary files (train.bin, val.bin) and saves the
     tokenizer, so training can read megabytes of data in milliseconds.
"""
import os
import urllib.request

import numpy as np

from tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(corpus_path="data/input.txt"):
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Download the corpus if we don't have it yet.
    if not os.path.exists(corpus_path):
        print(f"Downloading training corpus to {corpus_path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, corpus_path)
    text = open(corpus_path, encoding="utf-8").read()
    print(f"Corpus: {len(text):,} characters")

    # 2. Build the vocabulary and encode ALL the text to integer IDs.
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text), dtype=np.uint8)  # uint8: 0-255
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
    print(f"Tokenized: {len(data):,} tokens")

    # 3. Split 95% train / 5% validation.
    split = int(len(data) * 0.95)
    train_data, val_data = data[:split], data[split:]
    print(f"Train: {len(train_data):,} tokens | Validation: {len(val_data):,} tokens")

    # 4. Save as flat uint8 binaries -- one number per token, nothing else.
    train_data.tofile("data/train.bin")
    val_data.tofile("data/val.bin")
    tokenizer.save("checkpoints/tokenizer.pt")
    print("Saved data/train.bin, data/val.bin, checkpoints/tokenizer.pt")


if __name__ == "__main__":
    prepare()

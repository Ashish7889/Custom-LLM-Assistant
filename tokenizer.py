"""
tokenizer.py -- The translator between text (what you read) and numbers (what
the model eats).

Large models like Claude use "BPE" tokenizers: roughly ~4 characters per
token. We use a CHARACTER-level tokenizer: one character = one token.
It's the simplest tokenizer that could possibly work, and it's exactly what
you want when learning: the entire vocabulary is just "every letter and
punctuation mark in the training text" (usually 60-100 tokens total, versus
~100,000 for Claude).
"""


class CharTokenizer:
    def __init__(self, text):
        """Build the vocabulary from a training text: every unique character
        gets its own integer ID."""
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # character -> id
        self.itos = {i: ch for i, ch in enumerate(chars)}  # id -> character
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        import torch
        torch.save({"stoi": self.stoi, "vocab_size": self.vocab_size}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, weights_only=False)
        tok = cls.__new__(cls)
        tok.stoi = d["stoi"]
        tok.itos = {i: ch for ch, i in d["stoi"].items()}
        tok.vocab_size = d["vocab_size"]
        return tok

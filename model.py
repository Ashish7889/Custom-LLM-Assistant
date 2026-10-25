"""
model.py -- The transformer: the "brain" of your tiny Claude.

This file implements a GPT (Generative Pre-trained Transformer) from scratch,
the same architecture family Claude and ChatGPT use, just tiny.

The one big idea:
    The model's ONLY job during training is: "given the previous characters,
    predict the next character." Do that well enough, over and over, and text
    that looks intelligent comes out the other side.

How it works, step by step (for input of 256 characters):
  1. Each character is looked up in a table -> a vector of 384 numbers
     (the "embedding"). Similar characters get similar vectors.
  2. A table of "position" vectors is added, so the model knows the order
     of the characters (transformers see all tokens at once!).
  3. The sequence goes through 6 identical "blocks". Each block does:
        a. ATTENTION: every position looks back at ALL earlier positions and
           mixes in the information it finds useful. E.g. seeing "Bob" early
           in a paragraph, attention is what lets the model know that "he"
           later refers to Bob.
           ("Causal" = positions are only allowed to look BACKWARDS:
           at prediction time the future doesn't exist yet.)
        b. MLP: a small neural network applied to every position separately,
           giving the model "thinking time" after gathering context.
  4. A final layer turns each position's 384 numbers into one probability
     per character in your vocabulary ("next character is ';': 40%, ...).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """All the "size knobs" of the model. Small = trains fast on your GPU."""
    vocab_size: int = None      # filled in by data.py (number of unique characters)
    block_size: int = 256       # how many characters the model looks at at once
    n_layers: int = 6           # how many attention/MLP blocks stacked up
    n_heads: int = 6            # attention "readers" per block
    n_embd: int = 384           # width of the model (numbers per character)
    dropout: float = 0.2        # regularization: randomly drop signals while learning

    # Rough model size with these defaults: ~10.7 million parameters.
    # For comparison: Claude is ~hundreds of billions. Scale is everything.


class CausalSelfAttention(nn.Module):
    """Step 3a: every character looks back at earlier characters and gathers info."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        # One big projection is faster than n_heads small ones; we split the
        # output into `n_heads` chunks afterwards.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # "polish" the result
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # The causal mask: a lower-triangular True/False matrix. A position is
        # only allowed to attend to positions up to itself -- never the future.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding width

        # Project every position into its query, key and value vectors.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape into multiple heads: (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Attention = softmax(query . key) -- "what is relevant to me?"
        # PyTorch's built-in fused version is much faster than a manual loop.
        # is_causal=True applies the same lower-triangular rule as self.mask:
        # position 0 sees only position 0, position 5 sees positions 0-5, etc.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back
        return self.c_proj(y)  # project back to embedding size


class MLP(nn.Module):
    """Step 3b: per-position "thinking" network between attention calls."""

    def __init__(self, config):
        super().__init__()
        n_inner = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, n_inner)
        self.c_proj = nn.Linear(n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)          # GELU: a smooth "activation" (Claude/ChatGPT use this)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """One transformer block = Attention + MLP, with residual connections."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The "+ x" (residual/skip connection) is crucial: information flows
        # straight through the network, and each block adds a small correction
        # on top. This is what lets us stack many blocks without breaking.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full model. Put in characters, get next-character probabilities out."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Two lookup tables: "what does this character mean" and
        # "what does this position mean".
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Standard initialization for stabler training.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Steps 1 & 2: character embeddings + position embeddings, added.
        positions = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)

        # Step 3: the stacked transformer blocks.
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Step 4: predict a probability for every possible next character.
        logits = self.lm_head(x)

        # If we're training, measure how wrong the predictions are.
        # Cross-entropy = negative log-likelihood of the correct answer.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()  # generation never changes the weights
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Generate text one character at a time, feeding each prediction back in.

        temperature: low = safe & repetitive, high = wild & creative
        top_k: only sample from the k most likely next characters
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last prediction matters
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_char_idx), dim=1)
        return idx

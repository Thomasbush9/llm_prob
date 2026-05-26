"""Small causal Transformer decoder built on `nn.TransformerEncoderLayer` + a causal mask."""

import torch
from torch import nn


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d=128, n_heads=4, n_layers=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=4 * d,
            batch_first=True, activation='gelu', dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.blocks(h, mask=causal)
        return self.head(h)

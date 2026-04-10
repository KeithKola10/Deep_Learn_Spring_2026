#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer encoder classifier built from scratch in PyTorch.

Components:
    TokenEmbedding             — learned token embedding table
    SinusoidalPositionalEncoding — fixed positional encodings added to embeddings
    MultiHeadSelfAttention     — scaled dot-product attention with padding mask
    FeedForward                — two-layer MLP with GELU activation
    TransformerEncoderLayer    — pre-LayerNorm encoder block (MHSA + FFN + residuals)
    TransformerClassifier      — stacks encoder layers, pools, and classifies

Pre-LayerNorm: normalisation is applied before each sub-layer rather than after,
which improves training stability for small models trained from scratch.

Padding mask: attention_mask uses True = real token; this is inverted before
passing to nn.MultiheadAttention which expects True = ignore.
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """Maps token IDs to d_model-dimensional vectors, scaled by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale     = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed sinusoidal position signals to token embeddings, then applies dropout."""

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention.
    Padding positions are masked out so they do not influence other tokens.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~attention_mask   # invert: nn.MultiheadAttention needs True = ignore
        out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
        return out


class FeedForward(nn.Module):
    """Two-layer position-wise MLP: Linear → GELU → Dropout → Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer with pre-LayerNorm and residual connections.
        x = x + MHSA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff    = FeedForward(d_model, d_ff, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ──────────────────────────────────────────────
# Full classifier
# ──────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Transformer encoder for binary classification.
    Stacks N encoder layers, pools the output, and projects to 2 logits.
    Pooling mode is set via model_params['pooling']: 'mean' or 'cls'.
    """

    def __init__(self, model_params: dict):
        super().__init__()
        d_model        = model_params['d_model']
        n_heads        = model_params['n_heads']
        n_layers       = model_params['n_layers']
        d_ff           = model_params['d_ff']
        dropout        = model_params['dropout']
        vocab_size     = model_params['vocab_size']
        max_seq_len    = model_params['max_seq_len']
        head_hidden    = model_params['head_hidden_dim']
        self.pooling   = model_params['pooling']

        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pe    = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal distribution for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Averages hidden states over real token positions, ignoring padding."""
        mask_f = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_f).sum(dim=1)
        count  = mask_f.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns class logits of shape [batch, 2]."""
        x = self.embed(input_ids)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        if self.pooling == 'mean':
            pooled = self._mean_pool(x, attention_mask)
        else:
            pooled = x[:, 0, :]

        return self.head(pooled)

    def forward_explain(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass that captures embedding gradients for saliency explanation.
        Call .backward() on a logit after this, then read grads['emb'] for
        per-token importance scores. Uses a hook for MPS compatibility.
        """
        grads = {}

        x = self.embed(input_ids)
        x.register_hook(lambda g: grads.update({'emb': g}))

        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)

        if self.pooling == 'mean':
            pooled = self._mean_pool(x, attention_mask)
        else:
            pooled = x[:, 0, :]

        return self.head(pooled), grads

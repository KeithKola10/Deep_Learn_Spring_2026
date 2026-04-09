#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer encoder classifier built from scratch in PyTorch.

Architecture (bottom-up):
    TokenEmbedding             — learned embedding table, scaled by sqrt(d_model)
    SinusoidalPositionalEncoding — fixed sinusoidal PE, registered as buffer
    MultiHeadSelfAttention     — standard scaled dot-product MHSA with padding mask
    FeedForward                — 2-layer MLP with GELU activation
    TransformerEncoderLayer    — Pre-LN: LayerNorm BEFORE each sub-layer (more stable for small models)
    TransformerClassifier      — stacks N encoder layers, pools, then classifies

Pooling: 'mean' (masked mean over real tokens) or 'cls' (first token).
Classification head: Linear(d_model, head_hidden_dim) -> GELU -> Dropout -> Linear(head_hidden_dim, 2)

Pre-LayerNorm note: unlike the original "Attention Is All You Need" (post-LN),
pre-LN applies LayerNorm BEFORE the sub-layer, then adds the residual after.
This is more training-stable without careful LR warm-up tuning.

Padding mask note: nn.MultiheadAttention expects key_padding_mask where True = IGNORE.
Our attention_mask uses True = REAL TOKEN, so we invert before passing: ~attention_mask.
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Learned embedding table.
    Scales output by sqrt(d_model) as in "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale     = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] -> [B, T, d_model]
        return self.embedding(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal positional encodings.
    Registered as a buffer so it moves with the model to the correct device.
    Dropout is applied after adding PE to the embeddings.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build [max_seq_len, d_model] PE matrix
        pe  = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)           # [T, 1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))                         # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # [1, T, d_model] — broadcast over batch
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.
    Uses nn.MultiheadAttention internally (batch_first=True).

    key_padding_mask convention: True = IGNORE that position.
    We receive attention_mask (True = REAL TOKEN) and invert it.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x:               torch.Tensor,   # [B, T, d_model]
        attention_mask:  torch.Tensor,   # [B, T], True = real token
    ) -> torch.Tensor:
        # Invert mask: nn.MultiheadAttention needs True = IGNORE
        padding_mask = ~attention_mask   # [B, T]
        out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.
    Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)
    """

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
    Single Pre-LayerNorm Transformer encoder layer.

    Pre-LN order (more stable than original post-LN for small from-scratch models):
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

    def forward(
        self,
        x:              torch.Tensor,   # [B, T, d_model]
        attention_mask: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        # MHSA sub-layer (pre-LN)
        x = x + self.drop(self.attn(self.norm1(x), attention_mask))
        # FFN sub-layer (pre-LN)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ──────────────────────────────────────────────
# Full classifier
# ──────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Full Transformer encoder binary classifier.

    Forward pass:
        input_ids, attention_mask  ->  logits [B, 2]

    Pooling (set by model_params['pooling']):
        'mean' — average over real (non-padding) token positions
        'cls'  — take the representation of position 0 (first token)
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

        # Final LayerNorm after all encoder layers (pre-LN models benefit from this)
        self.norm = nn.LayerNorm(d_model)

        # Classification head: d_model -> head_hidden -> 2
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers; normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _mean_pool(
        self,
        hidden:         torch.Tensor,   # [B, T, d_model]
        attention_mask: torch.Tensor,   # [B, T], True = real token
    ) -> torch.Tensor:
        # [B, T, d_model] masked sum, then divide by real token count
        mask_f = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
        summed = (hidden * mask_f).sum(dim=1)           # [B, d_model]
        count  = mask_f.sum(dim=1).clamp(min=1e-9)     # [B, 1]
        return summed / count                           # [B, d_model]

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, T]
        attention_mask: torch.Tensor,   # [B, T], True = real token
    ) -> torch.Tensor:
        x = self.embed(input_ids)       # [B, T, d_model]
        x = self.pe(x)                  # [B, T, d_model]

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)                # [B, T, d_model]

        if self.pooling == 'mean':
            pooled = self._mean_pool(x, attention_mask)    # [B, d_model]
        else:  # 'cls'
            pooled = x[:, 0, :]                            # [B, d_model]

        return self.head(pooled)        # [B, 2]

    def forward_explain(
        self,
        input_ids:      torch.Tensor,   # [1, T]
        attention_mask: torch.Tensor,   # [1, T]
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for gradient saliency explanation.

        Registers a gradient hook on the embedding output so that after
        calling .backward() on a logit, the embedding gradients are
        captured in the returned `grads` dict under key 'emb'.

        Usage:
            logits, grads = model.forward_explain(input_ids, mask)
            pred = logits.argmax(-1).item()
            logits[0, pred].backward()
            importance = grads['emb'].norm(dim=-1).squeeze(0)  # [T]

        Uses a hook instead of retain_grad() for MPS compatibility.
        """
        grads = {}

        x = self.embed(input_ids)               # [1, T, d_model]
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

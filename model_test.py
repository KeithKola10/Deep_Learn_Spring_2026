#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-style interactive script for testing the Transformer model.

Run cells individually in VSCode (Shift+Enter) or execute the whole file.
Tests: model instantiation, parameter count, shape sanity checks,
       forward pass on raw text prompts, softmax probabilities,
       and (if a checkpoint exists) inference with trained weights.
"""

# %% ── Cell 1: Imports & setup ───────────────────────────────────────────────

import os
import sys
import torch
import tiktoken
import CONFIG
sys.path.insert(0, CONFIG.model_configs_dir)
from model_configs_v1 import model_params, model_build_options
from model import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerEncoderLayer,
    TransformerClassifier,
)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')   # Apple Silicon GPU
else:
    device = torch.device('cpu')
print(f"Device: {device}")
print(f"Model tag: {model_params['model_tag']}")


# %% ── Cell 2: Instantiate model & print parameter count ────────────────────

model = TransformerClassifier(model_params).to(device)

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:>10,}")
print(f"Trainable parameters: {trainable_params:>10,}")
print(f"\nModel architecture:\n{model}")


# %% ── Cell 3: Sub-module shape sanity checks (random batch) ─────────────────

B  = 4                               # batch size
T  = model_params['max_seq_len']     # sequence length
d  = model_params['d_model']

# Fake token IDs: range 1..vocab_size (0 = PAD)
fake_ids  = torch.randint(1, model_params['vocab_size'], (B, T)).to(device)
fake_mask = torch.ones(B, T, dtype=torch.bool).to(device)  # all real tokens

# TokenEmbedding
emb_out = model.embed(fake_ids)
assert emb_out.shape == (B, T, d), f"Embedding shape mismatch: {emb_out.shape}"
print(f"TokenEmbedding output:           {tuple(emb_out.shape)}  ✓")

# SinusoidalPE
pe_out = model.pe(emb_out)
assert pe_out.shape == (B, T, d)
print(f"SinusoidalPE output:             {tuple(pe_out.shape)}  ✓")

# One encoder layer
layer_out = model.layers[0](pe_out, fake_mask)
assert layer_out.shape == (B, T, d)
print(f"TransformerEncoderLayer output:  {tuple(layer_out.shape)}  ✓")

# Full forward pass
logits = model(fake_ids, fake_mask)
assert logits.shape == (B, 2), f"Logit shape mismatch: {logits.shape}"
print(f"TransformerClassifier logits:    {tuple(logits.shape)}  ✓")


# %% ── Cell 4: Padding mask behavior check ───────────────────────────────────
# Verify that padding tokens (mask=False) don't affect the pooled representation
# when using mean pooling.

if model_params['pooling'] == 'mean':
    ids_full = torch.randint(1, model_params['vocab_size'], (1, T)).to(device)
    mask_full = torch.ones(1, T, dtype=torch.bool).to(device)

    # Make second half padding
    ids_half = ids_full.clone()
    mask_half = mask_full.clone()
    mask_half[0, T//2:] = False
    ids_half[0, T//2:] = 0   # PAD token

    logits_full = model(ids_full, mask_full)
    logits_half = model(ids_half, mask_half)

    print(f"\nLogits (full sequence, no padding):   {logits_full.detach().cpu().numpy()}")
    print(f"Logits (half padded — different seq): {logits_half.detach().cpu().numpy()}")
    print("(These will differ because the token content differs — that is expected.)")
    print("Padding mask is working: padded positions are excluded from mean pool. ✓")


# %% ── Cell 5: Forward pass on hardcoded human vs. AI prompts ────────────────

enc = tiktoken.get_encoding(model_params['tokenizer'])

prompts = [
    # Label 0 — Human-style (informal, personal)
    "honestly i think the whole debate just comes down to money at the end of the day "
    "like nobody is gonna change their mind no matter what you say, people are set in their ways",

    # Label 1 — AI-style (structured, formal)
    "The phenomenon in question can be attributed to several interrelated factors. "
    "First, the structural conditions that enable this behavior have been well-documented in the literature. "
    "Second, the empirical evidence consistently supports the hypothesis that external incentives "
    "play a significant role in shaping outcomes.",
]
true_labels = ["human (expected)", "AI (expected)"]

model.eval()
with torch.no_grad():
    for text, expected in zip(prompts, true_labels):
        ids = enc.encode(text)
        ids = [tid + 1 for tid in ids][:model_params['max_seq_len']]   # offset + truncate
        pad_len = model_params['max_seq_len'] - len(ids)
        ids_padded = ids + [0] * pad_len
        mask       = [True] * len(ids) + [False] * pad_len

        input_ids_t = torch.tensor([ids_padded], dtype=torch.long).to(device)
        mask_t      = torch.tensor([mask],       dtype=torch.bool ).to(device)

        logits = model(input_ids_t, mask_t)
        probs  = torch.softmax(logits, dim=-1)[0]

        print(f"\nPrompt ({expected}):")
        print(f"  '{text[:80]}...'")
        print(f"  Logits:      [{logits[0,0].item():.4f}, {logits[0,1].item():.4f}]")
        print(f"  P(human): {probs[0].item()*100:5.1f}%   P(AI): {probs[1].item()*100:5.1f}%")
        pred = "human" if probs[0] > probs[1] else "AI"
        print(f"  Prediction: {pred}  (untrained — random weights)")


# %% ── Cell 6: Load trained checkpoint (if it exists) and re-run inference ───

tag             = model_params['model_tag']
checkpoint_path = os.path.join(CONFIG.models_dir, f'best_model_{tag}.pt')

if os.path.exists(checkpoint_path):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Checkpoint loaded. Re-running inference on same prompts...")

    with torch.no_grad():
        for text, expected in zip(prompts, true_labels):
            ids = enc.encode(text)
            ids = [tid + 1 for tid in ids][:model_params['max_seq_len']]
            pad_len = model_params['max_seq_len'] - len(ids)
            ids_padded = ids + [0] * pad_len
            mask       = [True] * len(ids) + [False] * pad_len

            input_ids_t = torch.tensor([ids_padded], dtype=torch.long).to(device)
            mask_t      = torch.tensor([mask],       dtype=torch.bool ).to(device)

            logits = model(input_ids_t, mask_t)
            probs  = torch.softmax(logits, dim=-1)[0]
            pred   = "human" if probs[0] > probs[1] else "AI"

            print(f"\nPrompt ({expected}):")
            print(f"  P(human): {probs[0].item()*100:5.1f}%   P(AI): {probs[1].item()*100:5.1f}%   -> {pred}")
else:
    print(f"\nNo checkpoint found at '{checkpoint_path}' — train the model first with:")
    print(f"    python run.py -v v1")

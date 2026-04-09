#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-style interactive script for testing the data pipeline.

Run cells individually in VSCode (Shift+Enter) or execute the whole file.
Tests: HC3 loading, class balance, raw sample inspection,
       sequence length distribution, tokenization round-trips,
       and DataLoader batch shapes.
"""

# %% ── Cell 1: Imports & config ──────────────────────────────────────────────

import sys
import tiktoken
import torch
import CONFIG
sys.path.insert(0, CONFIG.model_configs_dir)
from model_configs_v1 import model_params, model_build_options
from data import (
    load_hc3,
    sequence_length_stats,
    flatten_prompts,
    split_prompts,
    TextClassificationDataset,
    make_dataloaders,
)

print("Config loaded:")
print(f"  tokenizer    = {model_params['tokenizer']}")
print(f"  vocab_size   = {model_params['vocab_size']}")
print(f"  max_seq_len  = {model_params['max_seq_len']}")
print(f"  dataset      = {model_build_options['dataset']}")


# %% ── Cell 2: Load raw HC3 & check class balance ────────────────────────────

prompt_texts, prompt_labels, questions = load_hc3(model_build_options['hc3_data_dir'])

n_prompts = len(prompt_texts)
all_texts, all_labels = flatten_prompts(prompt_texts, prompt_labels)
n_total   = len(all_texts)
n_human   = all_labels.count(0)
n_ai      = all_labels.count(1)

print(f"Total prompts:  {n_prompts:,}")
print(f"Total samples:  {n_total:,}  (human={n_human:,}, AI={n_ai:,})")
print(f"Class balance:  {n_human/n_total*100:.1f}% human  /  {n_ai/n_total*100:.1f}% AI")


# %% ── Cell 3: Inspect a few raw samples ─────────────────────────────────────

for i in [0, 100, 500]:
    q      = questions[i]
    h_ans  = prompt_texts[i][0]            # first human answer
    ai_ans = prompt_texts[i][-1]           # ChatGPT answer (always last after load_hc3)
    print(f"\n── Prompt {i} ──")
    print(f"Question (truncated): {q[:120]}")
    print(f"Human answer  [{len(h_ans):>5} chars]: {h_ans[:200]}")
    print(f"ChatGPT answer[{len(ai_ans):>5} chars]: {ai_ans[:200]}")


# %% ── Cell 4: Sequence length statistics ────────────────────────────────────
# Use a random subset of 3000 samples to keep this fast

import random
random.seed(42)
sample_texts = random.sample(all_texts, min(3000, len(all_texts)))

print("Sequence length distribution (tiktoken GPT-2 BPE):")
sequence_length_stats(sample_texts, model_params['tokenizer'])

print(f"\nConfigured max_seq_len = {model_params['max_seq_len']} tokens")


# %% ── Cell 5: Tokenization round-trip check ─────────────────────────────────

enc = tiktoken.get_encoding(model_params['tokenizer'])

test_texts_sample = [
    "The French Revolution began in 1789 and fundamentally changed the political landscape of Europe.",
    "The French Revolution began in 1789. It was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799.",
]
test_labels_sample = [0, 1]

ds_sample = TextClassificationDataset(
    test_texts_sample,
    test_labels_sample,
    model_params['tokenizer'],
    model_params['max_seq_len'],
    min_answer_len=1,   # allow short strings for this test
)

for idx in range(len(ds_sample)):
    sample = ds_sample[idx]
    ids    = sample['input_ids']
    mask   = sample['attention_mask']
    label  = sample['label'].item()

    # Reverse the +1 offset to decode back to text
    real_ids = [tid.item() - 1 for tid, m in zip(ids, mask) if m.item()]
    decoded  = enc.decode(real_ids)

    n_real = mask.sum().item()
    print(f"\nSample {idx}  label={'human' if label==0 else 'AI'}")
    print(f"  input_ids shape:      {ids.shape}  dtype={ids.dtype}")
    print(f"  attention_mask shape: {mask.shape}  dtype={mask.dtype}")
    print(f"  Real tokens: {n_real} / {model_params['max_seq_len']}")
    print(f"  Decoded (first 200 chars): {decoded[:200]}")
    print(f"  Round-trip match: {decoded.strip() == test_texts_sample[idx][:len(decoded)].strip()}")


# %% ── Cell 6: Full DataLoaders — batch shape verification ───────────────────

train_loader, val_loader, test_loader = make_dataloaders(model_params, model_build_options)

batch = next(iter(train_loader))
print(f"\nBatch keys: {list(batch.keys())}")
print(f"  input_ids      shape: {batch['input_ids'].shape}       dtype={batch['input_ids'].dtype}")
print(f"  attention_mask shape: {batch['attention_mask'].shape}  dtype={batch['attention_mask'].dtype}")
print(f"  label          shape: {batch['label'].shape}           dtype={batch['label'].dtype}")
print(f"\n  Label distribution in this batch: "
      f"human={( batch['label']==0).sum().item()}, "
      f"AI={(batch['label']==1).sum().item()}")

print(f"\nDataLoader sizes:")
print(f"  train: {len(train_loader):>4} batches  ({len(train_loader.dataset):,} samples)")
print(f"  val:   {len(val_loader):>4} batches  ({len(val_loader.dataset):,} samples)")
print(f"  test:  {len(test_loader):>4} batches  ({len(test_loader.dataset):,} samples)")

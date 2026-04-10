#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading, tokenization, and DataLoader construction.

Labels: 0 = human, 1 = AI. Each answer in the dataset becomes an independent
sample. Splits are done at the prompt level so the same question never appears
in both train and test sets.
"""

import json
import os
import random
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# Raw data loading
# ──────────────────────────────────────────────

def load_hc3(data_path: str) -> tuple[list[list[str]], list[list[int]], list[str]]:
    """
    Load HC3 from a local all.jsonl file.
    Returns prompt_texts, prompt_labels, and questions as parallel lists grouped by prompt.
    """
    jsonl_path = os.path.join(data_path, 'all.jsonl')
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"HC3 data not found at '{jsonl_path}'.\n"
            f"Download it with: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='Hello-SimpleAI/HC3', repo_type='dataset', "
            f"local_dir='{data_path}')\""
        )

    rows = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    prompt_texts  = []
    prompt_labels = []
    questions     = []

    for row in rows:
        answers = []
        labels  = []

        for ans in row['human_answers']:
            if ans and ans.strip():
                answers.append(ans.strip())
                labels.append(0)

        chatgpt_ans = row['chatgpt_answers']
        if chatgpt_ans and chatgpt_ans[0] and chatgpt_ans[0].strip():
            answers.append(chatgpt_ans[0].strip())
            labels.append(1)

        if answers:
            prompt_texts.append(answers)
            prompt_labels.append(labels)
            questions.append(row['question'])

    return prompt_texts, prompt_labels, questions


def flatten_prompts(
    prompt_texts:  list[list[str]],
    prompt_labels: list[list[int]],
) -> tuple[list[str], list[int]]:
    """Flatten prompt-grouped lists into parallel (text, label) lists."""
    texts  = [ans for group in prompt_texts  for ans in group]
    labels = [lbl for group in prompt_labels for lbl in group]
    return texts, labels


# ──────────────────────────────────────────────
# Length statistics
# ──────────────────────────────────────────────

def sequence_length_stats(texts: list[str], tokenizer_name: str = 'gpt2') -> None:
    """Prints percentile breakdown of tokenized sequence lengths across a list of texts."""
    enc = tiktoken.get_encoding(tokenizer_name)
    lengths = np.array([len(enc.encode(t)) for t in texts])
    print(f"Sequence length stats (n={len(lengths):,}):")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  {p:3d}th percentile: {int(np.percentile(lengths, p)):>5d} tokens")


# ──────────────────────────────────────────────
# Prompt-level split
# ──────────────────────────────────────────────

def split_prompts(
    prompt_texts:  list[list[str]],
    prompt_labels: list[list[int]],
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """
    Splits at the prompt level to prevent the same question appearing in multiple splits.
    Returns (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels).
    """
    rng = random.Random(seed)
    indices = list(range(len(prompt_texts)))
    rng.shuffle(indices)

    n_train = int(len(prompt_texts) * train_frac)
    n_val   = int(len(prompt_texts) * val_frac)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    def collect(idx_list):
        texts  = [t for i in idx_list for t in prompt_texts[i]]
        labels = [l for i in idx_list for l in prompt_labels[i]]
        return texts, labels

    train_texts, train_labels = collect(train_idx)
    val_texts,   val_labels   = collect(val_idx)
    test_texts,  test_labels  = collect(test_idx)

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class TextClassificationDataset(Dataset):
    """
    Tokenizes and caches sequences at construction time.
    Token IDs are offset by +1 so index 0 is reserved as PAD.
    Sequences are truncated to max_seq_len then right-padded to that length.
    """

    def __init__(
        self,
        texts:          list[str],
        labels:         list[int],
        tokenizer_name: str = 'gpt2',
        max_seq_len:    int = 256,
        min_answer_len: int = 20,
    ):
        enc = tiktoken.get_encoding(tokenizer_name)

        self.input_ids       = []
        self.attention_masks = []
        self.labels          = []

        skipped = 0
        for text, label in zip(texts, labels):
            ids = [tid + 1 for tid in enc.encode(text)]  # +1 offset; 0 = PAD

            if len(ids) < min_answer_len:
                skipped += 1
                continue

            ids    = ids[:max_seq_len]
            pad_len = max_seq_len - len(ids)

            self.input_ids.append(torch.tensor(ids + [0] * pad_len, dtype=torch.long))
            self.attention_masks.append(torch.tensor([True] * len(ids) + [False] * pad_len, dtype=torch.bool))
            self.labels.append(torch.tensor(label, dtype=torch.long))

        if skipped:
            print(f"  [data] Skipped {skipped:,} answers shorter than {min_answer_len} tokens.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label':          self.labels[idx],
        }


# ──────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────

def make_dataloaders(
    model_params:        dict,
    model_build_options: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loads the dataset, splits it, tokenizes, and returns train/val/test DataLoaders."""
    print("Loading HC3 dataset from disk...")
    prompt_texts, prompt_labels, _ = load_hc3(model_build_options['hc3_data_dir'])

    n_prompts = len(prompt_texts)
    n_samples = sum(len(g) for g in prompt_texts)
    n_human   = sum(1 for group in prompt_labels for l in group if l == 0)
    n_ai      = sum(1 for group in prompt_labels for l in group if l == 1)
    print(f"  Prompts: {n_prompts:,} | Total samples: {n_samples:,} "
          f"(human={n_human:,}, AI={n_ai:,})")

    print("Splitting at prompt level (train/val/test)...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_prompts(
        prompt_texts, prompt_labels,
        train_frac=model_build_options['train_frac'],
        val_frac=model_build_options['val_frac'],
        seed=model_build_options['seed'],
    )
    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    tokenizer_name = model_params['tokenizer']
    max_seq_len    = model_params['max_seq_len']
    min_answer_len = model_build_options['min_answer_len']
    batch_size     = model_build_options['batch_size']

    print("Building datasets and tokenizing...")
    train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer_name, max_seq_len, min_answer_len)
    val_ds   = TextClassificationDataset(val_texts,   val_labels,   tokenizer_name, max_seq_len, min_answer_len)
    test_ds  = TextClassificationDataset(test_texts,  test_labels,  tokenizer_name, max_seq_len, min_answer_len)
    print(f"  Post-filter — Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader

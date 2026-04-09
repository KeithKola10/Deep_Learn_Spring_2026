#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training loop functions for the Transformer text classifier.

No CLI here — this module is imported by run.py.

Key details:
- Optimizer: AdamW
- Scheduler: OneCycleLR (linear warmup + cosine annealing)
- Loss: CrossEntropyLoss
- Gradient clipping applied before each optimizer step
- Early stopping monitors val F1 (macro) — more robust than accuracy for
  mildly imbalanced data
- Best model is saved to results_dir/best_model_{model_tag}.pt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from model import TransformerClassifier


# ──────────────────────────────────────────────
# Single epoch
# ──────────────────────────────────────────────

def train_one_epoch(
    model:      TransformerClassifier,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler._LRScheduler,
    criterion:  nn.Module,
    device:     torch.device,
    grad_clip:  float,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns
    -------
    avg_loss : float — mean cross-entropy loss over all batches
    accuracy : float — fraction of correct predictions
    """
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels    = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# ──────────────────────────────────────────────
# Evaluation (val or test)
# ──────────────────────────────────────────────

def evaluate_split(
    model:     TransformerClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate on a val or test DataLoader.

    Returns
    -------
    avg_loss : float
    accuracy : float
    f1_macro : float — macro-averaged F1 across both classes
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids, attn_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, f1_macro


# ──────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────

def run_training(
    model:               TransformerClassifier,
    train_loader:        DataLoader,
    val_loader:          DataLoader,
    device:              torch.device,
    model_params:        dict,
    model_build_options: dict,
) -> dict:
    """
    Full training loop with early stopping and checkpointing.

    Monitors val F1 (macro). Saves best checkpoint to
    results_dir/best_model_{model_tag}.pt.

    Returns
    -------
    history : dict with keys
        'train_loss', 'train_acc',
        'val_loss',   'val_acc',  'val_f1'
        (each a list of per-epoch values)
    """
    tag             = model_params['model_tag']
    results_dir     = model_build_options['results_dir']
    checkpoint_path = model_build_options['checkpoint_path']
    os.makedirs(results_dir, exist_ok=True)

    lr           = model_build_options['lr']
    weight_decay = model_build_options['weight_decay']
    max_epochs   = model_build_options['max_epochs']
    patience     = model_build_options['patience']
    grad_clip    = model_build_options['grad_clip']
    warmup_frac  = model_build_options['warmup_frac']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max_epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_frac,         # fraction of steps used for warmup
        anneal_strategy='cos',
        div_factor=25.0,               # initial lr = max_lr / 25
        final_div_factor=1e4,
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [], 'val_f1': [],
    }

    best_val_f1     = -1.0
    epochs_no_improve = 0

    print(f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'ValF1':>7}  {'LR':>10}")
    print("─" * 72)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, grad_clip
        )
        val_loss, val_acc, val_f1 = evaluate_split(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>5}  {train_loss:>9.4f}  {train_acc:>8.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_f1:>7.4f}  {current_lr:>10.2e}")

        # Checkpoint on improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"         ↑ New best val F1: {best_val_f1:.4f}  (saved to {checkpoint_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement in val F1 for {patience} epochs).")
                break

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return history

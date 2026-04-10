#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation functions: metrics, confusion matrix, and training curve plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from model import TransformerClassifier


def get_predictions(
    model:  TransformerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over a DataLoader and returns (predictions, labels) as arrays."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['label']

            logits = model(input_ids, attn_mask)
            preds  = logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_preds), np.array(all_labels)


def evaluate_test_set(
    model:               TransformerClassifier,
    test_loader:         DataLoader,
    device:              torch.device,
    model_params:        dict,
    model_build_options: dict,
) -> dict:
    """
    Loads the best checkpoint and evaluates on the held-out test set.
    Prints a classification report, saves a confusion matrix plot, and returns a metrics dict.
    """
    tag             = model_params['model_tag']
    results_dir     = model_build_options['results_dir']
    checkpoint_path = model_build_options['checkpoint_path']

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Run training first (python run.py -v v1)."
        )

    print(f"\nLoading best checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    preds, labels = get_predictions(model, test_loader, device)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_per   = f1_score(labels, preds, average=None)

    print("\n── Test Set Results ──────────────────────────────")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print()
    print(classification_report(labels, preds, target_names=['Human', 'AI (ChatGPT)']))

    if model_build_options.get('save_results', 1):
        os.makedirs(results_dir, exist_ok=True)
        cm_path = os.path.join(results_dir, f'confusion_matrix_{tag}.png')
        plot_confusion_matrix(labels, preds, save_path=cm_path)
        print(f"  Confusion matrix saved to: {cm_path}")

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_human': float(f1_per[0]),
        'f1_ai':    float(f1_per[1]),
    }


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, save_path: str) -> None:
    """Saves a normalized confusion matrix as a PNG."""
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_names = ['Human', 'AI']
    ax.set_xticks([0, 1]);  ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Normalized Confusion Matrix')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.2f}\n(n={cm[i,j]:,})',
                    ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: dict, save_path: str, showplots: bool = False) -> None:
    """Saves train/val loss and validation accuracy + F1 curves as a PNG."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history['train_loss'], label='Train loss', color='steelblue')
    ax1.plot(epochs, history['val_loss'],   label='Val loss',   color='coral')
    ax1.set_xlabel('Epoch');  ax1.set_ylabel('Cross-entropy loss')
    ax1.set_title('Loss');    ax1.legend();  ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['val_acc'], label='Val accuracy',  color='mediumseagreen')
    ax2.plot(epochs, history['val_f1'],  label='Val F1 (macro)', color='mediumpurple')
    ax2.set_xlabel('Epoch');  ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics');  ax2.legend();  ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.suptitle('Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to: {save_path}")

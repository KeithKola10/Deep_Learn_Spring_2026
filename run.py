#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI entry point: build, train, and evaluate the Transformer classifier.

Usage
-----
    python run.py           # uses v1 config by default
    python run.py -v v1     # explicitly use model_configs_v1.py
    python run.py -v v2     # use model_configs_v2.py

The -v flag dynamically imports model_configs_{version}.py from model_configs/.
Checkpoints are saved to models/. Results (plots, metrics) go to results/{model_tag}/.
"""

import argparse
import importlib
import os
import sys
import random
import numpy as np
import torch

import CONFIG

# Allow importing configs from model_configs/
sys.path.insert(0, CONFIG.model_configs_dir)

from data     import make_dataloaders
from model    import TransformerClassifier
from train    import run_training
from evaluate import evaluate_test_set, plot_training_curves


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(version: str) -> tuple[dict, dict]:
    """Dynamically import model_configs_{version}.py from model_configs/ and return both dicts."""
    module_name = f'model_configs_{version}'
    try:
        cfg = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise SystemExit(
            f"Config module '{module_name}.py' not found in {CONFIG.model_configs_dir}.\n"
            f"Create it following the format of model_configs/model_configs_v1.py."
        )
    return cfg.model_params, cfg.model_build_options


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate the Transformer AI text detector.'
    )
    parser.add_argument(
        '-v', '--version',
        default='v1',
        help='Config version to use, e.g. v1 loads model_configs_v1.py (default: v1)'
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────
    print(f"Loading config: model_configs/model_configs_{args.version}.py")
    model_params, model_build_options = load_config(args.version)
    model_build_options = dict(model_build_options)   # don't mutate the imported dict
    tag = model_params['model_tag']
    print(f"  model_tag:   {tag}")
    print(f"  d_model:     {model_params['d_model']}")
    print(f"  n_layers:    {model_params['n_layers']}")
    print(f"  n_heads:     {model_params['n_heads']}")
    print(f"  pooling:     {model_params['pooling']}")
    print(f"  max_seq_len: {model_params['max_seq_len']}")

    # ── Resolve output directories ─────────────
    # Checkpoints → models/best_model_{tag}.pt
    os.makedirs(CONFIG.models_dir, exist_ok=True)
    checkpoint_path = os.path.join(CONFIG.models_dir, f'best_model_{tag}.pt')

    # Results (plots) → results/{tag}/
    results_dir = os.path.join(CONFIG.results_dir, tag)
    os.makedirs(results_dir, exist_ok=True)
    model_build_options['results_dir']     = results_dir
    model_build_options['checkpoint_path'] = checkpoint_path

    print(f"\n  Checkpoint → {checkpoint_path}")
    print(f"  Results    → {results_dir}/")

    # ── Reproducibility ───────────────────────
    set_seed(model_build_options['seed'])

    # ── Device ────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')   # Apple Silicon GPU
    else:
        device = torch.device('cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print(f"  GPU: Apple Silicon (MPS)")

    # ── Data ──────────────────────────────────
    train_loader, val_loader, test_loader = make_dataloaders(model_params, model_build_options)

    # ── Model ─────────────────────────────────
    model = TransformerClassifier(model_params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {tag}  ({n_params:,} parameters)")

    # ── Train ─────────────────────────────────
    checkpoint_exists = os.path.exists(checkpoint_path)
    if checkpoint_exists:
        print(f"\nCheckpoint found — skipping training, running evaluation only.")
        print(f"  ({checkpoint_path})")
        print(f"  Delete the checkpoint to retrain from scratch.")
    else:
        history = run_training(
            model, train_loader, val_loader, device,
            model_params, model_build_options
        )

        if model_build_options.get('save_results', 1):
            curves_path = os.path.join(results_dir, f'training_curves_{tag}.png')
            plot_training_curves(
                history,
                save_path=curves_path,
                showplots=bool(model_build_options.get('showplots', 0)),
            )

    # ── Evaluate on test set ──────────────────
    metrics = evaluate_test_set(
        model, test_loader, device, model_params, model_build_options
    )

    print("\n── Final Metrics ────────────────────────────────")
    print(f"  Accuracy:   {metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (human): {metrics['f1_human']:.4f}")
    print(f"  F1 (AI):    {metrics['f1_ai']:.4f}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == '__main__':
    main()

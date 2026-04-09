#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import CONFIG

"""
Parameters and options that control model building and training.

transformer_v1 - Baseline Transformer encoder for AI text detection
- 3-layer Pre-LayerNorm Transformer encoder built from scratch in PyTorch
- tiktoken GPT-2 BPE tokenizer (vocab_size=50258, index 0 reserved for PAD)
- Mean pooling over non-padding token positions
- Trained on HC3 (Hello-SimpleAI/HC3) with prompt-level train/val/test split
- AdamW optimizer + OneCycleLR (linear warmup + cosine decay)
- Early stopping on val F1, patience=4

Usage:
    python run.py -v v1
"""

# %% MODEL PARAMETERS

model_params = {}
'''Model Parameters — determine the architecture of the Transformer classifier'''

model_params['model_tag']       = 'transformer_v1'  # used to name checkpoints and results

# Tokenizer
model_params['tokenizer']       = 'gpt2'    # tiktoken encoding name
model_params['vocab_size']      = 50258     # GPT-2 BPE vocab (50257 tokens) + 1 PAD at index 0
model_params['max_seq_len']     = 256       # truncate/pad all sequences to this length

# Transformer encoder
model_params['d_model']         = 128       # embedding dim and hidden dim throughout
model_params['n_heads']         = 4         # number of attention heads (d_model must be divisible by n_heads)
model_params['n_layers']        = 3         # number of stacked encoder layers
model_params['d_ff']            = 256       # feedforward inner dim (typically 2x d_model)
model_params['dropout']         = 0.1       # dropout rate applied throughout (attention, FF, embeddings)

# Pooling strategy
model_params['pooling']         = 'mean'    # 'mean' = masked mean pool | 'cls' = first token

# Classification head
model_params['head_hidden_dim'] = 64        # intermediate dim in 2-layer classification head


# %% TRAINING PARAMETERS

model_build_options = {}
'''Model build options — determine how the model is trained and evaluated'''

# Optimizer
model_build_options['optimizer']        = 'adamw'   # 'adamw' only for now
model_build_options['lr']               = 3e-4      # peak learning rate
model_build_options['weight_decay']     = 1e-2      # L2 regularization on weights

# Learning rate schedule
model_build_options['scheduler']        = 'onecycle'    # 'onecycle' (warmup + cosine decay)
model_build_options['warmup_frac']      = 0.1           # fraction of total steps used for linear warmup

# Training loop
model_build_options['batch_size']       = 64
model_build_options['max_epochs']       = 30
model_build_options['grad_clip']        = 1.0       # max gradient norm (prevents exploding gradients)
model_build_options['seed']             = 42

# Early stopping — monitors val F1 (not accuracy; dataset is mildly imbalanced)
model_build_options['patience']         = 4         # epochs without val F1 improvement before stopping

# Data
model_build_options['hc3_data_dir']     = CONFIG.hc3_data_dir
model_build_options['train_frac']       = 0.80
model_build_options['val_frac']         = 0.10
# test_frac is implicitly 1 - train_frac - val_frac = 0.10
model_build_options['min_answer_len']   = 20        # drop answers tokenizing to fewer than this many tokens

# Output
model_build_options['results_dir']      = CONFIG.results_dir
model_build_options['models_dir']       = CONFIG.models_dir
model_build_options['do_train']         = 1         # set to 0 to skip training (evaluation only)
model_build_options['showplots']        = 0         # plots saved to results/ (no popup when running from terminal)
model_build_options['save_results']     = 1         # save metrics + plots to results_dir

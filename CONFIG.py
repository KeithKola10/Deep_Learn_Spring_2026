#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project-level configuration — paths for all directories.

Import this in any module that needs to reference project directories:
    import CONFIG
    path = CONFIG.models_dir
"""

import os

# Root of this project
project_dir = '/Users/kolak/boltznet/deep_learning_proj'

# Versioned model config files (model_configs_v1.py, model_configs_v2.py, ...)
model_configs_dir = os.path.join(project_dir, 'model_configs')

# Saved model checkpoints (best_model_transformer_v1.pt, ...)
models_dir = os.path.join(project_dir, 'models')

# Training curves, confusion matrices, and metrics output
results_dir = os.path.join(project_dir, 'results')

# Downloaded HC3 dataset (parquet files)
hc3_data_dir = os.path.join(project_dir, 'data', 'HC3')

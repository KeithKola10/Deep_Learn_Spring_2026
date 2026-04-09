#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-style interactive script for explaining model predictions via gradient saliency.

For each input text, the model reports:
  - Prediction (Human / AI) and confidence %
  - Top tokens ranked by importance (gradient saliency)
  - A saved PNG bar chart

Method: Input Gradient Saliency
  Importance score per token = L2 norm of the gradient of the predicted class
  logit with respect to the token's embedding: ||d(logit) / d(emb_i)||_2

  This shows which token embeddings, if perturbed, would most change the
  prediction — a more principled measure than raw attention weights.

BPE token merging:
  GPT-2 BPE subword tokens are merged back into whole words before display.
  Tokens that continue a word (no leading space) are grouped with their
  preceding token and their importance scores are summed.

Run cells individually in VSCode (Shift+Enter) or execute the whole file.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works on Mac without display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import tiktoken
import CONFIG
sys.path.insert(0, CONFIG.model_configs_dir)


# %% ── Cell 1: Imports, config, device ───────────────────────────────────────

from model_configs_v1 import model_params, model_build_options
from model import TransformerClassifier

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

enc = tiktoken.get_encoding(model_params['tokenizer'])

tag             = model_params['model_tag']
checkpoint_path = os.path.join(CONFIG.models_dir, f'best_model_{tag}.pt')
results_dir     = os.path.join(CONFIG.results_dir, tag)
os.makedirs(results_dir, exist_ok=True)

LABEL_NAMES = {0: 'Human', 1: 'AI (ChatGPT)'}

print(f"Device:     {device}")
print(f"Model tag:  {tag}")
print(f"Checkpoint: {checkpoint_path}")


# %% ── Cell 2: Load model and checkpoint ─────────────────────────────────────

model = TransformerClassifier(model_params).to(device)

if not os.path.exists(checkpoint_path):
    print(f"[!] No checkpoint found at '{checkpoint_path}'")
    print(f"    Train the model first:  python run.py -v v1")
else:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint loaded.  ({n_params:,} parameters)")


# %% ── Cell 3: Helper — tokenize a raw text string ───────────────────────────

def tokenize(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text with tiktoken, apply +1 offset (0=PAD), pad/truncate to
    max_seq_len.  Returns (input_ids [1,T], attention_mask [1,T]) on device.
    """
    max_len = model_params['max_seq_len']
    ids     = [tid + 1 for tid in enc.encode(text)][:max_len]
    pad_len = max_len - len(ids)
    ids_pad = ids + [0] * pad_len
    mask    = [True] * len(ids) + [False] * pad_len

    input_ids_t = torch.tensor([ids_pad], dtype=torch.long).to(device)
    mask_t      = torch.tensor([mask],    dtype=torch.bool ).to(device)
    return input_ids_t, mask_t


# %% ── Cell 4: Helper — gradient saliency ────────────────────────────────────

def get_saliency(
    input_ids_t:    torch.Tensor,   # [1, T]
    attention_mask: torch.Tensor,   # [1, T]
) -> tuple[np.ndarray, int, float]:
    """
    Run forward_explain, backprop on predicted logit, return per-token importance.

    Returns
    -------
    importance  : np.ndarray [T] — L2 gradient norm per token, normalized to [0,1]
    pred_class  : int             — 0 = Human, 1 = AI
    confidence  : float           — softmax probability of predicted class (0-1)
    """
    model.eval()
    logits, grads = model.forward_explain(input_ids_t, attention_mask)

    probs      = torch.softmax(logits, dim=-1)[0]
    pred_class = logits.argmax(-1).item()
    confidence = probs[pred_class].item()

    # Backward on predicted class logit to fill grads['emb']
    logits[0, pred_class].backward()

    raw_importance = grads['emb'].norm(dim=-1).squeeze(0).detach().cpu().numpy()  # [T]

    # Zero out PAD positions before normalizing
    mask_np = attention_mask.squeeze(0).cpu().numpy()
    raw_importance = raw_importance * mask_np

    # Normalize to [0, 1]
    max_val = raw_importance.max()
    if max_val > 0:
        importance = raw_importance / max_val
    else:
        importance = raw_importance

    return importance, int(pred_class), confidence


# %% ── Cell 5: Helper — merge BPE subwords into whole words ──────────────────

def merge_bpe_tokens(
    input_ids_t: torch.Tensor,   # [1, T]
    importance:  np.ndarray,     # [T]
) -> list[tuple[str, float]]:
    """
    Decode token IDs back to text and merge BPE subword pieces into whole words.

    GPT-2 BPE convention: tokens that continue a word do NOT start with a space.
    Tokens that start a new word DO start with a space (or are the first token).
    Importance scores of subword pieces are summed into the parent word.

    Returns list of (word_str, importance_score) pairs, excluding PAD tokens.
    """
    token_ids = input_ids_t.squeeze(0).cpu().tolist()
    words   = []
    current = ''
    score   = 0.0

    for tid, imp in zip(token_ids, importance):
        if tid == 0:   # PAD — end of real content
            break

        token_bytes = enc.decode_single_token_bytes(tid - 1)
        try:
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            token_str = token_bytes.decode('utf-8', errors='replace')

        # A leading space (or first token) marks the start of a new word
        if token_str.startswith(' ') or not current:
            if current:
                words.append((current.strip(), score))
            current = token_str
            score   = float(imp)
        else:
            # Subword continuation — merge into current word
            current += token_str
            score   += float(imp)

    if current:
        words.append((current.strip(), score))

    return words


# %% ── Cell 6: Helper — print and plot explanation ───────────────────────────

def print_explanation(
    words:       list[tuple[str, float]],
    pred_class:  int,
    confidence:  float,
    text:        str,
    top_n:       int = 12,
):
    preview = text[:90].replace('\n', ' ')
    print(f"\n{'─'*62}")
    print(f"  Input:      \"{preview}{'...' if len(text) > 90 else ''}\"")
    print(f"  Prediction: {LABEL_NAMES[pred_class]}   (confidence: {confidence*100:.1f}%)")
    print(f"{'─'*62}")

    sorted_words = sorted(words, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"  Top {top_n} most important tokens:")
    print(f"  {'─'*42}")
    print(f"  {'Rank':<6}{'Token':<22}{'Importance'}")
    print(f"  {'─'*42}")
    for rank, (word, score) in enumerate(sorted_words, 1):
        print(f"  {rank:<6}{repr(word):<22}{score:.4f}")
    print(f"  {'─'*42}\n")


def plot_explanation(
    words:       list[tuple[str, float]],
    pred_class:  int,
    confidence:  float,
    save_path:   str,
    top_n:       int = 15,
):
    sorted_words = sorted(words, key=lambda x: x[1], reverse=True)[:top_n]
    labels  = [w for w, _ in reversed(sorted_words)]
    scores  = [s for _, s in reversed(sorted_words)]

    color = '#d73027' if pred_class == 1 else '#4575b4'   # red = AI, blue = human

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    bars = ax.barh(labels, scores, color=color, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Importance (normalized gradient saliency)', fontsize=10)
    ax.set_title(
        f'Token Importance — Predicted: {LABEL_NAMES[pred_class]} ({confidence*100:.1f}%)',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{score:.3f}', va='center', fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# %% ── Cell 7: Explain — hardcoded AI text example ───────────────────────────

ai_text = (
    "The phenomenon in question can be attributed to several interrelated factors. "
    "First, the structural conditions that enable this behavior have been well-documented "
    "in the literature. Furthermore, the empirical evidence consistently supports the "
    "hypothesis that external incentives play a significant role in shaping outcomes. "
    "In conclusion, a comprehensive understanding requires integrating multiple perspectives."
)

input_ids_t, mask_t = tokenize(ai_text)
importance, pred, conf = get_saliency(input_ids_t, mask_t)
words = merge_bpe_tokens(input_ids_t, importance)

print_explanation(words, pred, conf, ai_text)
plot_explanation(
    words, pred, conf,
    save_path=os.path.join(results_dir, 'explanation_ai_example.png'),
)


# %% ── Cell 8: Explain — hardcoded human text example ────────────────────────

human_text = (
    "honestly i think the whole debate just comes down to money at the end of the day "
    "like nobody is gonna change their mind no matter what you say, people are set in "
    "their ways and thats just how it is. ive seen this happen so many times already "
    "and it never really changes, you know?"
)

input_ids_t, mask_t = tokenize(human_text)
importance, pred, conf = get_saliency(input_ids_t, mask_t)
words = merge_bpe_tokens(input_ids_t, importance)

print_explanation(words, pred, conf, human_text)
plot_explanation(
    words, pred, conf,
    save_path=os.path.join(results_dir, 'explanation_human_example.png'),
)


# %% ── Cell 9: Explain — your own custom text ────────────────────────────────
# Edit the `custom_text` variable and re-run this cell.

custom_text = """
Paste or type your own text here.
"""

input_ids_t, mask_t = tokenize(custom_text.strip())
importance, pred, conf = get_saliency(input_ids_t, mask_t)
words = merge_bpe_tokens(input_ids_t, importance)

print_explanation(words, pred, conf, custom_text.strip())
plot_explanation(
    words, pred, conf,
    save_path=os.path.join(results_dir, 'explanation_custom.png'),
)


# %% ── Cell 10: Batch explain — N random test-set examples ───────────────────
# Loads the test split (same seed as training) and explains N random samples.
# Saves one PNG per sample to results/{tag}/explanation_batch_NNN.png

import random
from data import make_dataloaders

N_EXAMPLES = 6   # how many random test examples to explain

_, _, test_loader = make_dataloaders(model_params, model_build_options)

# Collect all test samples into a flat list
all_test = []
for batch in test_loader:
    ids   = batch['input_ids']
    masks = batch['attention_mask']
    lbls  = batch['label']
    for i in range(ids.size(0)):
        all_test.append((ids[i].unsqueeze(0), masks[i].unsqueeze(0), lbls[i].item()))

rng = random.Random(0)
samples = rng.sample(all_test, min(N_EXAMPLES, len(all_test)))

print(f"\nBatch explanation — {len(samples)} random test examples:")
for idx, (ids_i, mask_i, true_label) in enumerate(samples):
    ids_i  = ids_i.to(device)
    mask_i = mask_i.to(device)

    importance, pred, conf = get_saliency(ids_i, mask_i)
    words = merge_bpe_tokens(ids_i, importance)

    # Decode full text for preview
    token_ids = ids_i.squeeze(0).cpu().tolist()
    raw_ids   = [tid - 1 for tid in token_ids if tid > 0]
    try:
        full_text = enc.decode(raw_ids)
    except Exception:
        full_text = '(decode error)'

    correct = '✓' if pred == true_label else '✗'
    print(f"\n  [{idx+1}/{len(samples)}] True: {LABEL_NAMES[true_label]}  |  "
          f"Pred: {LABEL_NAMES[pred]} ({conf*100:.1f}%)  {correct}")
    print_explanation(words, pred, conf, full_text, top_n=8)

    save_path = os.path.join(results_dir, f'explanation_batch_{idx:03d}.png')
    plot_explanation(words, pred, conf, save_path, top_n=12)

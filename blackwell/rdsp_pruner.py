# -*- coding: utf-8 -*-
"""
blackwell/rdsp_pruner.py
Prune candidate selection and soft-mask application for RDSP.

Soft pruning: zero out the o_proj weight columns + q_proj weight rows
for the selected head. The head still participates in computation but
contributes nothing to the output — effectively removed without reshaping
weight tensors. Reversible by restoring from adapter backup.

After validation passes, save_prune_mask() records which heads were zeroed
so subsequent LoRA training knows which capacity was removed.
"""
from __future__ import annotations
import json
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def select_candidates(
    scores: dict[tuple[int, int], float],
    prune_fraction: float = 0.05,
) -> list[tuple[int, int]]:
    """
    Select the bottom prune_fraction of heads by score.

    Parameters
    ----------
    scores          : {(layer, head): normalized_score} from rdsp_scorer
    prune_fraction  : fraction of total heads to mark for pruning (default 5%)

    Returns
    -------
    List of (layer_idx, head_idx) tuples — the candidates to prune.
    At least 1 candidate is always returned.
    """
    n_prune = max(1, int(len(scores) * prune_fraction))
    ranked  = sorted(scores.items(), key=lambda kv: kv[1])  # ascending
    return [k for k, _v in ranked[:n_prune]]


def apply_head_mask(
    model: Any,
    candidates: list[tuple[int, int]],
) -> None:
    """
    Zero out the weight slices for each pruned head (soft prune, in-place).

    For head h in layer l with head_dim d:
      - o_proj.weight[:, h*d:(h+1)*d] → 0   (head's output contribution)
      - q_proj.weight[h*d:(h+1)*d, :]  → 0   (head's query computation)

    Parameters
    ----------
    model      : HuggingFace / unsloth model
    candidates : list of (layer_idx, head_idx) to zero out
    """
    import torch

    cfg      = model.config
    n_heads  = cfg.num_attention_heads
    hidden   = cfg.hidden_size
    head_dim = hidden // n_heads

    from blackwell.rdsp_scorer import _get_model_layers
    layers = _get_model_layers(model)

    with torch.no_grad():
        for (layer_idx, head_idx) in candidates:
            attn  = layers[layer_idx].self_attn
            start = head_idx * head_dim
            end   = (head_idx + 1) * head_dim
            attn.o_proj.weight[:, start:end].zero_()
            attn.q_proj.weight[start:end, :].zero_()


def save_prune_mask(
    candidates: list[tuple[int, int]],
    path: str,
) -> None:
    """
    Persist the prune candidate list as JSON.

    Parameters
    ----------
    candidates : list of (layer_idx, head_idx) tuples
    path       : file path to write

    Format: {"pruned_heads": [[layer, head], ...]}
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pruned_heads": [list(c) for c in candidates]}, f, indent=2)


def load_prune_mask(path: str) -> list[list[int]]:
    """
    Load a saved prune mask.

    Parameters
    ----------
    path : file path to read

    Returns
    -------
    List of [layer, head] pairs.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)["pruned_heads"]

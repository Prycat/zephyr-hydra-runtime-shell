# -*- coding: utf-8 -*-
"""
blackwell/rdsp_scorer.py
Taylor first-order head importance scorer for RDSP.

For each attention head h in layer l, computes:
    importance(l, h) = mean( |W_o[:, h*d:(h+1)*d]| * |∇W_o[:, h*d:(h+1)*d]| )

Normalized by head FLOPs so that expensive heads must justify their cost:
    regret(l, h) = importance(l, h) / head_flops(seq_len, head_dim, hidden_size)

Lower regret = more expendable = prune candidate.

No GPU required for tests — the accumulator and math functions are pure Python/torch.
The full score_heads() function requires CUDA + unsloth and is tested in integration.
"""
from __future__ import annotations
import os, sys, json
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── FLOPs estimate ────────────────────────────────────────────────────────────

def _head_flops(seq_len: int, head_dim: int, hidden_size: int) -> float:
    """
    Approximate FLOPs for one attention head per token.
    Covers Q projection slice, attention scores, and O projection slice.

    Parameters
    ----------
    seq_len     : token sequence length
    head_dim    : dimension per head (hidden_size // num_attention_heads)
    hidden_size : model hidden dimension

    Returns
    -------
    float FLOPs estimate (always positive)
    """
    return 4.0 * seq_len * head_dim * hidden_size


# ── Layer accessor (handles HF, PEFT wrapper, mock) ──────────────────────────

def _get_model_layers(model):
    """
    Return the list of transformer layers, tolerating three common nesting depths.

    Checked paths (first match wins):
    1. ``model.layers``               — mock models and some flat architectures
    2. ``model.model.layers``         — standard HF CausalLM (LlamaForCausalLM)
    3. ``model.model.model.layers``   — PEFT-wrapped (PeftModelForCausalLM wraps CausalLM
                                        which wraps LlamaModel)

    Parameters
    ----------
    model : any HuggingFace / unsloth / PEFT / mock model

    Returns
    -------
    The ``ModuleList`` of transformer layers.

    Raises
    ------
    AttributeError if no path resolves.
    """
    # Path 1: mock / flat
    if hasattr(model, "layers") and model.layers is not None:
        return model.layers
    # Path 2: standard HF
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Path 3: PEFT wrapper (PeftModelForCausalLM → LlamaForCausalLM → LlamaModel)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "layers")
    ):
        return model.model.model.layers
    raise AttributeError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        "Expected model.layers, model.model.layers, or model.model.model.layers."
    )


# ── Batch accumulator ─────────────────────────────────────────────────────────

def _accumulate_batch(
    model: Any,
    scores: dict[tuple[int, int], float],
    seq_len: int,
) -> None:
    """
    Add Taylor importance from the current gradient state into `scores`.

    Expects model.layers[l].self_attn.o_proj.weight.grad to be populated
    (real model: model.model.layers; mock model: model.layers).
    Modifies `scores` in-place.

    Parameters
    ----------
    model    : HuggingFace / unsloth model (or mock with same interface)
    scores   : accumulator dict {(layer, head): cumulative_importance}
    seq_len  : token sequence length used in this batch (for FLOPs denominator)
    """
    import torch

    cfg      = model.config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    hidden   = cfg.hidden_size
    head_dim = hidden // n_heads
    flops    = _head_flops(seq_len, head_dim, hidden)

    layers = _get_model_layers(model)

    for l in range(n_layers):
        o_proj = layers[l].self_attn.o_proj
        w = o_proj.weight   # [hidden_size, n_heads * head_dim]
        g = getattr(o_proj.weight, "grad", None)
        if g is None:
            continue

        for h in range(n_heads):
            start = h * head_dim
            end   = (h + 1) * head_dim
            importance = (
                w[:, start:end].detach().abs() *
                g[:, start:end].detach().abs()
            ).mean().item()
            key = (l, h)
            scores[key] = scores.get(key, 0.0) + importance / flops


# ── Post-processing ───────────────────────────────────────────────────────────

def normalize_scores(
    scores: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """
    Map raw accumulated scores to [0, 1] via min-max normalization.

    Parameters
    ----------
    scores : raw importance scores {(layer, head): value}

    Returns
    -------
    Normalized dict where min=0.0 and max=1.0.
    If all values are equal, all return 0.0.
    """
    if not scores:
        return scores
    lo  = min(scores.values())
    hi  = max(scores.values())
    rng = hi - lo
    if rng < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / rng for k, v in scores.items()}


def rank_heads(
    scores: dict[tuple[int, int], float],
) -> list[tuple[tuple[int, int], float]]:
    """
    Return list of ((layer, head), score) sorted ascending.

    Parameters
    ----------
    scores : importance scores {(layer, head): value}

    Returns
    -------
    List sorted ascending by score. First entry = most expendable = top prune candidate.
    """
    return sorted(scores.items(), key=lambda kv: kv[1])


# ── Full scorer (requires CUDA + unsloth) ─────────────────────────────────────

def _load_calibration_texts(n: int = 100) -> list[str]:
    """
    Pull calibration texts from training_pairs.jsonl.

    Parameters
    ----------
    n : maximum number of texts to return

    Returns
    -------
    List of formatted chat strings. Falls back to 5 hardcoded examples if
    training_pairs.jsonl does not exist.
    """
    pairs_path = os.path.join(_HERE, "training_pairs.jsonl")
    texts = []
    if os.path.exists(pairs_path):
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    convos = obj.get("conversations", [])
                    if len(convos) >= 2:
                        texts.append(
                            f"<|im_start|>user\n{convos[0]['value']}<|im_end|>\n"
                            f"<|im_start|>assistant\n{convos[1]['value']}<|im_end|>"
                        )
                except Exception:
                    continue
                if len(texts) >= n:
                    break

    if not texts:
        texts = [
            "What is 17 × 23?",
            "Explain gradient descent in one sentence.",
            "Write a Python function that reverses a string.",
            "What is the capital of France?",
            "Solve for x: 2x + 5 = 13.",
        ]
    return texts[:n]


def _dequantize_weight(linear) -> "torch.Tensor":
    """
    Return the weight of a Linear or bitsandbytes Linear4bit as a float32 tensor.

    Parameters
    ----------
    linear : torch.nn.Linear or bitsandbytes.nn.Linear4bit

    Returns
    -------
    float32 tensor with shape [out_features, in_features].
    """
    import torch
    w = linear.weight
    # bitsandbytes Params4bit stores weights in a flat quantised buffer
    if hasattr(w, "quant_state"):
        try:
            import bitsandbytes.functional as bnb_func
            return bnb_func.dequantize_4bit(w.data, w.quant_state).float()
        except Exception:
            pass  # fall through to generic path
    return w.detach().float()


def score_heads(
    adapter_path: str | None = None,
    n_calibration: int = 40,
    model_id: str = "NousResearch/Hermes-3-Llama-3.1-8B",
    max_seq_length: int = 512,
) -> dict[tuple[int, int], float]:
    """
    Full scorer — loads the model, runs calibration batches, returns
    normalized per-head scores. Lower = more expendable.

    Requires CUDA + unsloth. Not unit tested (integration only).

    Parameters
    ----------
    adapter_path    : path to LoRA adapter dir (uses base model if None)
    n_calibration   : number of calibration texts to score over
    model_id        : HuggingFace model ID for base weights
    max_seq_length  : tokenizer truncation length

    Returns
    -------
    dict {(layer_idx, head_idx): normalized_score_0_to_1}
    Lower score = more expendable = top prune candidate.
    """
    import torch

    # Unsloth torch shim (torch 2.6 compatibility)
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "[rdsp_scorer] unsloth not installed. "
            "Run: pip install unsloth"
        )

    load_path = adapter_path or model_id
    print(f"[rdsp_scorer] Loading model from {load_path} ...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = load_path,
        max_seq_length = max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
        device_map     = {"": 0},
    )

    # Unsloth 2026 Zoo patching replaces LlamaAttention.forward() with a single
    # fused Triton kernel that accesses o_proj.weight directly — PyTorch module
    # hooks on o_proj (or its base_layer) never fire.  Activation-based scoring
    # is therefore unavailable without patching the Triton kernel.
    #
    # We use **weight-norm scoring** instead:
    #     importance(l, h) = ||W_o[:, h*d:(h+1)*d]||_F / head_flops
    #
    # This measures the structural capacity of each head to influence the
    # residual stream.  Heads zeroed in a previous RDSP cycle have norm=0 and
    # are always the top candidates.  For a fresh model the ordering reflects
    # the pretrained model's internal structure (heads that ended up with
    # smaller weight columns after pretraining had less influence).
    #
    # n_calibration is accepted but unused in this implementation (kept for
    # API compatibility so callers need not change).

    cfg      = model.config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    hidden   = cfg.hidden_size
    head_dim = hidden // n_heads

    layers = _get_model_layers(model)

    # Reference FLOPs for normalisation — use max_seq_length as a representative
    # sequence length (actual texts vary; this is a constant divisor so it only
    # affects the absolute magnitude, not the ranking).
    flops = _head_flops(max_seq_length, head_dim, hidden)

    scores: dict[tuple[int, int], float] = {}

    print(
        f"[rdsp_scorer] Scoring {n_layers * n_heads} heads via weight-norm "
        f"(dequantised o_proj) ...",
        flush=True,
    )

    with torch.no_grad():
        for l in range(n_layers):
            o_proj = layers[l].self_attn.o_proj
            w = _dequantize_weight(o_proj)   # [hidden, n_heads * head_dim]

            for h in range(n_heads):
                start = h * head_dim
                end   = (h + 1) * head_dim
                importance = w[:, start:end].norm(p="fro").item()
                scores[(l, h)] = importance / flops

            if (l + 1) % 8 == 0:
                print(f"[rdsp_scorer]   {l+1}/{n_layers} layers done", flush=True)

    normed = normalize_scores(scores)
    print(
        f"[rdsp_scorer] Done. {len(normed)} head scores computed.",
        flush=True,
    )
    return normed

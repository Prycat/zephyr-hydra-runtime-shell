"""Tests for rdsp_pruner.py — prune candidate selection and mask application."""
import pytest
import json
import os
import torch


from blackwell.rdsp_pruner import (
    select_candidates,
    apply_head_mask,
    save_prune_mask,
    load_prune_mask,
)


def _make_scores(n_layers=4, n_heads=8) -> dict:
    """Deterministic scores: head 0 in every layer gets lowest score."""
    scores = {}
    for l in range(n_layers):
        for h in range(n_heads):
            scores[(l, h)] = float(l * n_heads + h) / (n_layers * n_heads)
    return scores


def test_select_candidates_count():
    """select_candidates returns floor(prune_fraction * total) candidates."""
    scores = _make_scores(4, 8)   # 32 heads total
    candidates = select_candidates(scores, prune_fraction=0.25)
    assert len(candidates) == 8   # 25% of 32


def test_select_candidates_are_lowest():
    """Candidates are the heads with the lowest scores."""
    scores = _make_scores(4, 8)
    candidates = select_candidates(scores, prune_fraction=0.25)
    candidate_scores = [scores[c] for c in candidates]
    non_candidate_scores = [v for k, v in scores.items() if k not in set(candidates)]
    assert max(candidate_scores) <= min(non_candidate_scores)


def test_select_candidates_minimum_one():
    """Even with tiny prune_fraction, at least 1 candidate is returned."""
    scores = {(0, h): float(h) for h in range(4)}
    candidates = select_candidates(scores, prune_fraction=0.001)
    assert len(candidates) >= 1


def test_apply_head_mask_zeros_weights():
    """apply_head_mask zeros the o_proj columns for the target head."""
    n_heads = 4
    hidden  = 16
    head_dim = hidden // n_heads

    class FakeAttn:
        def __init__(self):
            self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttn()

    class FakeConfig:
        num_hidden_layers = 1
        num_attention_heads = n_heads
        hidden_size = hidden

    class FakeSub:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()
            self.model = FakeSub()

    model = FakeModel()
    candidates = [(0, 0)]  # prune head 0 of layer 0

    apply_head_mask(model, candidates)

    o_proj_w = model.model.layers[0].self_attn.o_proj.weight
    assert o_proj_w[:, 0:head_dim].abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert o_proj_w[:, head_dim:].abs().max().item() > 0.0


def test_save_load_prune_mask_roundtrip(tmp_path):
    """save_prune_mask / load_prune_mask are inverses."""
    mask_path = str(tmp_path / "prune_mask.json")
    candidates = [(0, 1), (2, 3), (31, 7)]
    save_prune_mask(candidates, mask_path)
    loaded = load_prune_mask(mask_path)
    assert set(tuple(x) for x in loaded) == set(candidates)


def test_save_prune_mask_json_readable(tmp_path):
    """Saved mask is valid JSON with a 'pruned_heads' key."""
    mask_path = str(tmp_path / "prune_mask.json")
    save_prune_mask([(1, 2)], mask_path)
    with open(mask_path) as f:
        obj = json.load(f)
    assert "pruned_heads" in obj
    assert obj["pruned_heads"] == [[1, 2]]

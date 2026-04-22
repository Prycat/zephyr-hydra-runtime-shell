"""
Tests for rdsp_scorer.py — Taylor head importance scorer.
Uses a tiny mock model so no GPU is required.
"""
import pytest
import types
import torch


# ── Minimal mock of a Llama-style model ──────────────────────────────────────

def _make_mock_model(n_layers=2, n_heads=4, hidden=16):
    """Build a minimal object that satisfies rdsp_scorer's interface."""
    head_dim = hidden // n_heads

    class FakeAttn:
        def __init__(self):
            self.o_proj = types.SimpleNamespace(
                weight=torch.randn(hidden, hidden, requires_grad=True)
            )
            self.q_proj = types.SimpleNamespace(
                weight=torch.randn(hidden, hidden, requires_grad=True)
            )

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttn()

    class FakeConfig:
        num_hidden_layers = n_layers
        num_attention_heads = n_heads
        hidden_size = hidden

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()
            self.layers = [FakeLayer() for _ in range(n_layers)]

        def parameters(self):
            for layer in self.layers:
                yield layer.self_attn.o_proj.weight
                yield layer.self_attn.q_proj.weight

        def zero_grad(self):
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    return FakeModel()


from blackwell.rdsp_scorer import (
    _head_flops,
    _accumulate_batch,
    normalize_scores,
    rank_heads,
)


def test_head_flops_positive():
    """FLOPs estimate is positive and proportional to seq_len."""
    f1 = _head_flops(seq_len=128, head_dim=128, hidden_size=4096)
    f2 = _head_flops(seq_len=256, head_dim=128, hidden_size=4096)
    assert f1 > 0
    assert f2 == pytest.approx(f1 * 2, rel=1e-3)


def test_accumulate_batch_shape():
    """_accumulate_batch returns scores dict with correct key count."""
    model = _make_mock_model(n_layers=2, n_heads=4, hidden=16)
    scores = {}
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Fake a backward pass by manually setting .grad on o_proj weights
    for layer in model.layers:
        w = layer.self_attn.o_proj.weight
        w.grad = torch.rand_like(w)

    _accumulate_batch(model, scores, seq_len=32)

    expected_keys = model.config.num_hidden_layers * model.config.num_attention_heads
    assert len(scores) == expected_keys
    for v in scores.values():
        assert v >= 0.0


def test_normalize_scores_range():
    """normalize_scores maps values to [0, 1]."""
    raw = {(0, 0): 10.0, (0, 1): 0.0, (0, 2): 5.0}
    normed = normalize_scores(raw)
    assert min(normed.values()) == pytest.approx(0.0, abs=1e-6)
    assert max(normed.values()) == pytest.approx(1.0, abs=1e-6)


def test_rank_heads_order():
    """rank_heads returns ascending order — lowest score (most expendable) first."""
    scores = {(0, 0): 0.8, (0, 1): 0.1, (0, 2): 0.5}
    ranked = rank_heads(scores)
    assert ranked[0][0] == (0, 1)   # lowest score first
    assert ranked[-1][0] == (0, 0)  # highest score last


def test_rank_heads_length():
    scores = {(i, j): float(i * 4 + j) for i in range(3) for j in range(4)}
    ranked = rank_heads(scores)
    assert len(ranked) == 12


def test_normalize_scores_all_equal():
    """When all scores are equal, all normalized values should be 0.0."""
    raw = {(0, 0): 5.0, (0, 1): 5.0, (0, 2): 5.0}
    normed = normalize_scores(raw)
    for v in normed.values():
        assert v == pytest.approx(0.0, abs=1e-6)

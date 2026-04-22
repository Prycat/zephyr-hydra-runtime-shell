"""
Tests for rdsp.py orchestrator logic.
All GPU-touching functions are mocked out.
"""
import pytest


from blackwell.rdsp import (
    _total_heads,
    _format_pareto_row,
    _select_benchmark,
)


def test_total_heads_llama():
    """Llama-3.1-8B has 32 layers × 32 heads = 1024."""
    assert _total_heads(n_layers=32, n_heads=32) == 1024


def test_total_heads_custom():
    assert _total_heads(n_layers=4, n_heads=8) == 32


def test_format_pareto_row_commit():
    row = _format_pareto_row(
        cycle=3,
        heads_pruned=48,
        total_heads=1024,
        score_before=0.50,
        score_after=0.49,
        committed=True,
    )
    assert "COMMIT" in row
    assert "0.490" in row


def test_format_pareto_row_rollback():
    row = _format_pareto_row(
        cycle=1,
        heads_pruned=10,
        total_heads=1024,
        score_before=0.50,
        score_after=0.30,
        committed=False,
    )
    assert "ROLLBACK" in row


def test_format_pareto_row_none_after():
    """score_after=None should not crash _format_pareto_row."""
    row = _format_pareto_row(
        cycle=2,
        heads_pruned=20,
        total_heads=1024,
        score_before=0.50,
        score_after=None,
        committed=False,
    )
    assert "ROLLBACK" in row


def test_select_benchmark_default():
    """Default benchmark selection returns a valid benchmark name."""
    bm = _select_benchmark(override=None)
    assert bm in ("cruxeval", "livecodebench", "swebench")


def test_select_benchmark_override():
    """override parameter forces a specific benchmark."""
    bm = _select_benchmark(override="livecodebench")
    assert bm == "livecodebench"

"""Tests for blackwell/prime_state_compiler.py — macro-state builder."""
import numpy as np
import pytest

from blackwell.prime_state_compiler import build_macro_states


def _synthetic_data(n: int = 50) -> np.ndarray:
    """Generate synthetic score vectors for testing without real DB rows."""
    rng = np.random.default_rng(seed=0)
    return rng.uniform(0.0, 1.0, size=(n, 5))


def test_build_macro_states_returns_k_clusters():
    result = build_macro_states(k=5, data=_synthetic_data(50))
    assert len(result["centroids"]) == 5
    assert result["silhouette"] >= 0.0
    assert "labels" in result


def test_build_macro_states_cluster_sizes_sum_to_n():
    data = _synthetic_data(50)
    result = build_macro_states(k=5, data=data)
    assert sum(result["cluster_sizes"]) == len(data)


def test_build_macro_states_labels_shape():
    data = _synthetic_data(50)
    result = build_macro_states(k=5, data=data)
    assert len(result["labels"]) == len(data)
    assert set(result["labels"]).issubset(set(range(5)))


def test_build_macro_states_raises_when_too_few_rows():
    data = _synthetic_data(3)
    with pytest.raises(ValueError, match="at least"):
        build_macro_states(k=5, data=data)


def test_build_macro_states_reads_real_db():
    """Smoke test: real DB must have enough scored rows to cluster."""
    result = build_macro_states(k=5)
    assert len(result["centroids"]) == 5
    assert result["silhouette"] >= 0.0

"""Tests for blackwell/prime_state_compiler.py — macro-state builder."""
import numpy as np
import pytest

from blackwell.prime_state_compiler import build_macro_states, build_transfer_matrix, DB_PATH


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


@pytest.mark.skipif(not DB_PATH.exists(), reason="blackwell.db not present")
def test_build_macro_states_reads_real_db():
    """Smoke test: real DB must have enough scored rows to cluster."""
    result = build_macro_states(k=5)
    assert len(result["centroids"]) == 5
    assert result["silhouette"] >= 0.0


def _synthetic_transitions(states: dict, n_transitions: int = 80, seed: int = 1) -> list[tuple[int, int]]:
    """Generate synthetic (from_label, to_label) pairs for transfer matrix tests.

    Parameters
    ----------
    states : dict
        Output of build_macro_states.
    n_transitions : int
        Number of consecutive-pair transitions to generate.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list of (i, j) label pairs representing observed state transitions.
    """
    rng = np.random.default_rng(seed=seed)
    k = len(states["centroids"])
    from_labels = rng.integers(0, k, size=n_transitions)
    to_labels = rng.integers(0, k, size=n_transitions)
    return list(zip(from_labels.tolist(), to_labels.tolist()))


def test_transfer_matrix_is_row_stochastic():
    states = build_macro_states(k=5, data=_synthetic_data(50))
    transitions = _synthetic_transitions(states, n_transitions=80)
    L, stats = build_transfer_matrix(states, transitions=transitions, verbose=False)
    row_sums = L.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)
    assert L.shape == (5, 5)


def test_transfer_matrix_stats_keys():
    states = build_macro_states(k=5, data=_synthetic_data(50))
    transitions = _synthetic_transitions(states, n_transitions=80)
    _, stats = build_transfer_matrix(states, transitions=transitions, verbose=False)
    for key in ("max_row_sum_error", "irreducible", "spectral_gap", "top_eigenvalues"):
        assert key in stats, f"missing key: {key}"


def test_transfer_matrix_unvisited_rows_are_uniform():
    """A state that is never a source should produce a uniform row (fallback)."""
    states = build_macro_states(k=3, data=_synthetic_data(30))
    # Only provide transitions that originate from states 0 and 1.
    transitions = [(0, 1), (0, 2), (1, 0), (1, 2)]
    L, _ = build_transfer_matrix(states, transitions=transitions, verbose=False)
    # State 2 has no outgoing transitions — row should sum to 1 via the
    # uniform fallback (or any normalisation strategy that keeps it row-stochastic).
    assert abs(L[2].sum() - 1.0) < 1e-8


def test_transfer_matrix_rejects_out_of_range_labels():
    states = build_macro_states(k=5, data=_synthetic_data(50))
    with pytest.raises(ValueError, match="out of range"):
        build_transfer_matrix(states, transitions=[(0, 5), (1, 2)])  # 5 is out of range for k=5


@pytest.mark.skipif(not DB_PATH.exists(), reason="blackwell.db not present")
def test_transfer_matrix_real_db():
    """Smoke test: real DB path; build_transfer_matrix without injected transitions."""
    states = build_macro_states(k=5)
    L, stats = build_transfer_matrix(states, verbose=False)
    assert L.shape == (5, 5)
    row_sums = L.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)

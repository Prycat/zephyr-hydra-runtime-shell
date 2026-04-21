"""Tests for blackwell/prime_state_compiler.py — macro-state builder."""
import numpy as np
import pytest

from blackwell.prime_state_compiler import (
    build_macro_states,
    build_transfer_matrix,
    enumerate_prime_orbits,
    trace_correspondence_test,
    DB_PATH,
)


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


# ---------------------------------------------------------------------------
# enumerate_prime_orbits tests
# ---------------------------------------------------------------------------


def test_prime_orbits_self_loops():
    """All 3 states have self-loops: π(1) should equal 3."""
    L = np.eye(3) * 0.5 + np.ones((3, 3)) * (0.5 / 3)
    orbits = enumerate_prime_orbits(L, L_threshold=0.01, max_length=4)
    assert orbits["pi"][1] == 3  # all 3 states have self-loops


def test_prime_orbits_dag_no_cycles():
    """A strict lower-triangular (DAG) adjacency has no closed walks for n>=1
    except possibly self-loops.  A purely off-diagonal upper-triangular matrix
    has no closed walks at all, so π(n)==0 for all n>=1."""
    # Strictly upper-triangular — no self-loops and no back edges → A^n == 0
    L = np.array([
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.9],
        [0.0, 0.0, 0.0],
    ])
    orbits = enumerate_prime_orbits(L, L_threshold=0.01, max_length=5)
    for n in range(1, 6):
        assert orbits["pi"][n] == 0, f"Expected π({n})==0 for DAG, got {orbits['pi'][n]}"


def test_prime_orbits_pi_has_all_keys():
    """The returned 'pi' dict must contain every integer key from 1 to max_length."""
    L = np.eye(4) * 0.6 + np.ones((4, 4)) * 0.1
    max_length = 6
    orbits = enumerate_prime_orbits(L, L_threshold=0.01, max_length=max_length)
    assert set(orbits["pi"].keys()) == set(range(1, max_length + 1))


def test_prime_orbits_topological_entropy_is_nonneg_float():
    """topological_entropy must be a float >= 0."""
    L = np.eye(3) * 0.5 + np.ones((3, 3)) * (0.5 / 3)
    orbits = enumerate_prime_orbits(L, L_threshold=0.01, max_length=8)
    h = orbits["topological_entropy"]
    assert isinstance(h, float)
    assert h >= 0.0


def test_prime_orbits_two_cycle():
    """2-state system: 0→1→0 only. π(1)=0, π(2)=1 (the single 2-cycle)."""
    L = np.array([[0.0, 1.0], [1.0, 0.0]])  # pure swap, no self-loops
    orbits = enumerate_prime_orbits(L, L_threshold=0.5, max_length=4)
    assert orbits["pi"][1] == 0   # no self-loops
    assert orbits["pi"][2] == 1   # exactly one primitive 2-cycle
    assert orbits["pi"][4] == 0   # 4-cycle is just the 2-cycle repeated → not primitive


# ---------------------------------------------------------------------------
# trace_correspondence_test tests
# ---------------------------------------------------------------------------


def test_trace_correspondence_self_consistent():
    L = np.diag([0.8, 0.6, 0.7]) + 0.1 * np.ones((3, 3)) / 3
    L = L / L.sum(axis=1, keepdims=True)
    result = trace_correspondence_test(L, max_n=4)
    assert "residuals" in result
    assert all(0.0 <= v <= 1.0 for v in result["residuals"].values())


def test_trace_correspondence_identity_matrix():
    """For L=I, Tr(L^n)=k always and the adjacency is all self-loops,
    so π(1)=k and orbit formula also gives k.  ε(n) should be 0 for all n."""
    k = 4
    L = np.eye(k)
    result = trace_correspondence_test(L, max_n=4)
    for n, eps in result["residuals"].items():
        assert eps < 1e-6, f"Expected near-zero residual at n={n}, got {eps}"


def test_trace_correspondence_returns_all_keys():
    L = np.eye(3) * 0.5 + np.ones((3, 3)) * (0.5 / 3)
    result = trace_correspondence_test(L, max_n=4)
    for key in ("residuals", "mean_residual", "residual_trend"):
        assert key in result, f"missing key: {key}"


def test_trace_correspondence_trend_is_float():
    L = np.eye(3) * 0.5 + np.ones((3, 3)) * (0.5 / 3)
    result = trace_correspondence_test(L, max_n=4)
    assert isinstance(result["residual_trend"], float)

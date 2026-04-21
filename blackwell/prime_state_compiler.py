"""Prime-State Compiler (PSC) — macro-state builder.

Reads scored exchange vectors from the Blackwell DB and clusters them into
k macro-states using KMeans. The resulting centroids serve as the discrete
state alphabet for the Prime-State Compiler's first compilation stage.

Dimensions: v_accuracy, v_logic, v_tone, v_curiosity, v_safety (all floats
in [0, 1], NULL rows excluded).
"""

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Type alias for an injected transitions sequence: list of (from_label, to_label).
_Transitions = list[tuple[int, int]]

DB_PATH = Path(__file__).parent.parent / "blackwell.db"
DIMS = ["v_accuracy", "v_logic", "v_tone", "v_curiosity", "v_safety"]


def build_macro_states(
    k: int = 20,
    verbose: bool = False,
    data: Optional[np.ndarray] = None,
) -> dict:
    """Cluster scored exchange vectors into k macro-states.

    Parameters
    ----------
    k : int
        Number of macro-state clusters (centroids).
    verbose : bool
        Print cluster sizes and silhouette score when True.
    data : np.ndarray, optional
        Pre-loaded score matrix of shape (n, 5). When provided the DB is not
        read — used for testing and offline analysis.

    Returns
    -------
    dict with keys:
        centroids    : np.ndarray, shape (k, 5)
        labels       : np.ndarray, shape (n,), cluster index per row
        silhouette   : float, silhouette score in [-1, 1] (0.0 if k==1)
        cluster_sizes: list[int], row counts per cluster
        X            : np.ndarray, the input matrix used for clustering

    Raises
    ------
    ValueError
        If fewer than k scored exchanges are available.
    """
    if data is not None:
        X = np.asarray(data, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(DIMS):
            raise ValueError(
                f"data must have shape (n, {len(DIMS)}), got {X.shape}"
            )
        if np.isnan(X).any():
            raise ValueError("Score matrix contains NaN values — check upstream data source")
    else:
        null_checks = " AND ".join(f"{d} IS NOT NULL" for d in DIMS)
        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute(
                f"SELECT {','.join(DIMS)} FROM exchanges WHERE {null_checks}"
            ).fetchall()
        X = np.array(rows, dtype=float)

    if len(X) < k:
        raise ValueError(
            f"Need at least {k} scored exchanges, have {len(X)}"
        )

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    sizes = [int((labels == i).sum()) for i in range(k)]

    if verbose:
        print(f"Cluster sizes: {sizes}\nSilhouette: {sil:.4f}")

    return {
        "centroids": km.cluster_centers_,
        "labels": labels,
        "silhouette": sil,
        "cluster_sizes": sizes,
        "X": X,
    }


def _assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row of X to the nearest centroid (L2 distance).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Score vectors to assign.
    centroids : np.ndarray, shape (k, d)
        Cluster centroids from a prior KMeans fit.

    Returns
    -------
    np.ndarray of int, shape (n,)
        Index of the nearest centroid for each row.
    """
    # Squared L2: (n, k) matrix of distances
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
    sq_dists = (diffs ** 2).sum(axis=2)                        # (n, k)
    return sq_dists.argmin(axis=1)


def build_transfer_matrix(
    states: dict,
    transitions: Optional[_Transitions] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Estimate a row-stochastic transfer matrix over the k macro-states.

    Consecutive exchange pairs within the same session are treated as
    observed state transitions.  Rows for states with zero observed outgoing
    transitions are set to the uniform distribution (1/k) to ensure the
    matrix is always row-stochastic.

    Parameters
    ----------
    states : dict
        Output of :func:`build_macro_states`.  Must contain ``"centroids"``.
    transitions : list of (int, int), optional
        Pre-built list of ``(from_label, to_label)`` pairs.  When provided,
        the DB is not queried — used for testing and offline analysis.
        Labels must be in ``[0, k)``.
    verbose : bool
        Print stats dict when True.

    Returns
    -------
    L : np.ndarray, shape (k, k)
        Row-stochastic transfer matrix.  ``L[i, j]`` is the empirical
        probability of transitioning from macro-state i to macro-state j.
    stats : dict
        ``max_row_sum_error`` : float — max |row_sum - 1| across all rows.
        ``irreducible``       : bool  — True when every state is reachable
                                        from every other in at most k steps.
        ``spectral_gap``      : float — 1 - |λ₂|, where λ₂ is the second-
                                        largest eigenvalue by magnitude.
        ``top_eigenvalues``   : list[complex] — up to 5 eigenvalues sorted
                                        by descending magnitude.

    Notes
    -----
    When *transitions* is ``None`` the DB is queried with::

        SELECT session_id, turn, v_accuracy, v_logic, v_tone, v_curiosity,
               v_safety
        FROM exchanges
        WHERE v_accuracy IS NOT NULL AND v_logic IS NOT NULL
              AND v_tone IS NOT NULL AND v_curiosity IS NOT NULL
              AND v_safety IS NOT NULL
        ORDER BY session_id, turn

    Score vectors are then re-assigned to their nearest centroid (L2 norm)
    rather than relying on ``states["labels"]``, which may be ordered
    differently.
    """
    centroids: np.ndarray = states["centroids"]
    k = len(centroids)

    if transitions is None:
        # Re-query DB in session order and re-assign labels via nearest centroid.
        null_checks = " AND ".join(f"{d} IS NOT NULL" for d in DIMS)
        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute(
                f"SELECT session_id, turn, {','.join(DIMS)} "
                f"FROM exchanges "
                f"WHERE {null_checks} "
                f"ORDER BY session_id, turn"
            ).fetchall()

        # Build label sequence using nearest-centroid assignment.
        session_ids = [r[0] for r in rows]
        X_ordered = np.array([r[2:] for r in rows], dtype=float)
        labels_ordered = _assign_labels(X_ordered, centroids)

        transitions = []
        for idx in range(len(rows) - 1):
            if session_ids[idx] == session_ids[idx + 1]:  # same session
                transitions.append((int(labels_ordered[idx]), int(labels_ordered[idx + 1])))

    # Validate labels before accumulating (transitions is always non-None here).
    bad = [(f, t) for f, t in transitions if not (0 <= f < k and 0 <= t < k)]
    if bad:
        raise ValueError(f"Transition labels out of range [0, {k}): {bad[:5]}")

    # Accumulate transition counts.
    counts = np.zeros((k, k), dtype=float)
    for from_label, to_label in transitions:
        counts[int(from_label)][int(to_label)] += 1.0

    # Normalise rows; use uniform distribution for unvisited source states.
    row_sums = counts.sum(axis=1, keepdims=True)
    unvisited = (row_sums.ravel() == 0)
    row_sums[row_sums == 0] = 1.0          # avoid div-by-zero
    L = counts / row_sums
    L[unvisited] = 1.0 / k                 # uniform fallback for unvisited states

    # Spectral analysis.
    evals = np.linalg.eigvals(L)
    evals_sorted = sorted(evals, key=lambda x: -abs(x))
    gap = 1.0 - abs(evals_sorted[1]) if len(evals_sorted) > 1 else 1.0

    stats = {
        "max_row_sum_error": float(abs(L.sum(axis=1) - 1.0).max()),
        "irreducible": bool(np.all(np.linalg.matrix_power((counts > 0).astype(float) + np.eye(k), k) > 0)),
        "spectral_gap": float(np.real(gap)),
        "top_eigenvalues": [complex(e) for e in evals_sorted[:5]],
    }

    if verbose:
        for key, val in stats.items():
            print(f"{key}: {val}")

    return L, stats


def enumerate_prime_orbits(
    L: np.ndarray,
    L_threshold: float = 0.01,
    max_length: int = 8,
) -> dict:
    """Count primitive closed orbits on the adjacency graph of L.

    Uses the Möbius-style inversion of the trace formula for directed graphs:

        Tr(A^n) = Σ_{d | n} d · π(d)

    where π(d) is the number of primitive orbits of length d.  Inverting:

        π(n) = (1/n) · (Tr(A^n) - Σ_{d | n, d < n} d · π(d))

    The topological entropy h is the exponential growth rate of orbit counts,
    estimated via linear regression of log(π(n) · n) on n.

    Parameters
    ----------
    L : np.ndarray, shape (k, k)
        Row-stochastic transfer matrix (or any square matrix).
    L_threshold : float
        Entries of L strictly above this threshold are treated as edges
        (i.e., adjacency weight 1); entries at or below are treated as 0.
    max_length : int
        Maximum orbit length to enumerate.

    Returns
    -------
    dict with keys:
        pi                 : dict[int, int] — primitive orbit counts keyed 1..max_length.
        topological_entropy: float — slope of log(π(n)·n) vs n (≈ growth rate).
        power_law_r2       : float — R² of the linear fit used to estimate h.
    """
    k = L.shape[0]
    A = (L > L_threshold).astype(np.float64)  # adjacency matrix

    pi: dict[int, int] = {}
    for n in range(1, max_length + 1):
        An = np.linalg.matrix_power(A, n)
        total = int(round(An.diagonal().sum()))
        # Subtract contributions of shorter primitive orbits via the trace formula.
        prim_count = total
        for d in range(1, n):
            if n % d == 0:
                prim_count -= d * pi.get(d, 0)
        if prim_count > 0 and prim_count % n != 0:
            import warnings
            warnings.warn(
                f"Möbius inversion produced non-integer primitive count at n={n}: "
                f"prim_count={prim_count}. This may indicate threshold artefacts.",
                RuntimeWarning,
                stacklevel=2,
            )
        pi[n] = max(0, prim_count // n)

    # Topological entropy: π(n) ≈ e^{h·n} / n  ⟹  log(π(n)·n) ≈ h·n
    ns = np.array(sorted(pi.keys()), dtype=float)
    counts = np.array([pi[int(n)] for n in ns], dtype=float)
    valid = counts > 0
    if valid.sum() > 1:
        log_counts = np.log(counts[valid] * ns[valid])
        h = float(np.polyfit(ns[valid], log_counts, 1)[0])
        r2 = float(np.corrcoef(ns[valid], log_counts)[0, 1] ** 2)
    else:
        h, r2 = 0.0, 0.0

    return {"pi": pi, "topological_entropy": h, "power_law_r2": r2}


def trace_correspondence_test(L: np.ndarray, max_n: int = 8) -> dict:
    """Test whether Tr(L^n) ≈ Σ_{d|n} d · π(d) across orbit lengths.

    This is experiment E5 of the Prime-State Compiler.  The left side is the
    empirical power trace of the (float, row-stochastic) transfer matrix; the
    right side is reconstructed from the primitive orbit counts produced by
    :func:`enumerate_prime_orbits` over the binary adjacency graph.

    Because L is a weighted matrix while π(d) counts binary-adjacency orbits,
    the residuals measure the discrepancy between weighted closed-walk counts
    and binary orbit structure.  They will not generally be zero but should
    remain small and bounded when the orbit structure captures the dominant
    dynamics.

    Parameters
    ----------
    L : np.ndarray, shape (k, k)
        Row-stochastic transfer matrix (float values in [0, 1]).
    max_n : int
        Maximum power to test.  Residuals are computed for n = 1 … max_n.

    Returns
    -------
    dict with keys:
        residuals      : dict[int, float] — normalised absolute error ε(n) for
                         each n, defined as
                         |Tr(L^n) - Σ_{d|n} d·π(d)| / (max(|Tr(L^n)|, |formula|) + 1e-10).
        mean_residual  : float — arithmetic mean of ε values.
        residual_trend : float — slope of ε(n) vs n (positive → growing error,
                         negative → shrinking, zero if only one data point).
    """
    orbits = enumerate_prime_orbits(L, max_length=max_n)
    residuals: dict[int, float] = {}
    for n in range(1, max_n + 1):
        Ln = np.linalg.matrix_power(L, n)
        empirical = float(np.trace(Ln))
        formula = sum(d * orbits["pi"].get(d, 0) for d in range(1, n + 1) if n % d == 0)
        eps = abs(empirical - formula) / (max(abs(empirical), abs(formula)) + 1e-10)
        residuals[n] = float(eps)
    ns = list(residuals.keys())
    eps_vals = [residuals[n] for n in ns]
    trend = float(np.polyfit(ns, eps_vals, 1)[0]) if len(ns) > 1 else 0.0
    return {
        "residuals": residuals,
        "mean_residual": float(np.mean(eps_vals)),
        "residual_trend": trend,
    }

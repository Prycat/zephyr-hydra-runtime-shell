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
        "irreducible": bool(np.all(np.linalg.matrix_power(L + np.eye(k), k) > 0)),
        "spectral_gap": float(np.real(gap)),
        "top_eigenvalues": [complex(e) for e in evals_sorted[:5]],
    }

    if verbose:
        for key, val in stats.items():
            print(f"{key}: {val}")

    return L, stats

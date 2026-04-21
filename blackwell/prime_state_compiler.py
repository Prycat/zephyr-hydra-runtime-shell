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
    else:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            f"SELECT {','.join(DIMS)} FROM exchanges WHERE v_accuracy IS NOT NULL"
        ).fetchall()
        con.close()
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

"""
blackwell/calculate_projection.py
Step 2 of the Blackwell training loop — the Newtonian heart.

Given the current average payoff vector x̄, finds the closest point
s* in the Target Set S (orthogonal projection), then computes the
steering vector v = s* - x̄.

For box-constrained S (each dim has [lb, ub]), the projection is
analytically exact: s*_i = clip(x̄_i, lb_i, ub_i).

The steering vector tells the Oracle exactly which direction to push
and by how much — this is the Separating Hyperplane from Blackwell's proof.

Usage:
    python blackwell/calculate_projection.py
"""

import sys
import os
import json
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TARGET_SET_PATH = os.path.join(os.path.dirname(__file__), "target_set.json")
DIMS = ["accuracy", "logic", "tone", "curiosity", "safety"]


def load_target_set() -> dict:
    with open(TARGET_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def project_onto_S(x_bar: dict) -> dict:
    """
    Orthogonal projection of x̄ onto the box-constrained Target Set S.
    s*_i = clip(x̄_i, lb_i, ub_i)
    """
    ts = load_target_set()
    dims = ts["dimensions"]
    projection = {}
    for d in DIMS:
        lb = dims[d]["lb"]
        ub = dims[d]["ub"]
        val = x_bar.get(d, 0.0)
        projection[d] = round(max(lb, min(ub, val)), 4)
    return projection


def steering_vector(x_bar: dict, projection: dict = None) -> dict:
    """
    v = s* - x̄
    Each component is how far and in which direction to steer.
    Positive = need to increase. Zero = already inside S for this dim.
    """
    if projection is None:
        projection = project_onto_S(x_bar)
    return {
        d: round(projection[d] - x_bar.get(d, 0.0), 4)
        for d in DIMS
    }


def weighted_steering_vector(x_bar: dict) -> dict:
    """
    Steering vector weighted by the Target Set's steering_weights.
    This prioritises dimensions Prycat cares most about.
    """
    ts = load_target_set()
    weights = ts.get("steering_weights", {d: 1.0 for d in DIMS})
    s_star = project_onto_S(x_bar)
    return {
        d: round((s_star[d] - x_bar.get(d, 0.0)) * weights.get(d, 1.0), 4)
        for d in DIMS
    }


def in_target_set(x_bar: dict) -> bool:
    """True iff x̄ ∈ S (all dimensions within bounds)."""
    ts = load_target_set()
    dims = ts["dimensions"]
    return all(
        dims[d]["lb"] <= x_bar.get(d, 0.0) <= dims[d]["ub"]
        for d in DIMS
    )


def steering_magnitude(v: dict) -> float:
    """‖v‖₂ — total distance from Target Set boundary."""
    return round(math.sqrt(sum(c**2 for c in v.values())), 4)


def oracle_allocation(v: dict, n_pairs: int = 8) -> dict:
    """
    Allocate training pairs across dimensions proportional to
    weighted steering magnitude. This is the Regret-Matching allocation.
    Dimensions already inside S get 0 pairs.
    """
    # Only allocate to dims that need steering (v > 0)
    needs = {d: max(0.0, c) for d, c in v.items()}
    total = sum(needs.values())
    if total == 0:
        return {d: 0 for d in DIMS}
    return {
        d: round((needs[d] / total) * n_pairs)
        for d in DIMS
    }


def print_projection(x_bar: dict):
    """Pretty-print the full projection analysis."""
    ts = load_target_set()
    dims_cfg = ts["dimensions"]
    s_star = project_onto_S(x_bar)
    v = weighted_steering_vector(x_bar)
    alloc = oracle_allocation(v, n_pairs=8)
    inside = in_target_set(x_bar)

    print("\n=== Blackwell Projection Report ===")
    print(f"{'Dim':<12} {'x̄':>6}  {'s*':>6}  {'v':>7}  {'Alloc':>5}  S bounds")
    print("─" * 65)
    for d in DIMS:
        lb = dims_cfg[d]["lb"]
        ub = dims_cfg[d]["ub"]
        xi = x_bar.get(d, 0.0)
        si = s_star[d]
        vi = v[d]
        ai = alloc[d]
        in_s = "✓" if lb <= xi <= ub else "✗"
        print(f"{d:<12} {xi:>6.3f}  {si:>6.3f}  {vi:>+7.3f}  {ai:>5}  [{lb:.1f}, {ub:.1f}] {in_s}")
    print("─" * 65)
    print(f"‖v‖ = {steering_magnitude(v):.4f}   In Target Set: {'YES ✓' if inside else 'NO ✗'}")
    if inside:
        print("Zephyr's average behavior is inside S. No steering needed.")
    else:
        top = sorted(v, key=lambda d: abs(v[d]), reverse=True)[:2]
        print(f"Steering direction: primarily {top[0]} ({v[top[0]]:+.3f}), then {top[1]} ({v[top[1]]:+.3f})")
    print()
    return v, alloc


if __name__ == "__main__":
    from blackwell.logger import get_average_vector

    x_bar = get_average_vector()
    if x_bar is None:
        print("No scored data yet. Using observed baseline.")
        x_bar = {"accuracy": 0.60, "logic": 0.70, "tone": 0.30,
                 "curiosity": 0.10, "safety": 0.95}

    print_projection(x_bar)

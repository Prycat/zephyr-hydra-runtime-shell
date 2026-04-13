"""
blackwell/regret.py
Regret computation using proper SQL-backed vector averaging
and the Target Set projection geometry.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math

DIMS = ["accuracy", "logic", "tone", "curiosity", "safety"]

# Observed baseline — used when no scored data exists yet
BASELINE = {
    "accuracy":  0.60,
    "logic":     0.70,
    "tone":      0.30,
    "curiosity": 0.10,
    "safety":    0.95,
}


def average_payoff_vector(limit: int = 500) -> dict:
    """
    Get average payoff vector from DB (SQL AVG on numeric columns).
    Falls back to baseline if no scored data.
    """
    from blackwell.logger import get_average_vector
    db_avg = get_average_vector(limit)
    if db_avg is not None:
        return db_avg
    return dict(BASELINE)


def regret_vector(x_bar: dict = None) -> dict:
    """
    Per-dimension regret using Target Set bounds.
    regret_i = max(0, lb_i - x̄_i)  — how far BELOW the lower bound.
    """
    from blackwell.calculate_projection import load_target_set
    if x_bar is None:
        x_bar = average_payoff_vector()
    ts = load_target_set()
    dims_cfg = ts["dimensions"]
    return {
        d: round(max(0.0, dims_cfg[d]["lb"] - x_bar.get(d, 0.0)), 4)
        for d in DIMS
    }


def total_regret(regret_v: dict = None) -> float:
    if regret_v is None:
        regret_v = regret_vector()
    return round(math.sqrt(sum(v**2 for v in regret_v.values())), 4)


def highest_regret_dims(regret_v: dict = None, top_n: int = 2) -> list[str]:
    if regret_v is None:
        regret_v = regret_vector()
    return sorted(regret_v, key=lambda d: regret_v[d], reverse=True)[:top_n]


def print_status(x_bar: dict = None):
    """Pretty-print payoff vector, Target Set bounds, and regret."""
    from blackwell.calculate_projection import load_target_set
    if x_bar is None:
        x_bar = average_payoff_vector()

    ts = load_target_set()
    dims_cfg = ts["dimensions"]
    r = regret_vector(x_bar)
    tr = total_regret(r)

    print("\n=== Blackwell Status Report ===")
    print(f"{'Dim':<12} {'x̄':>6}  {'Target S':>12}  {'Regret':>7}  Progress")
    print("─" * 65)
    for d in DIMS:
        xi  = x_bar.get(d, 0.0)
        lb  = dims_cfg[d]["lb"]
        ub  = dims_cfg[d]["ub"]
        ri  = r[d]
        bar = "█" * int(xi * 20) + "░" * (20 - int(xi * 20))
        flag = " ← GAP" if ri > 0.05 else " ✓"
        print(f"{d:<12} {xi:>6.3f}  [{lb:.1f}, {ub:.1f}]     {ri:>7.4f}  {bar}{flag}")
    print("─" * 65)
    print(f"{'Total regret':>20}  {tr:.4f}  (0 = inside Target Set S)")
    top = highest_regret_dims(r)
    print(f"Oracle targets:  {', '.join(top)}\n")


if __name__ == "__main__":
    x_bar = average_payoff_vector()
    print_status(x_bar)

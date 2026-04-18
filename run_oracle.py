"""
run_oracle.py — The full Blackwell training loop (Steps 1-3).

Step 1: evaluate_vector   → score unscored exchanges, write to DB
Step 2: calculate_projection → find s* ∈ S, compute steering vector v = s* - x̄
Step 3: oracle             → generate counter-regret data along v
        (Step 4: lora_steer → run when dataset is large enough)

Run with: python run_oracle.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from blackwell.logger              import get_average_vector, init_db
from blackwell.regret              import average_payoff_vector, print_status
from blackwell.calculate_projection import (
    project_onto_S, steering_vector, weighted_steering_vector,
    oracle_allocation, print_projection, in_target_set, steering_magnitude
)
from blackwell.oracle              import synthesise, save_training_pairs
from blackwell.evaluate_vector     import run as evaluate_unscored
from blackwell.lora_steer          import check_training_data

N_PAIRS = 8


def main():
    print("=" * 62)
    print("  PRYCAT RESEARCH — BLACKWELL TRAINING LOOP")
    print("  Vector-Space Steering via Orthogonal Projection")
    print("=" * 62)

    init_db()

    # ── Step 1: Evaluate any unscored exchanges ───────────────
    print("\n[ Step 1 ] Evaluating unscored exchanges...")
    n_evaluated = evaluate_unscored(limit=50, verbose=False)
    if n_evaluated > 0:
        print(f"  Evaluated {n_evaluated} new exchanges → written to DB")
    else:
        print("  No new exchanges to evaluate (using baseline vector)")

    # ── Step 2: Get x̄ and compute projection ─────────────────
    print("\n[ Step 2 ] Computing orthogonal projection onto Target Set S...")
    x_bar = average_payoff_vector()
    print_status(x_bar)

    if in_target_set(x_bar):
        print("  ✓ x̄ is already inside Target Set S.")
        print("  No steering required. Run more conversations to collect data.")
        return

    v, alloc = print_projection(x_bar)
    mag = steering_magnitude(v)
    print(f"  Steering vector ‖v‖ = {mag:.4f}")
    print(f"  Oracle allocation: { {d:n for d,n in alloc.items() if n > 0} }")

    # ── Step 3: Oracle — counter-regret synthesis ─────────────
    print(f"\n[ Step 3 ] Oracle generating {N_PAIRS} counter-regret pairs...")
    print("  (This may take 30-90 seconds — Zephyr is writing corrective data)\n")

    pairs = synthesise(x_bar, v, alloc, n_pairs=N_PAIRS)

    if not pairs:
        print("  Oracle returned no pairs. Is Ollama running?")
        print("  Check: ollama serve")
        return

    save_training_pairs(pairs)

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  ORACLE OUTPUT — {len(pairs)} COUNTER-REGRET PAIRS")
    print(f"{'=' * 62}")

    by_dim = {}
    for p in pairs:
        d = p.get("target_dim", "unknown")
        by_dim.setdefault(d, []).append(p)

    for dim, dim_pairs in by_dim.items():
        print(f"\n  ── {dim.upper()} ({len(dim_pairs)} pairs) ──")
        for i, p in enumerate(dim_pairs, 1):
            print(f"\n  [{i}] Human  : {p['human'][:120]}")
            print(f"       Zephyr : {p['zephyr'][:200]}")

    # ── Analysis ──────────────────────────────────────────────
    questions  = [p for p in pairs if "?" in p["zephyr"]]
    avg_words  = sum(len(p["zephyr"].split()) for p in pairs) // len(pairs)
    curiosity_pairs = [p for p in pairs if p.get("target_dim") == "curiosity"]

    print(f"\n{'=' * 62}")
    print(f"  ANALYSIS")
    print(f"{'=' * 62}")
    print(f"  Total pairs generated      : {len(pairs)}")
    print(f"  Pairs with question back   : {len(questions)}/{len(pairs)}")
    print(f"  Avg Zephyr response length : {avg_words} words")
    print(f"  Curiosity pairs            : {len(curiosity_pairs)}")

    if questions:
        print(f"\n  Zephyr's questions this cycle:")
        for p in questions:
            for s in p["zephyr"].split("."):
                if "?" in s:
                    print(f"    → {s.strip()}")

    # ── Phase 2 readiness ─────────────────────────────────────
    print(f"\n{'─' * 62}")
    check_training_data()


if __name__ == "__main__":
    main()

"""
blackwell/evaluate_vector.py
Step 1 of the Blackwell training loop.

Projects each Zephyr exchange into the 5-dimensional reward space
by running the LLM-as-judge evaluator and writing scores back
to the numeric columns in blackwell.db.

Usage:
    python blackwell/evaluate_vector.py          # evaluate all unscored exchanges
    python blackwell/evaluate_vector.py --limit 50
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
from blackwell.logger import DB_PATH, update_scores, DIMS, init_db
from blackwell.evaluator import evaluate_exchange


def get_unscored(limit: int = 100) -> list[dict]:
    """Fetch exchanges that have not yet been evaluated."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, human, zephyr FROM exchanges
           WHERE v_accuracy IS NULL
           ORDER BY timestamp ASC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run(limit: int = 100, verbose: bool = True):
    """Evaluate all unscored exchanges and write scores to DB."""
    unscored = get_unscored(limit)

    if not unscored:
        if verbose:
            print("No unscored exchanges found. All vectors are up to date.")
        return 0

    if verbose:
        print(f"Evaluating {len(unscored)} exchanges...\n")

    evaluated = 0
    for i, ex in enumerate(unscored, 1):
        scores = evaluate_exchange(ex["human"], ex["zephyr"])
        update_scores(ex["id"], scores)
        evaluated += 1

        if verbose:
            v_str = "  ".join(f"{d[:3].upper()}={scores.get(d, 0):.2f}" for d in DIMS)
            note = scores.get("notes", "")[:60]
            print(f"  [{i:>3}/{len(unscored)}]  {v_str}  | {note}")

    if verbose:
        print(f"\nEvaluated {evaluated} exchanges. Run calculate_projection.py next.")

    return evaluated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate unscored Zephyr exchanges")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max exchanges to evaluate (default 100)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(limit=args.limit, verbose=not args.quiet)

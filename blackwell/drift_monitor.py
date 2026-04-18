"""
blackwell/drift_monitor.py
Drift Watchdog — the dead man's switch (Fix B).

Tracks the gap between LLM judge scores and rule-based scores per dimension
across a rolling window of recent evaluations.

  gap_d = llm_score_d - rule_score_d

When the LLM judge starts systematically inflating scores relative to the
rule layer (gap > DRIFT_THRESHOLD), this is a signal that the judge and
the student are co-evolving — the judge has learned to reward exactly
what the student learned to produce, not what is objectively good.

On DRIFT_DETECTED, the system writes a drift_state.json with:
  - which dimensions are drifting
  - the current gap magnitude per dimension
  - an ABORT_TRAIN recommendation

lora_steer checks drift_state.json before launching any training run.

GAP_THRESHOLD tuning:
  0.10  — very sensitive; fires early in the drift curve
  0.20  — balanced (default); requires sustained inflation to trigger
  0.30  — tolerant; only catches gross drift

WINDOW_SIZE:
  Number of recent evaluations to average over.  Too small = noisy.
  Too large = slow to react to real drift.  Default 100.

Usage:
  from blackwell.drift_monitor import record_scores, check_drift, DriftState
"""

import json
import os
import time
import sqlite3
from dataclasses import dataclass, field

DRIFT_STATE_PATH = os.path.join(os.path.dirname(__file__), "drift_state.json")
DB_PATH          = os.path.join(os.path.dirname(__file__), "..", "blackwell.db")

DIMS             = ["accuracy", "logic", "tone", "safety"]  # curiosity has no rule layer
GAP_THRESHOLD    = 0.20   # LLM score - rule score > this → suspect drift
WINDOW_SIZE      = 100    # rolling average over last N evaluations
MIN_SAMPLES      = 10     # need at least this many before triggering


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DriftState:
    timestamp:        str
    drifting_dims:    list[str]           # dims where gap > threshold
    gaps:             dict                # {dim: mean_gap}
    n_samples:        int
    drift_detected:   bool
    abort_train:      bool
    notes:            str = ""


# ── DB schema for gap tracking ────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_gap_table() -> None:
    """Create gap_log table if it doesn't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gap_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                dim         TEXT NOT NULL,
                llm_score   REAL NOT NULL,
                rule_score  REAL NOT NULL,
                gap         REAL NOT NULL
            )
        """)
        conn.commit()


# ── Recording ─────────────────────────────────────────────────────────────────

def record_scores(eval_scores: dict) -> None:
    """
    Record LLM vs rule score gap for this evaluation.

    Call this from background_eval after every scored exchange.
    eval_scores must contain both 'dim' and 'rule_dim' keys
    (the blended evaluator returns both).

    Example keys in eval_scores:
        accuracy, logic, tone, safety          → LLM-blended scores
        rule_accuracy, rule_logic, rule_tone,  → pure rule scores
        rule_safety
    """
    _ensure_gap_table()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for dim in DIMS:
        rule_key = f"rule_{dim}"
        if dim in eval_scores and rule_key in eval_scores:
            llm_score  = float(eval_scores[dim])
            rule_score = float(eval_scores[rule_key])
            gap = round(llm_score - rule_score, 6)
            rows.append((ts, dim, llm_score, rule_score, gap))

    if not rows:
        return

    with _connect() as conn:
        conn.executemany(
            "INSERT INTO gap_log (timestamp, dim, llm_score, rule_score, gap) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()

    # Prune to keep only the last WINDOW_SIZE * 5 rows per dim (housekeeping)
    _prune_gap_log()


def _prune_gap_log(keep: int = WINDOW_SIZE * 5) -> None:
    """Keep gap_log from growing unbounded."""
    try:
        with _connect() as conn:
            for dim in DIMS:
                count = conn.execute(
                    "SELECT COUNT(*) FROM gap_log WHERE dim=?", (dim,)
                ).fetchone()[0]
                if count > keep:
                    excess = count - keep
                    conn.execute(
                        """DELETE FROM gap_log WHERE id IN (
                            SELECT id FROM gap_log WHERE dim=? ORDER BY id ASC LIMIT ?
                        )""",
                        (dim, excess),
                    )
            conn.commit()
    except Exception:
        pass


# ── Gap calculation ───────────────────────────────────────────────────────────

def _get_mean_gaps(window: int = WINDOW_SIZE) -> dict[str, dict]:
    """
    Compute mean gap (llm - rule) per dimension over the last `window` samples.
    Returns {dim: {"mean_gap": float, "n": int}}.
    """
    _ensure_gap_table()
    result = {}
    with _connect() as conn:
        for dim in DIMS:
            rows = conn.execute(
                """SELECT gap FROM gap_log WHERE dim=?
                   ORDER BY id DESC LIMIT ?""",
                (dim, window),
            ).fetchall()
            if rows:
                gaps = [r["gap"] for r in rows]
                result[dim] = {
                    "mean_gap": round(sum(gaps) / len(gaps), 4),
                    "n":        len(gaps),
                }
    return result


# ── Drift check ───────────────────────────────────────────────────────────────

def check_drift(threshold: float = GAP_THRESHOLD,
                window: int = WINDOW_SIZE) -> DriftState:
    """
    Compute the current drift state and persist it to drift_state.json.

    Returns a DriftState.  If drift_detected=True, abort_train=True.

    Logic:
      - For each dimension: mean_gap > threshold → drifting
      - Need >= MIN_SAMPLES for the dimension to count
      - Any drifting dimension → drift_detected = True
    """
    mean_gaps = _get_mean_gaps(window)

    drifting = {}
    for dim, stats in mean_gaps.items():
        if stats["n"] < MIN_SAMPLES:
            continue
        if stats["mean_gap"] > threshold:
            drifting[dim] = stats["mean_gap"]

    n_total = min(stats["n"] for stats in mean_gaps.values()) if mean_gaps else 0
    drift_detected = len(drifting) > 0

    notes_parts = []
    if n_total < MIN_SAMPLES:
        notes_parts.append(
            f"Insufficient samples ({n_total} < {MIN_SAMPLES} required). "
            "Drift detection inactive."
        )
    for dim, gap in drifting.items():
        notes_parts.append(
            f"{dim}: LLM judge inflated by {gap:+.3f} above rule layer "
            f"(threshold {threshold:.2f})"
        )

    state = DriftState(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        drifting_dims=list(drifting.keys()),
        gaps={dim: stats["mean_gap"] for dim, stats in mean_gaps.items()},
        n_samples=n_total,
        drift_detected=drift_detected,
        abort_train=drift_detected and n_total >= MIN_SAMPLES,
        notes=" | ".join(notes_parts) if notes_parts else "No drift detected.",
    )

    _persist_state(state)
    return state


def _persist_state(state: DriftState) -> None:
    """Write drift_state.json so lora_steer can read it."""
    try:
        data = {
            "timestamp":      state.timestamp,
            "drifting_dims":  state.drifting_dims,
            "gaps":           state.gaps,
            "n_samples":      state.n_samples,
            "drift_detected": state.drift_detected,
            "abort_train":    state.abort_train,
            "notes":          state.notes,
        }
        with open(DRIFT_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[drift] could not persist state: {e}", flush=True)


def load_drift_state() -> DriftState | None:
    """
    Read the last persisted drift state from disk.
    Returns None if no state exists yet.
    """
    if not os.path.exists(DRIFT_STATE_PATH):
        return None
    try:
        with open(DRIFT_STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return DriftState(**data)
    except Exception:
        return None


# ── Summary printer ───────────────────────────────────────────────────────────

def print_drift_report(state: DriftState) -> None:
    print("\n[drift] ── Drift Monitor Report ───────────────────────────────")
    print(f"  Timestamp : {state.timestamp}")
    print(f"  Samples   : {state.n_samples} (window={WINDOW_SIZE})")
    print(f"  Threshold : LLM − rule > {GAP_THRESHOLD:.2f}")
    print()
    for dim in DIMS:
        gap = state.gaps.get(dim, None)
        if gap is None:
            continue
        flag = "⚠ DRIFT" if dim in state.drifting_dims else "  OK   "
        print(f"  {flag}  {dim:<10} gap={gap:+.4f}")
    print()
    if state.drift_detected:
        print(f"  !! DRIFT DETECTED in: {state.drifting_dims}")
        print(f"  !! ABORT_TRAIN = {state.abort_train}")
    else:
        print("  No drift detected — Oracle-Evaluator gap within bounds.")
    print(f"  Note: {state.notes}")
    print("[drift] ───────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blackwell drift monitor")
    parser.add_argument("--threshold", type=float, default=GAP_THRESHOLD,
                        help="Gap threshold for drift detection (default 0.20)")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE,
                        help="Rolling window size (default 100)")
    args = parser.parse_args()

    state = check_drift(threshold=args.threshold, window=args.window)
    print_drift_report(state)
    exit(1 if state.abort_train else 0)

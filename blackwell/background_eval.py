"""
blackwell/background_eval.py
Daemon thread that scores Zephyr conversation turns in the background.

Fix 2 (Granularity Trap):
    Oracle is triggered per-dimension rather than on a collapsed scalar.
    Each dimension has its own ceiling; breaching any one is enough to fire.
    This prevents catastrophic single-dimension failures being masked by
    general competence across other dimensions.

Fix 5 (Stagnation Point / Novelty):
    Every exchange is scored for novelty (how unlike recent history it is).
    High-novelty + high-regret  → synthesis pairs doubled (novel failure).
    High-novelty + low-regret   → archived as positive anchor for LoRA replay.
    This ensures the model keeps exploring rather than converging to a narrow
    attractor of familiar question types.
"""

import queue
import threading
import concurrent.futures
import time
import json
import os
from typing import Optional
from blackwell.evaluator import evaluate_exchange, total_regret, regret_from_scores
from blackwell.logger import update_scores, get_average_vector, get_recent_exchange_ids

# ── Oracle state ───────────────────────────────────────────────────────────────
_oracle_last_fired: float = 0.0
_oracle_lock = threading.Lock()

# ── Tuning constants ───────────────────────────────────────────────────────────
ORACLE_TIMEOUT_SECONDS  = 120    # synthesis body timeout (not first token)
ORACLE_COOLDOWN_SECONDS = 60     # minimum gap between Oracle triggers

# Fix 2 — Per-dimension regret ceilings.
# Oracle fires when ANY dimension's gap exceeds its ceiling.
# Tighter ceiling = more sensitive trigger for that dimension.
DIMENSION_CEILINGS = {
    "accuracy":  0.20,   # strict: factual errors are critical
    "logic":     0.25,
    "tone":      0.30,
    "curiosity": 0.35,
    "safety":    0.15,   # strictest: safety failures never tolerated
}

# Legacy scalar threshold kept for _maybe_trigger_oracle fallback path
ORACLE_REGRET_THRESHOLD = 0.25

# File written by _maybe_trigger_oracle so lora_steer can read current target dims
_STEERING_STATE_PATH = os.path.join(os.path.dirname(__file__), "steering_state.json")


# ── Synthesis helpers ─────────────────────────────────────────────────────────

def _call_synthesise_with_timeout(
    avg: dict, steering_v: dict, allocation: dict,
    n_pairs: int, timeout: float
) -> None:
    """Run synthesise() in a thread pool with a hard timeout."""
    def _work():
        from blackwell.oracle import synthesise
        synthesise(avg, steering_v, allocation, n_pairs=n_pairs)

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_work)
    try:
        fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"[oracle] timed out after {timeout}s — skipping synthesis", flush=True)
    except Exception as e:
        print(f"[oracle] synthesis error: {e}", flush=True)
    finally:
        ex.shutdown(wait=False)


# ── Fix 2: Per-dimension breach detection ─────────────────────────────────────

def _breaching_dimensions(avg: dict) -> dict[str, float]:
    """
    Return a dict of {dim: gap} for every dimension whose gap from the
    Target Set projection exceeds that dimension's individual ceiling.

    Gap is computed as:  max(0, projection[d] - avg[d])
    (how far below the lower bound of S we are for dimension d).

    An empty dict means no dimension is breaching → no Oracle trigger.
    """
    try:
        from blackwell.calculate_projection import project_onto_S
        projection = project_onto_S(avg)
        breaching = {}
        for d in avg:
            gap = max(0.0, projection[d] - avg[d])
            ceiling = DIMENSION_CEILINGS.get(d, ORACLE_REGRET_THRESHOLD)
            if gap > ceiling:
                breaching[d] = round(gap, 4)
        return breaching
    except Exception as e:
        print(f"[trajectory] breach-detection error: {e}", flush=True)
        return {}


def _write_steering_state(breaching_dims: dict, n_pairs: int) -> None:
    """Persist which dimensions are being steered so lora_steer can read it."""
    try:
        state = {
            "target_dims":  list(breaching_dims.keys()),
            "gaps":         breaching_dims,
            "n_pairs":      n_pairs,
            "triggered_at": time.time(),
        }
        with open(_STEERING_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[oracle] could not write steering_state.json: {e}", flush=True)


# ── Main Oracle trigger ────────────────────────────────────────────────────────

def _maybe_trigger_oracle(
    threshold: float,
    exchange_id: str = "",
    human: str = "",
    zephyr: str = "",
) -> None:
    """
    Trigger Oracle synthesis if any dimension breaches its ceiling and
    the cooldown has elapsed.

    Fix 2: fires on per-dimension gaps, not a single scalar.
    Fix 5: doubles n_pairs when the triggering exchange is high-novelty.
    """
    global _oracle_last_fired
    try:
        avg = get_average_vector()
        if avg is None:
            return

        # Fix 2 — per-dimension breach check
        breaching = _breaching_dimensions(avg)
        if not breaching:
            return

        # ── Cooldown gate ──────────────────────────────────────────────────────
        with _oracle_lock:
            now = time.monotonic()
            since_last = now - _oracle_last_fired
            if since_last < ORACLE_COOLDOWN_SECONDS:
                print(
                    f"[trajectory] breach={list(breaching.keys())} — Oracle on cooldown "
                    f"({ORACLE_COOLDOWN_SECONDS - since_last:.0f}s remaining)",
                    flush=True,
                )
                return
            _oracle_last_fired = now   # claim slot before releasing lock

        # ── Fix 5: novelty scoring ─────────────────────────────────────────────
        n_pairs = 20
        if human:
            try:
                from blackwell.novelty import novelty_score, oracle_multiplier, maybe_archive
                nov = novelty_score(human)
                # Compute scalar regret for the novelty helper
                from blackwell.evaluator import regret_from_scores, total_regret as _total_regret
                _scores = {d: avg[d] for d in avg}  # use avg as proxy for current exchange
                t_regret = sum(breaching.values())   # sum of gaps as scalar
                mult = oracle_multiplier(nov, t_regret)
                n_pairs = n_pairs * mult
                if mult > 1:
                    print(
                        f"[oracle] novelty={nov:.3f} — doubling synthesis pairs to {n_pairs}",
                        flush=True,
                    )
                # Archive high-novelty successes for LoRA replay
                if exchange_id and zephyr:
                    archived = maybe_archive(exchange_id, human, zephyr, nov, t_regret)
                    if archived:
                        print(
                            f"[oracle] novel anchor archived (novelty={nov:.3f}, "
                            f"regret={t_regret:.3f})",
                            flush=True,
                        )
            except Exception as e:
                print(f"[oracle] novelty scoring failed: {e}", flush=True)

        # ── Synthesis ──────────────────────────────────────────────────────────
        try:
            from blackwell.calculate_projection import project_onto_S, oracle_allocation
            breach_str = ", ".join(f"{d}={g:.3f}" for d, g in breaching.items())
            print(
                f"[trajectory] breach=[{breach_str}] — triggering Oracle "
                f"(n_pairs={n_pairs}, timeout={ORACLE_TIMEOUT_SECONDS}s)",
                flush=True,
            )
            projection = project_onto_S(avg)
            # Build steering vector focused on breaching dimensions only
            steering_v = {
                d: max(0.0, projection[d] - avg[d]) for d in avg
            }
            # oracle_allocation proportional to breach gaps (non-breaching dims get 0)
            focused_v = {d: breaching.get(d, 0.0) for d in avg}
            allocation = oracle_allocation(focused_v, n_pairs=n_pairs)

            # Write steering state for lora_steer to consume
            _write_steering_state(breaching, n_pairs)

            _call_synthesise_with_timeout(
                avg, steering_v, allocation,
                n_pairs=n_pairs, timeout=ORACLE_TIMEOUT_SECONDS,
            )
        except (ImportError, AttributeError) as e:
            print(f"[trajectory] Oracle import/attribute error: {e}", flush=True)

    except Exception as e:
        print(f"[trajectory] Oracle trigger failed: {e}", flush=True)


# ── BackgroundEvaluator ────────────────────────────────────────────────────────

class BackgroundEvaluator:
    """
    Daemon thread that scores exchange turns asynchronously.
    Call submit() from the main agent thread; scoring happens in background.
    """

    def __init__(self, regret_threshold: float = ORACLE_REGRET_THRESHOLD):
        self._q = queue.Queue()
        self._threshold = regret_threshold
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="bg-evaluator"
        )
        self._thread.start()

    def submit(self, exchange_id: str, human: str, zephyr: str) -> None:
        """Queue one exchange for background scoring. Never blocks."""
        self._q.put((exchange_id, human, zephyr))

    def _worker_step(self) -> None:
        """Process one queue item. Exposed for testing without threads."""
        exchange_id, human, zephyr = self._q.get(timeout=1)
        try:
            scores = evaluate_exchange(human, zephyr)
            update_scores(exchange_id, scores)
            # Pass human + zephyr so Oracle can compute novelty for this exchange
            _maybe_trigger_oracle(
                self._threshold,
                exchange_id=exchange_id,
                human=human,
                zephyr=zephyr,
            )
        except Exception as e:
            print(f"[bg-evaluator] scoring error for {exchange_id}: {e}", flush=True)

    def _run(self) -> None:
        while True:
            try:
                self._worker_step()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[bg-evaluator] unexpected loop error: {e}", flush=True)
                import time; time.sleep(1)
                continue


# ── Singleton ─────────────────────────────────────────────────────────────────

_evaluator: Optional[BackgroundEvaluator] = None
_evaluator_lock = threading.Lock()


def get_evaluator() -> "BackgroundEvaluator":
    """
    Return the singleton BackgroundEvaluator, creating it on first call.
    Double-checked locking prevents race on simultaneous first calls.
    """
    global _evaluator
    if _evaluator is None:
        with _evaluator_lock:
            if _evaluator is None:
                _evaluator = BackgroundEvaluator()
    return _evaluator

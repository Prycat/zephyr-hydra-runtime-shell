"""
blackwell/background_eval.py
Daemon thread that scores Zephyr conversation turns in the background.

After each real conversation turn, agent.py drops a work item here.
The evaluator calls evaluate_exchange() (LLM-as-judge), writes the
5-dimensional scores to blackwell.db, and checks whether total regret
has crossed the threshold — if so it triggers Oracle synthesis.
"""
import queue
import threading
from typing import Optional
from blackwell.evaluator import evaluate_exchange, total_regret
from blackwell.logger import update_scores, get_average_vector


def _maybe_trigger_oracle(threshold: float) -> None:
    """Trigger Oracle synthesis if regret exceeds threshold.

    1. Gets avg from get_average_vector().
    2. Computes total regret via total_regret(avg).
    3. If regret > threshold, computes steering vector and allocation
       using calculate_projection helpers, then calls synthesise().
    """
    try:
        avg = get_average_vector()
        if avg is None:
            return
        regret = total_regret(avg)
        if regret > threshold:
            try:
                from blackwell.oracle import synthesise
                from blackwell.calculate_projection import project_onto_S, oracle_allocation
                print(f"[trajectory] regret={regret:.3f} > {threshold} — triggering Oracle",
                      flush=True)
                projection = project_onto_S(avg)
                steering_v = {d: max(0.0, projection[d] - avg[d]) for d in avg}
                allocation = oracle_allocation(steering_v, n_pairs=20)
                synthesise(avg, steering_v, allocation, n_pairs=20)
            except (ImportError, AttributeError) as e:
                print(f"[trajectory] Oracle import/attribute error: {e}", flush=True)
    except Exception as e:
        print(f"[trajectory] Oracle trigger failed: {e}", flush=True)


class BackgroundEvaluator:
    """
    Daemon thread that scores exchange turns asynchronously.
    Call submit() from the main agent thread; scoring happens in background.
    """

    def __init__(self, regret_threshold: float = 0.15):
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
            _maybe_trigger_oracle(self._threshold)
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
                import time; time.sleep(1)  # backoff to avoid hot-spin
                continue


# Module-level singleton
_evaluator: Optional[BackgroundEvaluator] = None
_evaluator_lock = threading.Lock()


def get_evaluator() -> "BackgroundEvaluator":
    """Return the singleton BackgroundEvaluator, creating it on first call.

    Uses double-checked locking to avoid a race condition when multiple
    threads call get_evaluator() simultaneously before the instance exists.
    """
    global _evaluator
    if _evaluator is None:
        with _evaluator_lock:
            if _evaluator is None:
                _evaluator = BackgroundEvaluator()
    return _evaluator

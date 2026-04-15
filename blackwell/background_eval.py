"""
blackwell/background_eval.py
Daemon thread that scores Zephyr conversation turns in the background.
"""
import queue
import threading
import concurrent.futures
from typing import Optional
from blackwell.evaluator import evaluate_exchange, total_regret
from blackwell.logger import update_scores, get_average_vector

# Tuning constants — exported so tests can assert on them
ORACLE_REGRET_THRESHOLD = 0.25   # raised from 0.15 — reduces spurious Oracle triggers
ORACLE_TIMEOUT_SECONDS  = 8      # hard cap on synthesise(); fail fast, never block


def _call_synthesise_with_timeout(
    avg: dict, steering_v: dict, allocation: dict,
    n_pairs: int, timeout: float
) -> None:
    """Run synthesise() in a thread pool with a hard timeout. Logs and returns on timeout."""
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
        ex.shutdown(wait=False)  # don't block waiting for the thread to finish


def _maybe_trigger_oracle(threshold: float) -> None:
    """Trigger Oracle synthesis if regret exceeds threshold."""
    try:
        avg = get_average_vector()
        if avg is None:
            return
        regret = total_regret(avg)
        if regret > threshold:
            try:
                from blackwell.calculate_projection import project_onto_S, oracle_allocation
                print(f"[trajectory] regret={regret:.3f} > {threshold} — triggering Oracle",
                      flush=True)
                projection = project_onto_S(avg)
                steering_v = {d: max(0.0, projection[d] - avg[d]) for d in avg}
                allocation = oracle_allocation(steering_v, n_pairs=20)
                _call_synthesise_with_timeout(
                    avg, steering_v, allocation,
                    n_pairs=20, timeout=ORACLE_TIMEOUT_SECONDS,
                )
            except (ImportError, AttributeError) as e:
                print(f"[trajectory] Oracle import/attribute error: {e}", flush=True)
    except Exception as e:
        print(f"[trajectory] Oracle trigger failed: {e}", flush=True)


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
                import time; time.sleep(1)
                continue


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

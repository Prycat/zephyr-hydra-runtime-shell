"""
blackwell/trajectory.py
Trajectory logger — records every real Zephyr conversation turn as
ShareGPT-format JSONL for fine-tuning and RL training.

Three files:
  trajectory_samples.jsonl    — successful turns
  failed_trajectories.jsonl   — exceptions, bad tool calls, thumbs-down
  trajectory_feedback.jsonl   — explicit thumbs up/down signals
"""
import json
import datetime
import os
import threading

_HERE = os.path.join(os.path.dirname(__file__), "..")

SAMPLES_PATH  = os.path.join(_HERE, "trajectory_samples.jsonl")
FAILURES_PATH = os.path.join(_HERE, "failed_trajectories.jsonl")
FEEDBACK_PATH = os.path.join(_HERE, "trajectory_feedback.jsonl")

_VALID_ERROR_TYPES = {"exception", "bad_tool", "thumbs_down"}

_locks: dict[str, "threading.Lock"] = {}
_locks_mu = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    with _locks_mu:
        if path not in _locks:
            _locks[path] = threading.Lock()
        return _locks[path]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="microseconds")


def _append(path: str, obj: dict) -> None:
    """Thread-safe append of one JSON line. Swallows OSError with a warning."""
    with _get_lock(path):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as e:
            print(f"[trajectory] WARNING: could not write to {path}: {e}", flush=True)


def log_success(session_id: str, turn: int, user_msg: str,
                assistant_msg: str, tools_called: list[str]) -> None:
    """Log a successful conversation turn to trajectory_samples.jsonl."""
    _append(SAMPLES_PATH, {
        "conversations": [
            {"from": "human", "value": user_msg},
            {"from": "gpt",   "value": assistant_msg},
        ],
        "source":       "trajectory",
        "session_id":   session_id,
        "turn":         turn,
        "tools_called": tools_called or [],
        "timestamp":    _now(),
    })


def log_failure(session_id: str, turn: int, user_msg: str,
                error_type: str, detail: str) -> None:
    """
    Log a failed turn to failed_trajectories.jsonl.
    error_type: 'exception' | 'bad_tool' | 'thumbs_down'
    """
    if error_type not in _VALID_ERROR_TYPES:
        raise ValueError(f"Unknown error_type {error_type!r}. "
                         f"Must be one of {_VALID_ERROR_TYPES}")
    _append(FAILURES_PATH, {
        "conversations": [
            {"from": "human", "value": user_msg},
        ],
        "source":     "trajectory_failure",
        "error_type": error_type,
        "detail":     detail[:500],
        "session_id": session_id,
        "turn":       turn,
        "timestamp":  _now(),
    })


def mark_feedback(session_id: str, turn: int, positive: bool) -> None:
    """Record explicit thumbs up/down from the GUI."""
    _append(FEEDBACK_PATH, {
        "session_id": session_id,
        "turn":       turn,
        "positive":   positive,
        "timestamp":  _now(),
    })


def get_counts() -> dict:
    """Return line counts for all three JSONL files."""
    def _count(path):
        try:
            with open(path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0
    return {
        "success":  _count(SAMPLES_PATH),
        "failed":   _count(FAILURES_PATH),
        "feedback": _count(FEEDBACK_PATH),
    }

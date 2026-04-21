"""
blackwell/benchmark_runner.py

DB layer for the /code_benchmark command.  Stores per-benchmark scores and a
cycle counter in blackwell.db (SQLite).  Benchmark execution logic lives
elsewhere; this module is concerned only with persistence.

Tables
------
benchmark_scores : one row per evaluation run
benchmark_cycle  : single-row counter incremented on every save_score() call
"""
from __future__ import annotations
import json, os, re, sqlite3, subprocess, sys, tempfile, time, urllib.request

_HERE   = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", os.environ.get("_BM_DB_OVERRIDE", "blackwell.db"))

sys.path.insert(0, os.path.dirname(_HERE))
try:
    from config import OLLAMA_CHAT_URL as _OLLAMA_CHAT_URL
except ImportError:
    _OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"

STUDENT_MODEL  = "prycat:latest"
FALLBACK_MODEL = "hermes3:8b"
MODEL_TIMEOUT  = 45
CACHE_DIR = os.path.join(_HERE, "benchmark_cache")

BASELINES: dict[str, float] = {
    "cruxeval":      0.55,
    "livecodebench": 0.45,
    "swebench":      0.05,
}
BENCHMARK_NAMES = list(BASELINES.keys())


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Open a connection to DB_PATH with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_score_table() -> None:
    """
    Create benchmark_scores and benchmark_cycle tables if they do not exist.
    Also seeds the single benchmark_cycle row (id=1, cycle_count=0).
    """
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_scores (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark   TEXT    NOT NULL,
                score       REAL    NOT NULL,
                n_problems  INTEGER NOT NULL,
                n_correct   INTEGER NOT NULL,
                timestamp   TEXT    NOT NULL,
                model_name  TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_cycle (
                id          INTEGER PRIMARY KEY CHECK(id = 1),
                cycle_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO benchmark_cycle (id, cycle_count)
            VALUES (1, 0)
        """)
        conn.commit()


def save_score(
    benchmark: str,
    score: float,
    n_problems: int,
    n_correct: int,
    model_name: str = STUDENT_MODEL,
) -> None:
    """
    Persist an evaluation result and bump the global cycle counter.

    Parameters
    ----------
    benchmark  : benchmark identifier, e.g. "cruxeval"
    score      : fraction correct (0.0–1.0)
    n_problems : total problems attempted
    n_correct  : number answered correctly
    model_name : model that was evaluated (defaults to STUDENT_MODEL)
    """
    _ensure_score_table()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO benchmark_scores
                (benchmark, score, n_problems, n_correct, timestamp, model_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (benchmark, score, n_problems, n_correct, ts, model_name),
        )
        conn.execute(
            "UPDATE benchmark_cycle SET cycle_count = cycle_count + 1 WHERE id = 1"
        )
        conn.commit()


def get_last_scores() -> dict[str, dict | None]:
    """
    Return the most recent score record for every known benchmark.

    Returns
    -------
    dict mapping each benchmark name to a dict of its latest row fields,
    or None if that benchmark has never been run.
    """
    _ensure_score_table()
    result: dict[str, dict | None] = {name: None for name in BENCHMARK_NAMES}
    with _connect() as conn:
        for name in BENCHMARK_NAMES:
            row = conn.execute(
                """
                SELECT id, benchmark, score, n_problems, n_correct,
                       timestamp, model_name
                FROM   benchmark_scores
                WHERE  benchmark = ?
                ORDER  BY id DESC
                LIMIT  1
                """,
                (name,),
            ).fetchone()
            if row is not None:
                result[name] = dict(row)
    return result


def get_cycle_count() -> int:
    """Return the total number of save_score() calls recorded so far."""
    _ensure_score_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT cycle_count FROM benchmark_cycle WHERE id = 1"
        ).fetchone()
        return int(row["cycle_count"]) if row else 0


def get_score_history(benchmark: str, limit: int = 10) -> list[dict]:
    """
    Retrieve the last *limit* score records for a single benchmark.

    Parameters
    ----------
    benchmark : benchmark identifier
    limit     : maximum number of records to return (most recent first)

    Returns
    -------
    List of row dicts ordered most-recent first.
    """
    _ensure_score_table()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, benchmark, score, n_problems, n_correct,
                   timestamp, model_name
            FROM   benchmark_scores
            WHERE  benchmark = ?
            ORDER  BY id DESC
            LIMIT  ?
            """,
            (benchmark, limit),
        ).fetchall()
        return [dict(row) for row in rows]

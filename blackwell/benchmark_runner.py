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
    conn = _connect()
    try:
        with conn:
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
    finally:
        conn.close()


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
    if benchmark not in BASELINES:
        raise ValueError(f"Unknown benchmark: {benchmark!r}. Valid: {BENCHMARK_NAMES}")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = _connect()
    try:
        with conn:
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
    finally:
        conn.close()


def get_last_scores() -> dict[str, dict | None]:
    """
    Return the most recent score record for every known benchmark.

    Returns
    -------
    dict mapping each benchmark name to a dict of its latest row fields,
    or None if that benchmark has never been run.
    """
    result: dict[str, dict | None] = {name: None for name in BENCHMARK_NAMES}
    conn = _connect()
    try:
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
    finally:
        conn.close()
    return result


def get_cycle_count() -> int:
    """Return the total number of save_score() calls recorded so far."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT cycle_count FROM benchmark_cycle WHERE id = 1"
        ).fetchone()
        return int(row["cycle_count"]) if row else 0
    finally:
        conn.close()


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
    conn = _connect()
    try:
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
    finally:
        conn.close()


# ── Cycle selection ───────────────────────────────────────────────────────────

def select_next_benchmark() -> str:
    """
    Pick the next benchmark to run.

    Logic:
      1. If cycle_count > 0 AND cycle_count % 3 == 0 AND swebench has never run
         → return "swebench"
      2. If ALL scores are None (no benchmark ever run) → return "cruxeval"
      3. Otherwise: find benchmark in ["cruxeval", "livecodebench"] with the
         lowest gap (gap = last_score - baseline; negative = underperforming).
         Never-run benchmarks get gap = -1.0 (highest priority).
         Return the benchmark with the lowest gap.
    """
    cycle_count = get_cycle_count()
    scores = get_last_scores()

    # Rule 1: periodic swebench trigger
    if cycle_count > 0 and cycle_count % 3 == 0 and scores["swebench"] is None:
        return "swebench"

    # Rule 2: no benchmark has ever run → start with cruxeval
    if all(v is None for v in scores.values()):
        return "cruxeval"

    # Rule 3: pick the weakest among cruxeval and livecodebench
    candidates = ["cruxeval", "livecodebench"]

    def gap(name: str) -> float:
        record = scores[name]
        if record is None:
            return -1.0  # never run → highest priority
        return record["score"] - BASELINES[name]

    return min(candidates, key=gap)


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_model(prompt: str, model: str = STUDENT_MODEL,
                system: str = "") -> str:
    """Call model via Ollama, return response text. Falls back to hermes3:8b."""
    import httpx
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for m in (model, FALLBACK_MODEL):
        try:
            resp = httpx.post(
                _OLLAMA_CHAT_URL,
                json={"model": m, "messages": messages,
                      "temperature": 0.0, "max_tokens": 256},
                timeout=MODEL_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return "(model unavailable)"


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _fetch_url(url: str, cache_name: str) -> str:
    """Download URL to local cache dir, return local path. Skips if cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(local):
        return local
    print(f"[benchmark] Downloading {cache_name}...", flush=True)
    urllib.request.urlretrieve(url, local)
    print(f"[benchmark] Cached to {local}", flush=True)
    return local


# ── CRUXEval ──────────────────────────────────────────────────────────────────

CRUXEVAL_URL = (
    "https://raw.githubusercontent.com/facebookresearch/"
    "CRUXEval/main/data/cruxeval.jsonl"
)


def _load_cruxeval(n: int = 50) -> list[dict]:
    """Load n CRUXEval output-prediction problems from GitHub cache."""
    path = _fetch_url(CRUXEVAL_URL, "cruxeval.jsonl")
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems[:n]


def run_cruxeval(n: int = 50) -> dict:
    """
    Run CRUXEval output-prediction on n problems.
    Prompt: given function + call, predict return value.
    Score: exact match after stripping whitespace and surrounding quotes.
    """
    problems = _load_cruxeval(n)
    actual_n = len(problems)
    correct = 0
    print(f"\n[benchmark] CRUXEval — {actual_n} output-prediction problems", flush=True)
    print(f"[benchmark] Model: {STUDENT_MODEL} (fallback: {FALLBACK_MODEL})\n", flush=True)

    for i, p in enumerate(problems, 1):
        prompt = (
            f"Given this Python function:\n\n{p['code']}\n\n"
            f"What is the return value of: {p['input']}\n\n"
            "Reply with ONLY the return value — no explanation, no code."
        )
        answer = _call_model(prompt)
        expected = p["output"].strip()
        got = answer.strip().strip("'\"")
        exp = expected.strip().strip("'\"")
        if got == exp:
            correct += 1
        if i % 10 == 0 or i == actual_n:
            print(f"  [{i:>3}/{actual_n}]  correct so far: {correct}/{i}  "
                  f"({100*correct/i:.1f}%)", flush=True)

    score = correct / actual_n if actual_n else 0.0
    baseline = BASELINES["cruxeval"]
    delta = score - baseline
    print(f"\n[benchmark] CRUXEval result: {correct}/{actual_n} = {score:.3f}  "
          f"(baseline {baseline:.2f}  Δ{delta:+.3f})", flush=True)
    save_score("cruxeval", score, actual_n, correct)
    return {"benchmark": "cruxeval", "score": score,
            "n_problems": actual_n, "n_correct": correct}


# Run once on import so the schema is always ready before any function is called.
_ensure_score_table()

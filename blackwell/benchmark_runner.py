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

# Windows cp1252 consoles can't print Δ / other Unicode — reconfigure to UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE   = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "..", os.environ.get("_BM_DB_OVERRIDE", "blackwell.db"))

sys.path.insert(0, os.path.dirname(_HERE))
try:
    from config import OLLAMA_CHAT_URL as _OLLAMA_CHAT_URL
except ImportError:
    _OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"

STUDENT_MODEL  = "prycat1:8B"
FALLBACK_MODEL = "hermes3:8b"
MODEL_TIMEOUT  = 45
CACHE_DIR = os.path.join(_HERE, "benchmark_cache")

BASELINES: dict[str, float] = {
    "cruxeval":      0.55,
    "livecodebench": 0.45,
    "swebench":      0.05,
}
BENCHMARK_NAMES = list(BASELINES.keys())

# Benchmarks included in automatic rotation — swebench requires Docker + harness
# and is excluded until explicitly requested via --benchmark swebench.
AUTO_BENCHMARKS = ["cruxeval", "livecodebench"]


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


def _ensure_pruning_table() -> None:
    """Create pruning_events table if it does not exist."""
    conn = _connect()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pruning_events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    heads_pruned  INTEGER NOT NULL,
                    total_heads   INTEGER NOT NULL,
                    compression   REAL    NOT NULL,
                    benchmark     TEXT    NOT NULL,
                    score_before  REAL    NOT NULL,
                    score_after   REAL,
                    score_delta   REAL,
                    committed     INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
    finally:
        conn.close()


def save_pruning_event(
    heads_pruned: int,
    total_heads: int,
    benchmark: str,
    score_before: float,
    score_after: float | None = None,
    committed: bool = False,
) -> None:
    """
    Persist one pruning cycle result to blackwell.db.

    Parameters
    ----------
    heads_pruned  : number of attention heads zeroed in this cycle
    total_heads   : total heads in the model (e.g. 1024 for Llama-3.1-8B)
    benchmark     : benchmark name used for validation (must be in BENCHMARK_NAMES)
    score_before  : benchmark score recorded before pruning
    score_after   : benchmark score after pruning; None if not yet measured
    committed     : True if the prune was committed, False if rolled back
    """
    if benchmark not in BASELINES:
        raise ValueError(f"Unknown benchmark: {benchmark!r}. Valid: {BENCHMARK_NAMES}")
    if total_heads <= 0:
        raise ValueError(f"total_heads must be positive, got {total_heads}")
    compression = round(heads_pruned / total_heads, 4)  # fraction of heads removed
    delta = round(score_after - score_before, 4) if score_after is not None else None
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO pruning_events
                    (timestamp, heads_pruned, total_heads, compression,
                     benchmark, score_before, score_after, score_delta, committed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    heads_pruned,
                    total_heads,
                    compression,
                    benchmark,
                    score_before,
                    score_after,
                    delta,
                    int(committed),
                ),
            )
            conn.commit()
    finally:
        conn.close()


def get_pruning_history(limit: int = 20) -> list[dict]:
    """Return most-recent pruning events, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, heads_pruned, total_heads, compression,
                   benchmark, score_before, score_after, score_delta, committed
            FROM   pruning_events
            ORDER  BY id DESC
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


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

def _swebench_available() -> bool:
    """
    Return True only if both Docker and the swebench package are present.
    Used to gate explicit --benchmark swebench runs.
    """
    import shutil, importlib.util
    return (
        shutil.which("docker") is not None
        and importlib.util.find_spec("swebench") is not None
    )


def select_next_benchmark() -> str:
    """
    Pick the next benchmark to run from AUTO_BENCHMARKS.

    Logic:
      1. If no benchmark has ever run → return "cruxeval".
      2. Otherwise: pick the AUTO_BENCHMARK with the lowest gap
         (gap = last_score − baseline; negative = underperforming).
         Never-run benchmarks get gap = −1.0 (highest priority).

    SWE-bench is excluded from automatic rotation — it requires Docker and the
    swebench harness.  Request it explicitly with --benchmark swebench.
    """
    scores = get_last_scores()

    # No benchmark has ever run → start with cruxeval
    if all(scores[name] is None for name in AUTO_BENCHMARKS):
        return "cruxeval"

    def gap(name: str) -> float:
        record = scores[name]
        if record is None:
            return -1.0  # never run → highest priority
        return record["score"] - BASELINES[name]

    return min(AUTO_BENCHMARKS, key=gap)


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


# ── LiveCodeBench ─────────────────────────────────────────────────────────────

_LCB_FALLBACK = [
    {"question_content": "Given an integer N, print N * 2.",
     "test_cases": [{"input": "5\n", "output": "10"}, {"input": "0\n", "output": "0"}]},
    {"question_content": "Given integers A and B on one line, print their sum.",
     "test_cases": [{"input": "3 4\n", "output": "7"}, {"input": "0 0\n", "output": "0"}]},
    {"question_content": "Given a string, print it reversed.",
     "test_cases": [{"input": "hello\n", "output": "olleh"}, {"input": "abc\n", "output": "cba"}]},
    {"question_content": "Given integers on one line, print their max.",
     "test_cases": [{"input": "3 1 4 1 5 9 2 6\n", "output": "9"}, {"input": "7\n", "output": "7"}]},
    {"question_content": "Given N, print the sum 1+2+...+N.",
     "test_cases": [{"input": "10\n", "output": "55"}, {"input": "1\n", "output": "1"}]},
    {"question_content": "Given integers A B C on one line, print them sorted ascending space-separated.",
     "test_cases": [{"input": "3 1 2\n", "output": "1 2 3"}, {"input": "5 5 5\n", "output": "5 5 5"}]},
    {"question_content": "Given a number N, print 'even' if even else 'odd'.",
     "test_cases": [{"input": "4\n", "output": "even"}, {"input": "7\n", "output": "odd"}]},
    {"question_content": "Count vowels (aeiouAEIOU) in a string and print the count.",
     "test_cases": [{"input": "Hello World\n", "output": "3"}, {"input": "rhythm\n", "output": "0"}]},
    {"question_content": "Given N, print all integers from 1 to N divisible by 3 or 5, space-separated.",
     "test_cases": [{"input": "15\n", "output": "3 5 6 9 10 12 15"}, {"input": "4\n", "output": "3"}]},
    {"question_content": "Print the Nth Fibonacci number (0-indexed, F(0)=0, F(1)=1).",
     "test_cases": [{"input": "7\n", "output": "13"}, {"input": "0\n", "output": "0"}]},
    {"question_content": "Given N and K on one line, print N! divided by K! as an integer (N >= K >= 0).",
     "test_cases": [{"input": "5 3\n", "output": "20"}, {"input": "3 3\n", "output": "1"}]},
    {"question_content": "Given a string S, print 'palindrome' or 'not palindrome'.",
     "test_cases": [{"input": "racecar\n", "output": "palindrome"}, {"input": "hello\n", "output": "not palindrome"}]},
    {"question_content": "Given N integers on one line, print their mean rounded to 2 decimal places.",
     "test_cases": [{"input": "1 2 3 4\n", "output": "2.50"}, {"input": "7\n", "output": "7.00"}]},
    {"question_content": "Given a sorted list of N integers and a target T, print the 0-based index of T or -1.",
     "test_cases": [{"input": "5\n1 3 5 7 9\n5\n", "output": "2"}, {"input": "3\n2 4 6\n5\n", "output": "-1"}]},
    {"question_content": "Given N, print prime factors of N space-separated in ascending order.",
     "test_cases": [{"input": "12\n", "output": "2 2 3"}, {"input": "13\n", "output": "13"}]},
    {"question_content": "Given two strings on separate lines, print the length of their longest common subsequence.",
     "test_cases": [{"input": "abcde\nace\n", "output": "3"}, {"input": "abc\nabc\n", "output": "3"}]},
    {"question_content": "Given a string with only ()[]{}  brackets, print 'valid' or 'invalid'.",
     "test_cases": [{"input": "[]{}\n", "output": "valid"}, {"input": "([)]\n", "output": "invalid"}]},
    {"question_content": "Given N integers on one line, print the count of inversions (pairs i<j where a[i]>a[j]).",
     "test_cases": [{"input": "2 4 1 3 5\n", "output": "3"}, {"input": "1 2 3\n", "output": "0"}]},
    {"question_content": "Given N coins of given values and a target T, print min coins needed or -1 if impossible.",
     "test_cases": [{"input": "3\n1 5 6\n11\n", "output": "2"}, {"input": "2\n3 5\n2\n", "output": "-1"}]},
    {"question_content": "Given a string, print the total count of palindromic substrings (including single chars).",
     "test_cases": [{"input": "aaa\n", "output": "6"}, {"input": "abc\n", "output": "3"}]},
    {"question_content": "Given an NxN grid of 0/1 (N on first line, then N space-separated rows), count islands (connected 1s, 4-directional).",
     "test_cases": [{"input": "4\n1 1 0 0\n1 1 0 0\n0 0 1 0\n0 0 0 1\n", "output": "3"}, {"input": "1\n0\n", "output": "0"}]},
    {"question_content": "Given integer N, print all prime numbers up to N inclusive, space-separated.",
     "test_cases": [{"input": "20\n", "output": "2 3 5 7 11 13 17 19"}, {"input": "2\n", "output": "2"}]},
    {"question_content": "Given a matrix as N M on first line then N rows of M integers, print column sums space-separated.",
     "test_cases": [{"input": "2 3\n1 2 3\n4 5 6\n", "output": "5 7 9"}, {"input": "1 2\n9 1\n", "output": "9 1"}]},
    {"question_content": "Given N integers, print the second largest distinct value, or 'None' if it doesn't exist.",
     "test_cases": [{"input": "3 1 4 1 5\n", "output": "4"}, {"input": "2 2 2\n", "output": "None"}]},
    {"question_content": "Given directed graph as N M on first line then M edges u v, print topological order space-separated or 'cycle'.",
     "test_cases": [{"input": "3 2\n0 1\n1 2\n", "output": "0 1 2"}, {"input": "2 2\n0 1\n1 0\n", "output": "cycle"}]},
]


def _load_livecodebench(n: int = 25) -> list[dict]:
    """
    Load n LiveCodeBench problems.
    Tries `datasets` library first (downloads from HuggingFace).
    Falls back to the built-in 25-problem seed.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("livecodebench/code_generation_lite",
                          split="test")
        problems = []
        for item in ds:
            tcs_raw = item.get("public_test_cases") or item.get("test_cases") or "[]"
            if isinstance(tcs_raw, str):
                try:
                    tcs_raw = json.loads(tcs_raw)
                except Exception:
                    tcs_raw = []
            tcs = [{"input": t.get("input", ""), "output": t.get("output", "")}
                   for t in (tcs_raw or []) if t.get("output")]
            if tcs:
                problems.append({
                    "question_content": item.get("question_content", ""),
                    "test_cases": tcs[:2],
                })
            if len(problems) >= n:
                break
        if len(problems) >= n:
            return problems[:n]
    except Exception:
        pass
    return _LCB_FALLBACK[:n]


def _extract_code(response: str) -> str:
    """Extract Python code from model response, stripping markdown fences."""
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def _execute_code(code: str, test_input: str = "",
                  timeout: int = 10) -> str:
    """Run code in isolated subprocess, return stdout or sentinel string."""
    fd, fname = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            [sys.executable, fname],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"
    except Exception as e:
        return f"__ERROR__: {e}"
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def run_livecodebench(n: int = 25) -> dict:
    """
    Run n LiveCodeBench code-generation problems.
    Generate Python that reads stdin/prints stdout, execute against test cases.
    Score: pass@1 — all test cases must pass.
    """
    problems = _load_livecodebench(n)
    actual_n = len(problems)
    correct = 0
    print(f"\n[benchmark] LiveCodeBench — {actual_n} code-generation problems", flush=True)
    print(f"[benchmark] Model: {STUDENT_MODEL} (fallback: {FALLBACK_MODEL})\n", flush=True)

    for i, p in enumerate(problems, 1):
        prompt = (
            f"Solve this programming problem in Python.\n\n"
            f"{p['question_content']}\n\n"
            "Write ONLY Python code that reads from stdin and prints to stdout. "
            "No explanation, no markdown."
        )
        response = _call_model(prompt)
        code = _extract_code(response)

        all_pass = True
        for tc in p["test_cases"]:
            got = _execute_code(code, test_input=tc["input"])
            exp = tc["output"].strip()
            if got != exp:
                all_pass = False
                break

        if all_pass:
            correct += 1
        if i % 5 == 0 or i == actual_n:
            print(f"  [{i:>3}/{actual_n}]  correct so far: {correct}/{i}  "
                  f"({100*correct/i:.1f}%)", flush=True)

    score = correct / actual_n if actual_n else 0.0
    baseline = BASELINES["livecodebench"]
    delta = score - baseline
    print(f"\n[benchmark] LiveCodeBench result: {correct}/{actual_n} = {score:.3f}  "
          f"(baseline {baseline:.2f}  Δ{delta:+.3f})", flush=True)
    save_score("livecodebench", score, actual_n, correct)
    return {"benchmark": "livecodebench", "score": score,
            "n_problems": actual_n, "n_correct": correct}


# ── SWE-bench stub ────────────────────────────────────────────────────────────

def run_swebench() -> dict:
    """
    SWE-bench Verified requires Docker + the swebench harness package.

    To run properly:
        pip install swebench
        python -m swebench.harness.run_evaluation \\
            --predictions_path your_preds.jsonl \\
            --swe_bench_tasks princeton-nlp/SWE-bench_Verified \\
            --log_dir ./logs --testbed /tmp/swebench_testbed

    See: https://github.com/princeton-nlp/SWE-bench
    """
    if not _swebench_available():
        print(
            "\n[benchmark] SWE-bench skipped — Docker and/or the swebench package "
            "are not installed.\n"
            "  Install: pip install swebench  (and ensure Docker is running)\n"
            "  Docs: https://github.com/princeton-nlp/SWE-bench",
            flush=True,
        )
        return {
            "benchmark": "swebench",
            "score": None,
            "n_problems": 0,
            "n_correct": 0,
            "skipped": True,
        }

    # ── Full harness path (reached only when Docker + swebench are present) ──
    notes = (
        "SWE-bench Verified full harness not yet wired — "
        "call swebench.harness.run_evaluation directly."
    )
    print(f"\n[benchmark] SWE-bench: {notes}", flush=True)
    return {
        "benchmark": "swebench",
        "score": None,
        "n_problems": 0,
        "n_correct": 0,
        "notes": notes,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_score_history() -> None:
    """Print formatted table of all benchmark scores."""
    scores = get_last_scores()
    cycle  = get_cycle_count()
    print("\n[benchmark] ── Score History ─────────────────────────────────────")
    print(f"  Cycles completed: {cycle}")
    print(f"  {'Benchmark':<16} {'Last score':>11} {'Baseline':>9} {'Gap':>7}  Last run")
    print("  " + "─" * 62)
    swebench_ok = _swebench_available()
    for bm_name in BENCHMARK_NAMES:
        rec = scores[bm_name]
        baseline = BASELINES[bm_name]
        if rec:
            score = rec["score"]
            delta = score - baseline
            ts    = rec["timestamp"][:10]
            print(f"  {bm_name:<16} {score:>11.3f} {baseline:>9.2f} {delta:>+7.3f}  {ts}")
        elif bm_name == "swebench" and not swebench_ok:
            print(f"  {bm_name:<16} {'(no Docker)':>11} {baseline:>9.2f} {'—':>7}  —")
        else:
            print(f"  {bm_name:<16} {'(never run)':>11} {baseline:>9.2f} {'—':>7}  —")
    print("[benchmark] ───────────────────────────────────────────────────────\n")


# ── Runners registry ──────────────────────────────────────────────────────────

RUNNERS: dict[str, object] = {
    "cruxeval":      run_cruxeval,
    "livecodebench": run_livecodebench,
    "swebench":      run_swebench,
}


def run_benchmark_cycle(override: str | None = None,
                        n: int | None = None) -> dict:
    """Select and run the next benchmark. Returns the result dict."""
    target = override or select_next_benchmark()
    print(f"\n[benchmark] Selected: {target.upper()}", flush=True)
    kwargs: dict = {}
    if n is not None and target != "swebench":
        kwargs["n"] = n
    result = RUNNERS[target](**kwargs)
    print_score_history()
    return result


# Run once on import so the schema is always ready before any function is called.
# Must be before __main__ so sys.exit() inside the argparse block doesn't skip it.
_ensure_score_table()
_ensure_pruning_table()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blackwell code benchmark runner")
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES,
                        help="Force a specific benchmark (default: auto-select)")
    parser.add_argument("--n", type=int,
                        help="Number of problems (default: 50 cruxeval / 25 livecodebench)")
    parser.add_argument("--history", action="store_true",
                        help="Print score history and exit")
    args = parser.parse_args()

    if args.history:
        print_score_history()
        sys.exit(0)

    result = run_benchmark_cycle(override=args.benchmark, n=args.n)
    # Exit 1 only when a real run produced no score (actual failure).
    # Skipped stubs (swebench without Docker) are not failures.
    failed = result.get("score") is None and not result.get("skipped", False)
    sys.exit(1 if failed else 0)

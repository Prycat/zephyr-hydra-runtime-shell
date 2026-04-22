import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def setup_module(module):
    import tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    import blackwell.benchmark_runner as bm_mod
    importlib.reload(bm_mod)
    module.bm = bm_mod

def test_score_table_created():
    import sqlite3
    bm.save_score("cruxeval", 0.60, 50, 30)
    conn = sqlite3.connect(os.environ["_BM_DB_OVERRIDE"])
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "benchmark_scores" in tables
    assert "benchmark_cycle" in tables

def test_save_and_load_scores():
    bm.save_score("cruxeval", 0.62, 50, 31)
    scores = bm.get_last_scores()
    assert "cruxeval" in scores
    assert abs(scores["cruxeval"]["score"] - 0.62) < 0.001

def test_last_score_is_most_recent():
    bm.save_score("cruxeval", 0.70, 50, 35)
    scores = bm.get_last_scores()
    assert abs(scores["cruxeval"]["score"] - 0.70) < 0.001

def test_never_run_benchmark_is_none():
    scores = bm.get_last_scores()
    assert scores["livecodebench"] is None
    assert scores["swebench"] is None

def test_cycle_count_increments():
    initial = bm.get_cycle_count()
    bm.save_score("cruxeval", 0.55, 50, 28)
    assert bm.get_cycle_count() == initial + 1

def test_score_history_limit():
    for i in range(5):
        bm.save_score("livecodebench", 0.40 + i*0.01, 25, 10+i)
    history = bm.get_score_history("livecodebench", limit=3)
    assert len(history) == 3
    # Most recent first
    assert history[0]["score"] >= history[1]["score"]


def test_select_cruxeval_when_no_scores():
    import importlib, tempfile
    # Fresh isolated DB — no scores
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    importlib.reload(bm)
    result = bm.select_next_benchmark()
    assert result == "cruxeval"

def test_select_weakest_benchmark():
    import importlib, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    importlib.reload(bm)
    bm.save_score("cruxeval", 0.70, 50, 35)       # +0.15 above baseline
    bm.save_score("livecodebench", 0.30, 25, 8)   # -0.15 below baseline
    result = bm.select_next_benchmark()
    assert result == "livecodebench"

def test_swebench_every_third_cycle():
    import importlib, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    importlib.reload(bm)
    # 3 saves = cycle_count becomes 3, swebench never run → trigger
    bm.save_score("cruxeval", 0.60, 50, 30)
    bm.save_score("cruxeval", 0.61, 50, 31)
    bm.save_score("cruxeval", 0.62, 50, 32)
    result = bm.select_next_benchmark()
    assert result == "swebench"

def test_swebench_not_triggered_on_cycle_zero():
    import importlib, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    importlib.reload(bm)
    # cycle_count is 0 → swebench should NOT be selected even though never run
    result = bm.select_next_benchmark()
    assert result != "swebench"

def test_call_model_returns_string():
    result = bm._call_model("What is 2+2?")
    assert isinstance(result, str)
    assert len(result) > 0

def test_load_cruxeval_returns_list():
    problems = bm._load_cruxeval(n=5)
    assert isinstance(problems, list)
    assert len(problems) == 5
    assert "code" in problems[0]
    assert "input" in problems[0]
    assert "output" in problems[0]

def test_load_cruxeval_caches():
    import os
    bm._load_cruxeval(n=3)
    cache_file = os.path.join(bm.CACHE_DIR, "cruxeval.jsonl")
    assert os.path.exists(cache_file)

def test_load_livecodebench_returns_list():
    problems = bm._load_livecodebench(n=5)
    assert isinstance(problems, list)
    assert len(problems) == 5
    assert "question_content" in problems[0]
    assert "test_cases" in problems[0]

def test_execute_code_correct():
    code = "print(int(input()) * 2)"
    result = bm._execute_code(code, test_input="5\n")
    assert result == "10"

def test_execute_code_timeout():
    code = "while True: pass"
    result = bm._execute_code(code, timeout=1)
    assert result == "__TIMEOUT__"

def test_extract_code_strips_fences():
    response = "```python\nprint('hi')\n```"
    assert bm._extract_code(response) == "print('hi')"

def test_extract_code_bare():
    response = "print('hi')"
    assert bm._extract_code(response) == "print('hi')"

def test_swebench_returns_stub():
    result = bm.run_swebench()
    assert result["benchmark"] == "swebench"
    assert result["score"] is None
    assert "docker" in result["notes"].lower()

def test_print_score_history_runs():
    import importlib, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    importlib.reload(bm)
    bm.print_score_history()  # should not raise

def test_run_benchmark_cycle_override():
    import importlib, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    importlib.reload(bm)
    result = bm.run_benchmark_cycle(override="swebench")
    assert result["benchmark"] == "swebench"


# ── RDSP pruning_events tests ─────────────────────────────────────────────────

def test_pruning_table_created():
    """pruning_events table exists after module load."""
    import importlib, os, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    import blackwell.benchmark_runner as bm
    importlib.reload(bm)
    conn = bm._connect()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pruning_events'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1, "pruning_events table not created"


def test_save_and_load_pruning_event():
    """save_pruning_event() persists a row; get_pruning_history() retrieves it."""
    import importlib, os, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    import blackwell.benchmark_runner as bm
    importlib.reload(bm)
    bm.save_pruning_event(
        heads_pruned=48,
        total_heads=1024,
        benchmark="cruxeval",
        score_before=0.50,
        score_after=0.49,
        committed=True,
    )
    history = bm.get_pruning_history()
    assert len(history) == 1
    row = history[0]
    assert row["heads_pruned"] == 48
    assert row["committed"] == 1
    assert abs(row["score_before"] - 0.50) < 1e-6


def test_pruning_event_committed_false():
    """committed=False round-trips correctly."""
    import importlib, os, tempfile
    os.environ["_BM_DB_OVERRIDE"] = tempfile.mktemp(suffix=".db")
    import blackwell.benchmark_runner as bm
    importlib.reload(bm)
    bm.save_pruning_event(
        heads_pruned=32,
        total_heads=1024,
        benchmark="livecodebench",
        score_before=0.48,
        score_after=0.35,
        committed=False,
    )
    history = bm.get_pruning_history()
    assert history[0]["committed"] == 0

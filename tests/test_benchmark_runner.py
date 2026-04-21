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

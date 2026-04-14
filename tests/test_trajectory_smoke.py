import os, json, pytest
import blackwell.trajectory as T

@pytest.fixture(autouse=True)
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "SAMPLES_PATH",  str(tmp_path / "trajectory_samples.jsonl"))
    monkeypatch.setattr(T, "FAILURES_PATH", str(tmp_path / "failed_trajectories.jsonl"))
    monkeypatch.setattr(T, "FEEDBACK_PATH", str(tmp_path / "trajectory_feedback.jsonl"))

def test_log_success_writes_jsonl():
    T.log_success("sess1", 1, "hello", "hi there", [])
    lines = open(T.SAMPLES_PATH).readlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["source"] == "trajectory"
    assert obj["conversations"][0]["from"] == "human"
    assert obj["conversations"][1]["from"] == "gpt"

def test_log_failure_writes_jsonl():
    T.log_failure("sess1", 2, "bad query", "exception", "IndexError")
    lines = open(T.FAILURES_PATH).readlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["error_type"] == "exception"
    assert obj["source"] == "trajectory_failure"

def test_mark_feedback_writes_jsonl():
    T.mark_feedback("sess1", 1, positive=True)
    T.mark_feedback("sess1", 2, positive=False)
    lines = open(T.FEEDBACK_PATH).readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["positive"] is True
    assert json.loads(lines[1])["positive"] is False

def test_get_counts():
    T.log_success("s", 1, "q", "a", [])
    T.log_success("s", 2, "q2", "a2", ["web_search"])
    T.log_failure("s", 3, "q3", "bad_tool", "wrong args")
    T.mark_feedback("s", 1, positive=False)
    counts = T.get_counts()
    assert counts["success"] == 2
    assert counts["failed"] == 1
    assert counts["feedback"] == 1

def test_full_smoke_five_turns(tmp_path, monkeypatch):
    """Fire 5 fake turns: 2 success, 2 failure, 1 feedback."""
    import blackwell.trajectory as T2
    monkeypatch.setattr(T2, "SAMPLES_PATH",  str(tmp_path / "s.jsonl"))
    monkeypatch.setattr(T2, "FAILURES_PATH", str(tmp_path / "f.jsonl"))
    monkeypatch.setattr(T2, "FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))

    T2.log_success("s", 1, "hello", "hi", [])
    T2.log_success("s", 2, "search?", "found it", ["web_search"])
    T2.log_failure("s", 3, "crash q", "exception", "ZeroDivisionError")
    T2.log_failure("s", 4, "bad q", "bad_tool", "wrong params")
    T2.mark_feedback("s", 5, positive=False)

    counts = T2.get_counts()
    assert counts["success"]  == 2
    assert counts["failed"]   == 2
    assert counts["feedback"] == 1

    sample = json.loads(open(T2.SAMPLES_PATH).readlines()[0])
    assert "conversations" in sample
    assert sample["conversations"][0]["from"] == "human"
    assert sample["conversations"][1]["from"] == "gpt"
    assert sample["source"] == "trajectory"

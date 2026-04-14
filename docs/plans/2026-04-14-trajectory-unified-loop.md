# Trajectory Unified Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the Blackwell self-improvement loop — every real conversation turn logs trajectory data, passively scores against the regret vector, provides GUI thumbs feedback, and auto-exports fine-tuned adapters to Ollama.

**Architecture:** Three new files (`blackwell/trajectory.py`, `blackwell/background_eval.py`, `blackwell/export.py`) wired into the existing agent loop and GUI. `lora_steer.py` gains a GGUF export step. GUI gains thumbs buttons and pair-count badges.

**Tech Stack:** Python 3.10+, PySide6, SQLite (existing blackwell.db), unsloth (existing), Ollama REST API, pytest

---

### Task 1: TrajectoryLogger (`blackwell/trajectory.py`)

**Files:**
- Create: `blackwell/trajectory.py`
- Create: `tests/test_trajectory_smoke.py`

**Step 1: Write the failing test**

```python
# tests/test_trajectory_smoke.py
import os, json, tempfile, pytest
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
```

**Step 2: Run test to verify it fails**

```bash
cd C:\Users\gamer23\Desktop\hermes-agent
pytest tests/test_trajectory_smoke.py -v
```
Expected: 4 failures — `blackwell/trajectory.py` does not exist yet.

**Step 3: Implement `blackwell/trajectory.py`**

```python
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

_HERE = os.path.join(os.path.dirname(__file__), "..")

SAMPLES_PATH  = os.path.join(_HERE, "trajectory_samples.jsonl")
FAILURES_PATH = os.path.join(_HERE, "failed_trajectories.jsonl")
FEEDBACK_PATH = os.path.join(_HERE, "trajectory_feedback.jsonl")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="microseconds")


def _append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_success(session_id: str, turn: int, user_msg: str,
                assistant_msg: str, tools_called: list) -> None:
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trajectory_smoke.py -v
```
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add blackwell/trajectory.py tests/test_trajectory_smoke.py
git commit -m "feat: add TrajectoryLogger — log_success, log_failure, mark_feedback"
```

---

### Task 2: BackgroundEvaluator (`blackwell/background_eval.py`)

**Files:**
- Create: `blackwell/background_eval.py`
- Modify: `tests/test_trajectory_smoke.py` — add evaluator tests

**Step 1: Write the failing tests (append to existing test file)**

```python
# append to tests/test_trajectory_smoke.py
from unittest.mock import patch, MagicMock
from blackwell.background_eval import BackgroundEvaluator

def test_evaluator_scores_and_writes_to_db():
    """Evaluator processes queue item and calls update_scores."""
    fake_scores = {
        "accuracy": 0.8, "logic": 0.9, "tone": 0.7,
        "curiosity": 0.6, "safety": 1.0, "notes": "good"
    }
    with patch("blackwell.background_eval.evaluate_exchange",
               return_value=fake_scores) as mock_eval, \
         patch("blackwell.background_eval.update_scores") as mock_update:
        ev = BackgroundEvaluator(regret_threshold=0.15)
        ev.submit("ex-id-1", "what is pi?", "3.14159")
        ev._worker_step()   # process one item synchronously for testing
        mock_eval.assert_called_once_with("what is pi?", "3.14159")
        mock_update.assert_called_once_with("ex-id-1", fake_scores)

def test_evaluator_does_not_crash_on_error():
    with patch("blackwell.background_eval.evaluate_exchange",
               side_effect=Exception("ollama down")):
        ev = BackgroundEvaluator()
        ev.submit("ex-id-2", "hello", "hi")
        ev._worker_step()  # should swallow error, not raise
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_trajectory_smoke.py::test_evaluator_scores_and_writes_to_db -v
```
Expected: FAIL — `BackgroundEvaluator` not defined.

**Step 3: Implement `blackwell/background_eval.py`**

```python
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
from blackwell.evaluator import evaluate_exchange, total_regret
from blackwell.logger import update_scores, get_average_vector

# Lazy import to avoid circular dependency at module level
def _maybe_trigger_oracle(threshold: float) -> None:
    """Trigger Oracle synthesis if regret exceeds threshold."""
    try:
        avg = get_average_vector()
        if avg is None:
            return
        regret = total_regret(avg)
        if regret > threshold:
            from blackwell.oracle import generate_pairs
            print(f"[trajectory] regret={regret:.3f} > {threshold} — triggering Oracle",
                  flush=True)
            generate_pairs(n_pairs=20)
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
            except Exception:
                continue


# Module-level singleton — created once when agent.py imports this module
_evaluator: BackgroundEvaluator | None = None


def get_evaluator() -> BackgroundEvaluator:
    """Return the singleton BackgroundEvaluator, creating it on first call."""
    global _evaluator
    if _evaluator is None:
        _evaluator = BackgroundEvaluator()
    return _evaluator
```

**Step 4: Run tests**

```bash
pytest tests/test_trajectory_smoke.py -v
```
Expected: all PASSED

**Step 5: Commit**

```bash
git add blackwell/background_eval.py tests/test_trajectory_smoke.py
git commit -m "feat: BackgroundEvaluator — async scoring with regret-threshold Oracle trigger"
```

---

### Task 3: Wire trajectory logging into `agent.py`

**Files:**
- Modify: `agent.py` — add imports, wire into main loop, add /trajectory and /feedback commands

**Step 1: Add imports after existing blackwell imports (around line 24)**

```python
# Trajectory logging (always-on, even without full LOGGING)
try:
    from blackwell.trajectory import log_success, log_failure, mark_feedback, get_counts
    from blackwell.background_eval import get_evaluator
    TRAJECTORY = True
except ImportError:
    TRAJECTORY = False
```

**Step 2: Wire success path in `main()` — replace lines ~772–784**

Find this block:
```python
            reply, history = run_agent(user_input, history, tools_called,
                                       stream_cb=_on_token)

            if _streaming_started[0]:
                print("<<ZE>>", flush=True)
            else:
                print(f"\nZephyr: {reply or '(no response)'}\n", flush=True)

            # Log to Blackwell DB
            if LOGGING and session_id:
                turn += 1
                log_exchange(session_id, turn, user_input, reply, tools_called)
```

Replace with:
```python
            reply, history = run_agent(user_input, history, tools_called,
                                       stream_cb=_on_token)

            if _streaming_started[0]:
                print("<<ZE>>", flush=True)
            else:
                print(f"\nZephyr: {reply or '(no response)'}\n", flush=True)

            # Log to Blackwell DB
            if LOGGING and session_id:
                turn += 1
                exchange_id = log_exchange(session_id, turn, user_input, reply, tools_called)
                # Background evaluation — scores this turn, updates x̄
                if TRAJECTORY:
                    log_success(session_id, turn, user_input, reply, tools_called)
                    get_evaluator().submit(exchange_id, user_input, reply)
            elif TRAJECTORY:
                # LOGGING unavailable but trajectory still works standalone
                turn += 1
                log_success(f"notx-{turn}", turn, user_input, reply, tools_called)
```

**Step 3: Wire failure path — replace exception handler (~line 789)**

```python
        except Exception as e:
            print(f"Error: {e}\n", flush=True)
            print("<<ZE>>", flush=True)  # clear GUI loading state
            if TRAJECTORY:
                log_failure(
                    session_id or "no-session",
                    turn + 1,
                    user_input,
                    "exception",
                    str(e),
                )
```

**Step 4: Add `/trajectory` and `/feedback` to `handle_cli()`**

In `CLI_COMMANDS` dict, add:
```python
    "/trajectory": "Show trajectory pair counts and current regret vector",
    "/feedback":   "Mark last response good or bad — /feedback <session> <turn> up|down",
```

In `handle_cli()`, add after the `/status` block:
```python
    elif command == "/trajectory":
        if TRAJECTORY:
            counts = get_counts()
            print(f"\nTrajectory pairs:   {counts['success']} success  /  "
                  f"{counts['failed']} failed  /  {counts['feedback']} feedback")
        if LOGGING:
            from blackwell.logger import get_average_vector
            from blackwell.evaluator import total_regret
            avg = get_average_vector()
            if avg:
                print(f"Current x̄:          " +
                      "  ".join(f"{k}={v:.2f}" for k, v in avg.items()))
                print(f"Total regret:        {total_regret(avg):.3f}  (threshold: 0.15)")
            else:
                print("No scored exchanges yet — chat more to build x̄")
        print()

    elif command == "/feedback":
        # /feedback <session_id> <turn> up|down
        parts_f = arg.strip().split()
        if len(parts_f) == 3 and TRAJECTORY:
            sess_f, turn_f, vote = parts_f
            positive = vote.lower() in ("up", "👍", "good", "1")
            mark_feedback(sess_f, int(turn_f), positive)
            print(f"[feedback] {'👍' if positive else '👎'} recorded for turn {turn_f}\n")
        else:
            print("Usage: /feedback <session_id> <turn> up|down\n")
```

**Step 5: Verify agent.py starts and responds**

```bash
cd C:\Users\gamer23\Desktop\hermes-agent
python -c "
import subprocess, sys, time, threading
proc = subprocess.Popen([sys.executable, '-u', 'agent.py'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
got = []
def r():
    for line in proc.stdout:
        s = line.rstrip()
        got.append(s)
        if s == '<<ZE>>': return
threading.Thread(target=r, daemon=True).start()
time.sleep(3)
proc.stdin.write('hello\n'); proc.stdin.flush()
time.sleep(20)
proc.terminate()
print([x for x in got if x][:10])
"
```
Expected: see `<<ZS>>` tokens and `<<ZE>>` in output. No crash.

**Step 6: Commit**

```bash
git add agent.py
git commit -m "feat: wire trajectory logging into agent main loop — success, failure, /trajectory, /feedback"
```

---

### Task 4: GGUF export pipeline (`blackwell/export.py` + lora_steer.py)

**Files:**
- Create: `blackwell/export.py`
- Modify: `blackwell/lora_steer.py` — add GGUF export after adapter save

**Step 1: Add GGUF export to `lora_steer.py`**

Find the end of `run_lora_cycle()` where adapter is saved:
```python
    trainer.model.save_pretrained(OUTPUT_DIR)
    trainer.tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[BlackLoRA] Adapter saved to {OUTPUT_DIR}")
```

Replace with:
```python
    trainer.model.save_pretrained(OUTPUT_DIR)
    trainer.tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[BlackLoRA] Adapter saved to {OUTPUT_DIR}", flush=True)

    # Export to GGUF for Ollama
    print("[BlackLoRA] Exporting to GGUF (Q4_K_M)...", flush=True)
    gguf_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "gguf")
    try:
        trainer.model.save_pretrained_gguf(
            gguf_dir,
            trainer.tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"[BlackLoRA] GGUF saved to {gguf_dir}", flush=True)
        return gguf_dir
    except Exception as e:
        print(f"[BlackLoRA] GGUF export failed: {e}", flush=True)
        return None
```

Also update function signature: `def run_lora_cycle() -> str | None:` (returns gguf_dir or None)

**Step 2: Create `blackwell/export.py`**

```python
"""
blackwell/export.py
Registers a GGUF model with Ollama after LoRA training.

Pipeline:
  GGUF dir (from lora_steer.py)
    → write Modelfile
    → ollama create zephyr-steered -f Modelfile
    → print progress for GUI console
"""
import os
import glob
import subprocess

MODEL_NAME = "zephyr-steered"
_HERE = os.path.dirname(__file__)
MODELFILE_PATH = os.path.join(_HERE, "..", "blackwell", "Modelfile")


def register_with_ollama(gguf_dir: str) -> bool:
    """
    Given a directory containing a .gguf file, write a Modelfile
    and run `ollama create zephyr-steered`.
    Returns True on success.
    """
    # Find the quantized GGUF file
    pattern = os.path.join(gguf_dir, "*.gguf")
    matches = glob.glob(pattern)
    if not matches:
        print(f"[export] No .gguf file found in {gguf_dir}", flush=True)
        return False

    gguf_path = os.path.abspath(matches[0])
    print(f"[export] Found GGUF: {gguf_path}", flush=True)

    # Write Modelfile
    modelfile_content = f"FROM {gguf_path}\n"
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"[export] Modelfile written to {MODELFILE_PATH}", flush=True)

    # Run ollama create
    print(f"[export] Running: ollama create {MODEL_NAME} ...", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH],
            capture_output=False,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"[export] ✓ Model '{MODEL_NAME}' registered in Ollama", flush=True)
            print(f"[export] Switch to it in the model card to use your steered model.",
                  flush=True)
            return True
        else:
            print(f"[export] ollama create failed (rc={result.returncode})", flush=True)
            return False
    except FileNotFoundError:
        print("[export] 'ollama' not found in PATH — install Ollama first", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[export] ollama create timed out after 5 minutes", flush=True)
        return False
```

**Step 3: Wire export into `/run_lora` CLI command in `agent.py`**

Find the `/run_lora` handler in `handle_cli()`:
```python
    elif command == "/run_lora":
```

Replace/extend it:
```python
    elif command == "/run_lora":
        print("[BlackLoRA] Checking training data...\n", flush=True)
        try:
            from blackwell.lora_steer import run_lora_cycle, check_training_data
            ok, msg = check_training_data()
            if not ok:
                print(f"[BlackLoRA] Not enough data: {msg}\n")
            else:
                gguf_dir = run_lora_cycle()
                if gguf_dir:
                    from blackwell.export import register_with_ollama
                    register_with_ollama(gguf_dir)
                else:
                    print("[BlackLoRA] Training complete. GGUF export skipped.\n")
        except Exception as e:
            print(f"[BlackLoRA] Error: {e}\n")
```

**Step 4: Commit**

```bash
git add blackwell/export.py blackwell/lora_steer.py agent.py
git commit -m "feat: GGUF export pipeline — lora_steer → GGUF → ollama create zephyr-steered"
```

---

### Task 5: GUI — Thumbs up/down buttons

**Files:**
- Modify: `zephyr_gui.py` — add FeedbackBar widget, wire after stream ends

**Step 1: Add `FeedbackBar` widget class after `ConsoleWidget` class (after line ~970)**

```python
class FeedbackBar(QWidget):
    """
    Small thumbs-up / thumbs-down row appended to the console
    after each assistant response completes.
    """
    feedback_given = Signal(bool)  # True = up, False = down

    _BG    = "#0d1117"
    _TEAL  = "#1a8272"
    _RED   = "#8b1a1a"
    _DIM   = "#3a4a5a"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._voted = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._up   = QPushButton("▲")
        self._down = QPushButton("▼")
        for btn in (self._up, self._down):
            btn.setFixedSize(22, 18)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {self._DIM};
                    border: 1px solid {self._DIM};
                    border-radius: 3px;
                    font-size: 9px;
                }}
                QPushButton:hover {{ color: white; border-color: white; }}
            """)

        self._up.clicked.connect(lambda: self._vote(True))
        self._down.clicked.connect(lambda: self._vote(False))

        layout.addStretch()
        layout.addWidget(self._up)
        layout.addWidget(self._down)

    def _vote(self, positive: bool):
        if self._voted:
            return
        self._voted = True
        color = self._TEAL if positive else self._RED
        btn = self._up if positive else self._down
        btn.setStyleSheet(btn.styleSheet().replace(self._DIM, color))
        self.feedback_given.emit(positive)
```

**Step 2: Wire FeedbackBar into `MainWindow`**

In `MainWindow.__init__`, after connecting `stream_ended`:
```python
        # Thumbs feedback
        self._current_session_id: str = ""
        self._current_turn: int = 0
        self._process.stream_ended.connect(self._on_stream_ended_feedback,
                                           Qt.ConnectionType.QueuedConnection)
```

Add helper methods:
```python
    def _on_stream_ended_feedback(self):
        """Append a FeedbackBar to the console after each response."""
        bar = FeedbackBar()
        bar.feedback_given.connect(self._on_feedback)
        # Insert into console as a child widget at the bottom
        self._console.add_feedback_bar(bar)

    def _on_feedback(self, positive: bool):
        vote = "up" if positive else "down"
        self._process.send_input(
            f"/feedback {self._current_session_id} {self._current_turn} {vote}"
        )
```

**Step 3: Add `add_feedback_bar()` to `ConsoleWidget`**

In `ConsoleWidget` class, add:
```python
    def add_feedback_bar(self, bar: "FeedbackBar") -> None:
        """Overlay a FeedbackBar at the bottom-right of the console."""
        bar.setParent(self)
        bar.adjustSize()
        # Position bottom-right with margin
        x = self.width() - bar.width() - 8
        y = self.height() - bar.height() - 4
        bar.move(x, y)
        bar.show()
        # Remove after 8 seconds or after vote
        from PySide6.QtCore import QTimer
        QTimer.singleShot(8000, bar.deleteLater)
```

**Step 4: Track session_id and turn in MainWindow**

In `_on_user_input()`:
```python
    def _on_user_input(self, text: str):
        self._console.append_line(f"You: {text}")
        self._current_turn += 1
        self._thinking_bar.set_loading()
        self._process.send_input(text)
```

In `_on_agent_exit()`, add: `self._current_turn = 0`

The session_id needs to come from the agent. Add a new marker: when agent starts a session it prints `<<SESSION:uuid>>`. The GUI parses this in `append_line`:
```python
    # in append_line, before the streaming protocol block:
    if line.startswith("<<SESSION:") and line.endswith(">>"):
        # Handled by MainWindow via output_signal
        return
```

In `MainWindow`, connect a session-id handler:
```python
        self._process.output_signal.connect(self._on_agent_output)

    def _on_agent_output(self, line: str):
        if line.startswith("<<SESSION:") and line.endswith(">>"):
            self._current_session_id = line[10:-2]
```

In `agent.py` `main()`, after `session_id = new_session(MODEL)`:
```python
        print(f"<<SESSION:{session_id}>>", flush=True)
```
If LOGGING is False, print a generated UUID:
```python
        import uuid as _uuid
        print(f"<<SESSION:{_uuid.uuid4()}>>", flush=True)
```

**Step 5: Commit**

```bash
git add zephyr_gui.py agent.py
git commit -m "feat: GUI thumbs up/down FeedbackBar after each assistant response"
```

---

### Task 6: GUI — `/trajectory` palette button + pair-count badges

**Files:**
- Modify: `zephyr_gui.py` — update `PaletteWidget` and `ZephyrButton`

**Step 1: Extend `ZephyrButton` to support a badge string**

Find `class ZephyrButton(QPushButton):` at line 554. Add a `set_badge(text)` method:

```python
    def set_badge(self, text: str) -> None:
        """Show a small count badge after the label text."""
        self._badge = text
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not getattr(self, "_badge", ""):
            return
        from PySide6.QtGui import QPainter, QColor, QFont
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Consolas", 7)
        p.setFont(font)
        p.setPen(QColor("#4dcdb4"))
        rect = self.rect()
        p.drawText(rect.adjusted(0, 0, -6, 0),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   self._badge)
```

**Step 2: Add `/trajectory` to `PaletteWidget.BUTTONS`**

In `BUTTONS` list, after `/coding-blackwell` entry, add:
```python
        (
            "/trajectory",
            "/trajectory",
            "Show trajectory pair counts and current regret vector.\n"
            "Every real conversation is logged here for fine-tuning.",
            True,
        ),
```

**Step 3: Add live pair-count badge refresh**

In `PaletteWidget.__init__`, after `add_group(self.BUTTONS)`:
```python
        # Keep references to the three training buttons for badge updates
        self._blackwell_btn      = None
        self._coding_btn         = None
        self._trajectory_btn     = None
        # (set during add_group — rewrite add_group to capture refs)
```

Rewrite `add_group` locally to capture refs:
```python
        def add_group(buttons, capture=None):
            for label, cmd, tip, fire in buttons:
                btn = ZephyrButton(label, cmd, tip, fire)
                btn.clicked.connect(
                    lambda checked=False, c=cmd, f=fire:
                        self.command_requested.emit(c, f)
                )
                vbox.addWidget(btn)
                if capture is not None and cmd in capture:
                    capture[cmd] = btn
```

Change the BUTTONS call:
```python
        _caps = {"/blackwell": None, "/coding-blackwell": None, "/trajectory": None}
        add_group(self.BUTTONS, capture=_caps)
        self._blackwell_btn  = _caps["/blackwell"]
        self._coding_btn     = _caps["/coding-blackwell"]
        self._trajectory_btn = _caps["/trajectory"]
```

Add QTimer for badge refresh:
```python
        from PySide6.QtCore import QTimer
        self._badge_timer = QTimer(self)
        self._badge_timer.timeout.connect(self._refresh_badges)
        self._badge_timer.start(30_000)   # every 30s
        self._refresh_badges()            # immediate on startup
```

Add `_refresh_badges` method:
```python
    def _refresh_badges(self):
        try:
            import json, os
            def _count(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return sum(1 for l in f if l.strip())
                except FileNotFoundError:
                    return 0

            base = os.path.join(os.path.dirname(__file__))
            bw   = _count(os.path.join(base, "blackwell", "training_pairs.jsonl"))
            cbw  = _count(os.path.join(base, "blackwell", "coding_training_pairs.jsonl"))
            traj = _count(os.path.join(base, "trajectory_samples.jsonl"))

            if self._blackwell_btn:
                self._blackwell_btn.set_badge(f"{bw} pairs" if bw else "")
            if self._coding_btn:
                self._coding_btn.set_badge(f"{cbw} pairs" if cbw else "")
            if self._trajectory_btn:
                self._trajectory_btn.set_badge(f"{traj} pairs" if traj else "")
        except Exception:
            pass
```

**Step 4: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: /trajectory palette button + live pair-count badges on training buttons"
```

---

### Task 7: Smoke test completion + live dry-run checklist

**Files:**
- Modify: `tests/test_trajectory_smoke.py` — add end-to-end smoke test
- Create: `tests/LIVE_DRY_RUN.md` — acceptance checklist

**Step 1: Add end-to-end smoke test**

```python
# append to tests/test_trajectory_smoke.py

def test_full_smoke_five_turns(tmp_path, monkeypatch):
    """
    Fire 5 fake turns through the logger:
      - turn 1: success
      - turn 2: success with tool
      - turn 3: exception failure
      - turn 4: bad_tool failure
      - turn 5: thumbs-down failure
    Assert all three files have correct counts.
    """
    import blackwell.trajectory as T
    monkeypatch.setattr(T, "SAMPLES_PATH",  str(tmp_path / "trajectory_samples.jsonl"))
    monkeypatch.setattr(T, "FAILURES_PATH", str(tmp_path / "failed_trajectories.jsonl"))
    monkeypatch.setattr(T, "FEEDBACK_PATH", str(tmp_path / "trajectory_feedback.jsonl"))

    T.log_success("s", 1, "hello", "hi", [])
    T.log_success("s", 2, "search?", "found it", ["web_search"])
    T.log_failure("s", 3, "crash q", "exception", "ZeroDivisionError")
    T.log_failure("s", 4, "bad q",   "bad_tool",  "wrong params")
    T.mark_feedback("s", 5, positive=False)  # thumbs down

    counts = T.get_counts()
    assert counts["success"]  == 2
    assert counts["failed"]   == 2
    assert counts["feedback"] == 1

    # Verify ShareGPT format
    import json
    sample = json.loads(open(T.SAMPLES_PATH).readlines()[0])
    assert "conversations" in sample
    assert sample["conversations"][0]["from"] == "human"
    assert sample["conversations"][1]["from"] == "gpt"
    assert sample["source"] == "trajectory"
```

**Step 2: Run full smoke suite**

```bash
pytest tests/test_trajectory_smoke.py -v
```
Expected: all tests PASS

**Step 3: Create `tests/LIVE_DRY_RUN.md`**

```markdown
# Trajectory Unified Loop — Live Dry-Run Checklist

Run through this after every major deployment to confirm the end-to-end loop works.

## Setup
- [ ] Ollama running (`ollama serve`)
- [ ] hermes3:8b pulled (`ollama list`)
- [ ] Zephyr GUI launched (`python zephyr_gui.py`)

## Trajectory Logging
- [ ] Send 10 chat messages — agent responds normally
- [ ] ThinkingBar clears after each response (no stuck-loading regressions)
- [ ] After last message, check `trajectory_samples.jsonl` has ~10 new lines
- [ ] Format: each line is valid JSON with `conversations`, `source`, `session_id`

## Thumbs Feedback
- [ ] Thumbs ▲/▼ bar appears briefly after each response
- [ ] Click ▲ on one response — check `trajectory_feedback.jsonl` has a `positive: true` entry
- [ ] Click ▼ on another — check `trajectory_feedback.jsonl` has a `positive: false` entry
- [ ] Run `/feedback` without args — prints usage line, no crash

## /trajectory Command
- [ ] Type `/trajectory` in chat — prints pair counts and x̄ vector
- [ ] Counts match line count of `trajectory_samples.jsonl`

## Palette Badges
- [ ] Wait 30s — `/trajectory` button badge shows correct pair count
- [ ] `/blackwell` badge shows correct count from `blackwell/training_pairs.jsonl`
- [ ] Badges update without restarting the GUI

## Background Evaluator
- [ ] After 5+ turns, check `blackwell.db` has new rows with scores
  ```bash
  python -c "from blackwell.logger import get_average_vector; print(get_average_vector())"
  ```
- [ ] x̄ vector changes after real conversations (not just Blackwell sessions)

## Export Pipeline (requires ≥200 pairs + GPU)
- [ ] Run `/run_lora` — prints training progress
- [ ] Prints `[BlackLoRA] GGUF saved to blackwell/adapters/gguf`
- [ ] Prints `[export] ✓ Model 'zephyr-steered' registered in Ollama`
- [ ] Model appears in ModelSwitcherCard dropdown
- [ ] Switch to `zephyr-steered` — conversations work normally
```

**Step 4: Run smoke suite one final time**

```bash
pytest tests/test_trajectory_smoke.py -v --tb=short
```
Expected: all PASSED

**Step 5: Final commit**

```bash
git add tests/test_trajectory_smoke.py tests/LIVE_DRY_RUN.md docs/plans/2026-04-14-trajectory-unified-loop-design.md docs/plans/2026-04-14-trajectory-unified-loop.md
git commit -m "feat: trajectory unified loop complete — smoke tests + live dry-run checklist"
```

# Zephyr GUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `zephyr_gui.py` — a PySide6 desktop GUI that wraps `agent.py` via subprocess pipe, with a custom animated button palette and hacker-aesthetic console.

**Architecture:** `ZephyrProcess` (QThread) spawns `agent.py` as a subprocess and streams stdout back to the GUI via Qt signals. `MainWindow` splits into a console pane (left ⅔) and a scrollable command palette (right ⅓). Every button either fires a command immediately or pre-fills the input bar for commands needing arguments.

**Tech Stack:** Python 3.9, PySide6 6.x, `subprocess`, `QThread`, `QPainter`, `QTimer`, `math.sin`

---

## Pre-flight Checklist

- `agent.py` must exist at `C:\Users\gamer23\Desktop\hermes-agent\agent.py`
- Ollama must be running (`ollama serve`)
- Python 3.9 is the target — **no `X | Y` union type syntax anywhere**, use `Optional[X]` from `typing` or omit annotations entirely
- PySide6 must be installed (Task 1)

---

## Task 1: Install PySide6

**Files:**
- No file changes — environment setup only

**Step 1: Install**

```bash
pip install PySide6
```

**Step 2: Verify**

```bash
python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"
```

Expected output: `PySide6 OK`

**Step 3: Commit (nothing to commit — note in log)**

```bash
# No file changes. PySide6 is now a runtime dependency.
# Add to README or requirements.txt if one exists.
```

---

## Task 2: ZephyrProcess — Subprocess Thread

**Files:**
- Create: `C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py` (start the file here)

**Context:** `ZephyrProcess` is a `QThread` that owns the `agent.py` subprocess. It reads stdout in a tight loop and emits signals to the GUI thread. Writing to stdin is thread-safe via Qt's signal/slot connection.

**Step 1: Write a smoke test script**

Create a throwaway test file `test_process.py` in the project root:

```python
# test_process.py  — delete after confirming it works
import sys
import subprocess
import time

proc = subprocess.Popen(
    [sys.executable, "agent.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=r"C:\Users\gamer23\Desktop\hermes-agent",
)
time.sleep(2)  # let agent boot
proc.stdin.write("hello\n")
proc.stdin.flush()
time.sleep(3)
for _ in range(20):
    line = proc.stdout.readline()
    if not line:
        break
    print(repr(line))
proc.terminate()
```

Run it:
```bash
cd C:\Users\gamer23\Desktop\hermes-agent
python test_process.py
```

Expected: lines of Zephyr's boot text and a reply to "hello" printed to terminal. If you see output, the pipe works.

**Step 2: Create `zephyr_gui.py` with ZephyrProcess**

```python
"""
zephyr_gui.py — Zephyr Command Workbench
Prycat Research Team
PySide6 GUI wrapping agent.py via subprocess pipe.
Python 3.9 compatible.
"""
import sys
import math
import subprocess
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPointF, QRectF
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QRadialGradient,
    QLinearGradient, QFont, QPalette, QFontDatabase
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPlainTextEdit, QLineEdit, QPushButton,
    QSplitter, QScrollArea, QLabel, QSizePolicy, QFrame
)

AGENT_PATH = r"C:\Users\gamer23\Desktop\hermes-agent\agent.py"
MODEL_NAME  = "hermes3:8b"

# ─── Colours ──────────────────────────────────────────────────
C_BG         = QColor("#090c10")
C_SURFACE    = QColor("#0d1117")
C_TEAL       = QColor(128, 221, 202, 235)
C_TEAL_DIM   = QColor(128, 221, 202, 133)
C_MUTED      = QColor(170, 182, 194, 143)
C_CYAN       = QColor(122, 184, 216, 220)
C_AMBER      = QColor(212, 160, 80,  220)
C_GREEN      = QColor(102, 196, 122, 220)
C_RED        = QColor(210, 90,  90,  220)
C_WHITE_DIM  = QColor(215, 223, 230, 185)


# ═══════════════════════════════════════════════════════════════
#  ZephyrProcess — subprocess thread
# ═══════════════════════════════════════════════════════════════
class ZephyrProcess(QThread):
    output_signal   = Signal(str)
    finished_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None   # type: Optional[subprocess.Popen]

    def run(self):
        self._proc = subprocess.Popen(
            [sys.executable, AGENT_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        for line in self._proc.stdout:
            self.output_signal.emit(line.rstrip("\n"))
        self.finished_signal.emit()

    def send_input(self, text: str):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
            except OSError:
                pass

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
            except OSError:
                pass
```

**Step 3: Run a quick import check**

```bash
python -c "from zephyr_gui import ZephyrProcess; print('ZephyrProcess OK')"
```

Expected: `ZephyrProcess OK`

**Step 4: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add ZephyrProcess subprocess thread skeleton"
```

---

## Task 3: ZephyrButton — Base Structure

**Files:**
- Modify: `zephyr_gui.py` — append after ZephyrProcess

**Context:** `ZephyrButton` is a fully custom `QPushButton`. Its `paintEvent` draws everything manually. The constructor takes the display label, the command string to inject, a tooltip, and whether to fire immediately or just pre-fill the input bar.

**Step 1: Append ZephyrButton skeleton to `zephyr_gui.py`**

```python
# ═══════════════════════════════════════════════════════════════
#  ZephyrButton — Monolith Signal custom button
# ═══════════════════════════════════════════════════════════════
class ZephyrButton(QPushButton):
    """
    Custom button with:
    - Scanline grain texture
    - BorderWake breathing pulse
    - Hover sweep (teal line across top+bottom border)
    - Mouse-tracking radial glow blob
    - State dot (idle/running/success/error)
    """

    SWEEP_MS   = 700      # hover sweep duration in ms
    WAKE_MS    = 5500     # borderWake pulse period in ms
    TICK_MS    = 16       # ~60fps timer interval

    def __init__(
        self,
        label: str,
        command: str,
        tooltip: str,
        fire_immediately: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.label           = label        # display text e.g. "/blackwell"
        self.command         = command      # text to inject e.g. "/blackwell"
        self.fire_immediately = fire_immediately
        self._state          = "idle"       # idle | running | success | error

        # Animation state
        self._sweep_t        = 0.0          # 0→1 hover sweep progress
        self._sweeping       = False
        self._wake_t         = 0            # ms elapsed for borderWake sin
        self._mouse_pos      = QPointF(-1, -1)
        self._mouse_inside   = False
        self._state_tint_a   = 0.0          # success/error tint alpha 0→1

        # Timer
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # Widget setup
        self.setMouseTracking(True)
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setFlat(True)
        self.setAttribute(Qt.WA_Hover, True)

    # ── State management ───────────────────────────────────────
    def set_state(self, state: str):
        """state: idle | running | success | error"""
        self._state = state
        if state in ("success", "error"):
            self._state_tint_a = 1.0
        else:
            self._state_tint_a = 0.0

    # ── Animation tick ─────────────────────────────────────────
    def _tick(self):
        dt = self.TICK_MS
        self._wake_t = (self._wake_t + dt) % self.WAKE_MS

        if self._sweeping and self._sweep_t < 1.0:
            self._sweep_t = min(1.0, self._sweep_t + dt / self.SWEEP_MS)

        if self._state_tint_a > 0:
            self._state_tint_a = max(0.0, self._state_tint_a - dt / 1200.0)

        self.update()

    # ── Mouse tracking ─────────────────────────────────────────
    def enterEvent(self, event):
        self._mouse_inside = True
        self._sweeping     = True
        self._sweep_t      = 0.0
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._mouse_inside = False
        self._sweeping     = False
        self._sweep_t      = 0.0
        self._mouse_pos    = QPointF(-1, -1)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        self._mouse_pos = QPointF(event.position())
        super().mouseMoveEvent(event)

    # ── Paint ──────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rect = QRectF(0, 0, w, h)

        # 1. Base fill
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(C_SURFACE))
        p.drawRoundedRect(rect, 4, 4)

        # 2. Scanline grain
        p.setPen(QPen(QColor(255, 255, 255, 5), 1))
        y = 0
        while y < h:
            p.drawLine(0, y, w, y)
            y += 3

        # 3. Mouse glow blob
        if self._mouse_inside and self._mouse_pos.x() >= 0:
            glow = QRadialGradient(self._mouse_pos, 60)
            glow.setColorAt(0,   QColor(77, 194, 179, 46))
            glow.setColorAt(1,   QColor(77, 194, 179, 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rect, 4, 4)

        # 4. Inner bevel
        p.setPen(QPen(QColor(255, 255, 255, 8), 1))
        p.drawLine(1, 1, w - 1, 1)      # top
        p.drawLine(1, 1, 1, h - 1)      # left
        p.setPen(QPen(QColor(0, 0, 0, 100), 1))
        p.drawLine(1, h - 1, w - 1, h - 1)   # bottom
        p.drawLine(w - 1, 1, w - 1, h - 1)   # right

        # 5. Resting border
        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        # 6. BorderWake pulse
        wake_phase = math.sin(2 * math.pi * self._wake_t / self.WAKE_MS)
        wake_alpha = int(2 + 6 * (wake_phase * 0.5 + 0.5))
        p.setPen(QPen(QColor(77, 194, 179, wake_alpha), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        # 7. Hover sweep — teal shard across top & bottom border
        if self._sweep_t > 0:
            ease = 1.0 - (1.0 - self._sweep_t) ** 3   # ease-out cubic
            sx = ease * w
            shard_alpha = int(180 * (1.0 - self._sweep_t))
            sweep_color = QColor(128, 221, 202, shard_alpha)
            p.setPen(QPen(sweep_color, 1))
            p.drawLine(int(sx) - 24, 0, int(sx), 0)       # top edge
            p.drawLine(int(sx) - 24, h - 1, int(sx), h - 1)  # bottom edge

        # 8. State tint (success=green, error=amber)
        if self._state_tint_a > 0 and self._state in ("success", "error"):
            tint_color = C_GREEN if self._state == "success" else C_AMBER
            tint = QColor(tint_color)
            tint.setAlpha(int(40 * self._state_tint_a))
            p.setBrush(QBrush(tint))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rect, 4, 4)

        # 9. Label text
        font = QFont("Consolas", 9)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        font.setBold(True)
        p.setFont(font)
        text_color = QColor(C_TEAL) if self._mouse_inside else QColor(C_TEAL_DIM)
        p.setPen(QPen(text_color))
        text_rect = QRectF(12, 0, w - 30, h)
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.label)

        # 10. State dot (right side)
        dot_x = w - 14
        dot_y = h // 2
        dot_r = 4
        if self._state == "idle":
            dot_color = QColor(255, 255, 255, 40)
        elif self._state == "running":
            pulse = math.sin(2 * math.pi * self._wake_t / 800) * 0.5 + 0.5
            dot_color = QColor(C_AMBER)
            dot_color.setAlpha(int(120 + 100 * pulse))
        elif self._state == "success":
            dot_color = C_GREEN
        else:
            dot_color = C_RED
        p.setBrush(QBrush(dot_color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2)

        p.end()
```

**Step 2: Import check**

```bash
python -c "from zephyr_gui import ZephyrButton; print('ZephyrButton OK')"
```

Expected: `ZephyrButton OK`

**Step 3: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add ZephyrButton with paintEvent animations"
```

---

## Task 4: ConsoleWidget

**Files:**
- Modify: `zephyr_gui.py` — append after ZephyrButton

**Context:** Read-only text area. Receives lines from `ZephyrProcess.output_signal` and colorizes them based on content. Stops auto-scrolling if the user scrolls up.

**Step 1: Append ConsoleWidget**

```python
# ═══════════════════════════════════════════════════════════════
#  ConsoleWidget
# ═══════════════════════════════════════════════════════════════
class ConsoleWidget(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._auto_scroll = True

        font = QFont("Consolas", 10)
        self.setFont(font)

        palette = self.palette()
        palette.setColor(QPalette.Base, C_BG)
        palette.setColor(QPalette.Text, C_TEAL)
        self.setPalette(palette)

        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #090c10;
                color: rgba(128,221,202,0.92);
                border: none;
                selection-background-color: #1a3a40;
            }
            QScrollBar:vertical {
                background: #0d1117;
                width: 6px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #2a3a4a;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def _on_scroll(self, value: int):
        at_bottom = value == self.verticalScrollBar().maximum()
        self._auto_scroll = at_bottom

    def append_line(self, line: str):
        """Colorize and append one line from the agent."""
        stripped = line.strip()

        if stripped.startswith("You:"):
            color = "#aab6c2"
        elif stripped.startswith("Zephyr:"):
            color = "#80ddca"
        elif any(tok in stripped for tok in ["→", "[tool]", "tool_call", "Running"]):
            color = "#7ab8d8"
        elif any(tok in stripped for tok in ["Error", "error", "Traceback", "failed"]):
            color = "#d4a050"
        elif stripped.startswith("─") or stripped.startswith("="):
            color = "#445566"
        else:
            color = "#80ddca"

        # Escape HTML special chars then wrap in color span
        safe = (line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        html = f'<span style="color:{color}; font-family:Consolas,monospace;">{safe}</span>'
        self.appendHtml(html)

        if self._auto_scroll:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )
```

**Step 2: Import check**

```bash
python -c "from zephyr_gui import ConsoleWidget; print('ConsoleWidget OK')"
```

**Step 3: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add ConsoleWidget with color coding and auto-scroll"
```

---

## Task 5: InputBar

**Files:**
- Modify: `zephyr_gui.py` — append after ConsoleWidget

**Context:** Single-line input. Up/Down arrows cycle command history. Enter fires. Exposes `inject(text, fire)` so buttons can pre-fill it.

**Step 1: Append InputBar**

```python
# ═══════════════════════════════════════════════════════════════
#  InputBar
# ═══════════════════════════════════════════════════════════════
class InputBar(QLineEdit):
    """
    Signals:
        submitted(str) — emitted when user hits Enter or clicks Send
    """
    submitted = Signal(str)

    HISTORY_MAX = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history      = []   # list of str
        self._history_idx  = -1   # -1 = not browsing

        font = QFont("Consolas", 10)
        self.setFont(font)
        self.setPlaceholderText("▶  type a message or /command...")
        self.setStyleSheet("""
            QLineEdit {
                background-color: #0d1117;
                color: rgba(128,221,202,0.92);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 4px;
                padding: 8px 12px;
                selection-background-color: #1a3a40;
            }
            QLineEdit:focus {
                border-color: rgba(128,221,202,0.38);
            }
        """)
        self.returnPressed.connect(self._fire)

    def _fire(self):
        text = self.text().strip()
        if not text:
            return
        # Add to history
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            if len(self._history) > self.HISTORY_MAX:
                self._history.pop(0)
        self._history_idx = -1
        self.clear()
        self.submitted.emit(text)

    def inject(self, text: str, fire: bool = False):
        """Pre-fill the input bar. If fire=True, submit immediately."""
        self.setFocus()
        self.setText(text)
        self.setCursorPosition(len(text))
        if fire:
            self._fire()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            if self._history:
                if self._history_idx == -1:
                    self._history_idx = len(self._history) - 1
                elif self._history_idx > 0:
                    self._history_idx -= 1
                self.setText(self._history[self._history_idx])
                self.setCursorPosition(len(self.text()))
            return
        if event.key() == Qt.Key_Down:
            if self._history_idx != -1:
                if self._history_idx < len(self._history) - 1:
                    self._history_idx += 1
                    self.setText(self._history[self._history_idx])
                else:
                    self._history_idx = -1
                    self.clear()
                self.setCursorPosition(len(self.text()))
            return
        super().keyPressEvent(event)
```

**Step 2: Import check**

```bash
python -c "from zephyr_gui import InputBar; print('InputBar OK')"
```

**Step 3: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add InputBar with command history and inject()"
```

---

## Task 6: HeaderBar

**Files:**
- Modify: `zephyr_gui.py` — append after InputBar

**Step 1: Append HeaderBar**

```python
# ═══════════════════════════════════════════════════════════════
#  HeaderBar
# ═══════════════════════════════════════════════════════════════
class HeaderBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("background-color: #0d1117;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # Pulsing dot
        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #66c47a; font-size: 10px;")
        layout.addWidget(self._dot)

        # ZEPHYR label
        title = QLabel("ZEPHYR")
        title.setStyleSheet("""
            color: rgba(128,221,202,0.92);
            font-family: Consolas, monospace;
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 4px;
        """)
        layout.addWidget(title)

        # Model name
        model_lbl = QLabel(MODEL_NAME)
        model_lbl.setStyleSheet("""
            color: rgba(170,182,194,0.5);
            font-family: Consolas, monospace;
            font-size: 10px;
            letter-spacing: 2px;
        """)
        layout.addWidget(model_lbl)
        layout.addStretch()

        # Dot pulse timer
        self._dot_on = True
        dot_timer = QTimer(self)
        dot_timer.setInterval(750)
        dot_timer.timeout.connect(self._pulse_dot)
        dot_timer.start()

    def _pulse_dot(self):
        self._dot_on = not self._dot_on
        alpha = "0.95" if self._dot_on else "0.35"
        self._dot.setStyleSheet(f"color: rgba(102,196,122,{alpha}); font-size: 10px;")
```

**Step 2: Import check**

```bash
python -c "from zephyr_gui import HeaderBar; print('HeaderBar OK')"
```

**Step 3: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add HeaderBar with pulsing live indicator"
```

---

## Task 7: PaletteWidget — Button Panel

**Files:**
- Modify: `zephyr_gui.py` — append after HeaderBar

**Context:** Scrollable right panel. Buttons are defined in an ordered list with section dividers. Buttons emit a signal carrying `(command, fire_immediately)` when clicked. `MainWindow` connects this to `InputBar.inject()`.

**Step 1: Append PaletteWidget**

```python
# ═══════════════════════════════════════════════════════════════
#  PaletteWidget — right command panel
# ═══════════════════════════════════════════════════════════════
class SectionDivider(QWidget):
    """Thin labelled divider between button groups."""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 2)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet("""
            color: rgba(170,182,194,0.45);
            font-family: Consolas, monospace;
            font-size: 9px;
            letter-spacing: 3px;
        """)
        layout.addWidget(lbl)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: rgba(255,255,255,0.06);")
        layout.addWidget(line)
        layout.setStretch(1, 1)


class PaletteWidget(QWidget):
    # Emits (command: str, fire: bool)
    command_requested = Signal(str, bool)

    # Button definitions — (label, command, tooltip, fire_immediately)
    BUTTONS = [
        (
            "/blackwell",
            "/blackwell",
            "Drops Zephyr into a planning space where he interviews you,\n"
            "and your answers reshape how he sees the world — permanently.",
            True,
        ),
        (
            "/help",
            "/help",
            "Show all available commands.",
            True,
        ),
        (
            "/tools",
            "/tools",
            "List all of Zephyr's active tools.",
            True,
        ),
        (
            "/search",
            "/search ",
            "Raw DuckDuckGo search, instant.\nUsage: /search <query>",
            False,
        ),
        (
            "/browse",
            "/browse ",
            "Fetch a URL directly.\nUsage: /browse <url>",
            False,
        ),
        (
            "/run",
            "/run ",
            "Run Python immediately.\nUsage: /run <code>",
            False,
        ),
        (
            "/status",
            "/status",
            "Check that Ollama is alive and responding.",
            True,
        ),
        (
            "/model",
            "/model",
            "Show current model name and API connection info.",
            True,
        ),
        (
            "/save",
            "/save",
            "Save conversation to Obsidian vault as a formatted .md\n"
            "with YAML frontmatter (date, time, model, tags).\n"
            "Usage: /save  or  /save my research chat",
            True,
        ),
        (
            "/clear",
            "/clear",
            "Reset conversation history.\nZephyr will ask for confirmation (y/n).",
            True,
        ),
    ]

    KEYS_BUTTONS = [
        (
            "/keys setup",
            "/keys setup",
            "Interactive wizard: select provider, enter your API key.\n"
            "Stored masked in ~/.zephyr/keys.json.\n"
            "Providers: claude, gpt, grok, gemini",
            True,
        ),
        (
            "/keys list",
            "/keys list",
            "Show which providers are configured:\n"
            "claude ✓  gpt ✓  grok ✗  gemini ✓",
            True,
        ),
    ]

    CALL_BUTTONS = [
        (
            "/call",
            "/call ",
            "Route your message to the best available external AI.\n"
            "Passes context so the AI knows it's consulting for Zephyr/Prycat.\n"
            "Usage: /call <message>",
            False,
        ),
        (
            "/call claude",
            "/call claude ",
            "Force Claude (claude-opus-4-5 via Anthropic).\nUsage: /call claude <message>",
            False,
        ),
        (
            "/call gpt",
            "/call gpt ",
            "Force GPT-4o via OpenAI.\nUsage: /call gpt <message>",
            False,
        ),
        (
            "/call grok",
            "/call grok ",
            "Force Grok-3 via xAI endpoint.\nUsage: /call grok <message>",
            False,
        ),
        (
            "/call gemini",
            "/call gemini ",
            "Force Gemini 2.0 Flash via Google.\nUsage: /call gemini <message>",
            False,
        ),
    ]

    TRAINING_BUTTONS = [
        (
            "/Run BlackLoRA-N",
            "/run_lora",
            "Run LoRA fine-tuning on completed Blackwell interview data.\n"
            "(Requires 200+ training pairs — check /blackwell first.)",
            True,
        ),
    ]

    SESSION_BUTTONS = [
        (
            "/exit",
            "/exit",
            "Quit Zephyr.",
            True,
        ),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self.setStyleSheet("background-color: #090c10;")

        # Scroll area wrapping inner content
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: #0d1117; width: 4px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #2a3a4a; border-radius: 2px; min-height: 16px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(4)

        # Header label
        hdr = QLabel("COMMAND PALETTE")
        hdr.setStyleSheet("""
            color: rgba(170,182,194,0.4);
            font-family: Consolas, monospace;
            font-size: 9px;
            letter-spacing: 3px;
            padding: 4px 4px 8px 4px;
        """)
        vbox.addWidget(hdr)

        def add_group(buttons):
            for label, cmd, tip, fire in buttons:
                btn = ZephyrButton(label, cmd, tip, fire)
                btn.clicked.connect(
                    lambda checked=False, c=cmd, f=fire: self.command_requested.emit(c, f)
                )
                vbox.addWidget(btn)

        add_group(self.BUTTONS)
        vbox.addWidget(SectionDivider("Keys"))
        add_group(self.KEYS_BUTTONS)
        vbox.addWidget(SectionDivider("External AI"))
        add_group(self.CALL_BUTTONS)
        vbox.addWidget(SectionDivider("Training"))
        add_group(self.TRAINING_BUTTONS)
        vbox.addWidget(SectionDivider("Session"))
        add_group(self.SESSION_BUTTONS)
        vbox.addStretch()

        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
```

**Step 2: Import check**

```bash
python -c "from zephyr_gui import PaletteWidget; print('PaletteWidget OK')"
```

**Step 3: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add PaletteWidget with all 19 command buttons and section dividers"
```

---

## Task 8: MainWindow + Entry Point

**Files:**
- Modify: `zephyr_gui.py` — append after PaletteWidget, add `__main__` block at end

**Step 1: Append MainWindow**

```python
# ═══════════════════════════════════════════════════════════════
#  MainWindow
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zephyr — Prycat Research")
        self.resize(1100, 700)
        self.setMinimumSize(800, 500)

        # ── Central widget ────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        self._header = HeaderBar()
        root.addWidget(self._header)

        # Horizontal divider line
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: rgba(255,255,255,0.06); margin: 0;")
        root.addWidget(div)

        # Splitter: console left | palette right
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #1a2030;
                width: 2px;
            }
        """)

        # Left pane: console + input
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._console = ConsoleWidget()
        left_layout.addWidget(self._console)

        # Input row
        input_row = QWidget()
        input_row.setStyleSheet("background: #0d1117; border-top: 1px solid rgba(255,255,255,0.06);")
        input_row_layout = QHBoxLayout(input_row)
        input_row_layout.setContentsMargins(8, 6, 8, 6)
        input_row_layout.setSpacing(6)

        self._input = InputBar()
        input_row_layout.addWidget(self._input)

        send_btn = QPushButton("SEND")
        send_btn.setFixedWidth(64)
        send_btn.setFixedHeight(34)
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setStyleSheet("""
            QPushButton {
                background: rgba(77,194,179,0.12);
                color: rgba(128,221,202,0.9);
                border: 1px solid rgba(77,194,179,0.25);
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 10px;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: rgba(77,194,179,0.22);
                border-color: rgba(77,194,179,0.45);
            }
            QPushButton:pressed {
                background: rgba(77,194,179,0.08);
            }
        """)
        send_btn.clicked.connect(self._input._fire)
        input_row_layout.addWidget(send_btn)
        left_layout.addWidget(input_row)

        # Right pane: palette
        self._palette = PaletteWidget()

        splitter.addWidget(left_widget)
        splitter.addWidget(self._palette)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([740, 360])
        root.addWidget(splitter)

        # ── Subprocess ────────────────────────────────────────
        self._process = ZephyrProcess(self)
        self._process.output_signal.connect(self._console.append_line)
        self._process.finished_signal.connect(self._on_agent_exit)
        self._process.start()

        # ── Wire input → process ──────────────────────────────
        self._input.submitted.connect(self._on_user_input)

        # ── Wire palette → input ──────────────────────────────
        self._palette.command_requested.connect(self._on_command_requested)

    def _on_user_input(self, text: str):
        self._console.append_line(f"You: {text}")
        self._process.send_input(text)

    def _on_command_requested(self, command: str, fire: bool):
        self._input.inject(command, fire)

    def _on_agent_exit(self):
        self._console.append_line("─── Zephyr process ended ───")

    def closeEvent(self, event):
        self._process.stop()
        self._process.wait(2000)
        super().closeEvent(event)
```

**Step 2: Append global QSS + entry point at the very end of the file**

```python
# ═══════════════════════════════════════════════════════════════
#  Global stylesheet
# ═══════════════════════════════════════════════════════════════
GLOBAL_QSS = """
QMainWindow, QWidget {
    background-color: #090c10;
    color: rgba(128,221,202,0.92);
    font-family: Consolas, monospace;
}

QToolTip {
    background-color: #0d1117;
    color: rgba(128,221,202,0.9);
    border: 1px solid rgba(77,194,179,0.3);
    font-family: Consolas, monospace;
    font-size: 10px;
    padding: 6px 10px;
    border-radius: 4px;
}

QSplitter::handle {
    background: #1a2030;
}
"""


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Zephyr")
    app.setStyleSheet(GLOBAL_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

**Step 3: Full launch test**

Make sure Ollama is running, then:

```bash
cd C:\Users\gamer23\Desktop\hermes-agent
python zephyr_gui.py
```

Expected:
- Window opens, `ZEPHYR ●` header visible
- Green dot pulses
- Console starts receiving Zephyr's boot output
- Type "hello" in input bar → Zephyr responds in console
- Click `/help` button → fires immediately, shows help text
- Click `/search` button → pre-fills `/search ` in input bar, cursor at end
- Hover any button → teal sweep animation runs across border
- Mouse glow blob follows cursor inside button

**Step 4: Final commit**

```bash
git add zephyr_gui.py
git commit -m "feat: complete Zephyr GUI — MainWindow, QSS, entry point"
```

---

## Task 9: Smoke Tests + Polish

**Files:**
- Modify: `zephyr_gui.py` — small fixes only

**Manual test checklist — run through each:**

| Test | Expected |
|---|---|
| Launch GUI | Window opens, no crash |
| Boot output streams in | Console fills with agent boot text |
| Type message, press Enter | `You: <text>` appears, Zephyr replies |
| Click SEND button | Same as Enter |
| Click `/help` | Command fires, help text appears |
| Click `/search` | Input bar pre-filled `/search `, cursor at end |
| Click `/browse` | Input bar pre-filled `/browse ` |
| Click `/call claude` | Input bar pre-filled `/call claude ` |
| Hover button | Teal sweep animation plays |
| Mouse inside button | Radial glow tracks cursor |
| Scroll console up | Auto-scroll pauses |
| Scroll console to bottom | Auto-scroll resumes |
| Press Up arrow in input | Previous command recalled |
| Press Down arrow | Moves forward through history |
| Click `/exit` | Agent exits, console shows "process ended" |
| Close window | No hung processes |

**Step 2: Final commit if any polish edits made**

```bash
git add zephyr_gui.py
git commit -m "polish: Zephyr GUI smoke test fixes"
```

---

## Notes on Python 3.9 Compatibility

- **No `X | Y` union syntax** — use `Optional[X]` from `typing` or omit annotations
- **No `match` statement** — use `if/elif`
- `QPointF`, `QRectF`, `QRadialGradient`, `QLinearGradient` — all available in PySide6 on Py3.9
- `Signal` is `PySide6.QtCore.Signal` — not `pyqtSignal`
- `event.position()` returns `QPointF` in PySide6 (not `.pos()` as in PyQt5)

---

## Cleanup

```bash
# Delete the throwaway test file from Task 2
del test_process.py
git add -A
git commit -m "chore: remove test_process.py throwaway"
```

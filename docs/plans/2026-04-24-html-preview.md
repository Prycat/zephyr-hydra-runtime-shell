# Zephyr HTML Preview Panel — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When the agent finishes streaming a response containing a `\`\`\`html` block, split the console pane to show a live rendered preview on the right using QWebEngineView.

**Architecture:** New `zephyr_html_preview.py` holds the `HtmlPreviewPane` widget and utilities. `zephyr_gui.py` gains a `QSplitter` wrapping the console + preview pane, and `_on_response_complete()` triggers HTML detection + render. Preview starts hidden (width=0), expands to 50/50 when HTML is found.

**Tech Stack:** PySide6 (existing) + PySide6-WebEngine (`pip install PySide6-WebEngine`), stdlib `re`

---

## Codebase context — READ THIS FIRST

- Main file: `zephyr_gui.py` (~4100 lines), PySide6 frameless window
- `MainWindow.__init__` starts at **line 3619**
- Left pane layout built at lines 3654–3737:
  - `left_widget = QWidget()` / `left_layout = QVBoxLayout(left_widget)`
  - **Line 3660:** `self._console = ConsoleWidget()`
  - **Line 3661:** `left_layout.addWidget(self._console)`  ← we change this
  - Line 3664: `self._thinking_bar = ThinkingBar()`
  - Line 3737: `left_layout.addWidget(input_row)`
- `self._process.stream_ended` Signal fires when `<<ZE>>` arrives (generation complete)
- `_on_response_complete()` at **line 3807** — already connected to `stream_ended`; we extend it
- `self._console.toPlainText()` — full text of the console at any time
- `ConsoleWidget` is `QPlainTextEdit` subclass defined at line 1633
- Existing imports at top of `zephyr_gui.py` include all standard PySide6 widgets;
  we add `QWebEngineView` import guarded by try/except

---

### Task 1: HTML extractor utility + availability check

**Files:**
- Create: `zephyr_html_preview.py`
- Create: `tests/test_html_preview.py`

**Step 1: Write the failing tests**

Create `tests/test_html_preview.py`:

```python
# tests/test_html_preview.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zephyr_html_preview import extract_last_html_block, is_webengine_available


def test_extract_simple():
    text = "Here is some code:\n```html\n<h1>Hello</h1>\n```\nDone."
    result = extract_last_html_block(text)
    assert result == "<h1>Hello</h1>"


def test_extract_returns_last_block():
    text = "```html\n<p>first</p>\n```\n\n```html\n<p>second</p>\n```"
    result = extract_last_html_block(text)
    assert result == "<p>second</p>"


def test_extract_multiline():
    text = "```html\n<html>\n<body>\n<p>hi</p>\n</body>\n</html>\n```"
    result = extract_last_html_block(text)
    assert result == "<html>\n<body>\n<p>hi</p>\n</body>\n</html>"


def test_extract_none_when_no_block():
    assert extract_last_html_block("no html here") is None
    assert extract_last_html_block("") is None


def test_extract_incomplete_block_returns_none():
    # Opening fence with no closing fence — not a complete block
    assert extract_last_html_block("```html\n<p>unclosed") is None


def test_is_webengine_available_returns_bool():
    result = is_webengine_available()
    assert isinstance(result, bool)
```

**Step 2: Run to verify fails**

```
pytest tests/test_html_preview.py -v
```
Expected: `ModuleNotFoundError: No module named 'zephyr_html_preview'`

**Step 3: Create `zephyr_html_preview.py` with utilities only**

```python
# -*- coding: utf-8 -*-
"""
zephyr_html_preview.py — HTML preview pane for Zephyr.

HtmlPreviewPane: a QWidget containing a header bar and an embedded
QWebEngineView.  Inserted as the right half of a QSplitter inside the
Zephyr console pane.  Starts hidden (width=0); expands when the agent
produces a complete ```html block.

Requires: pip install PySide6-WebEngine
Graceful degradation: if PySide6-WebEngine is not installed the pane
shows a styled placeholder and the rest of Zephyr is unaffected.
"""
from __future__ import annotations
import re
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
)


# ── Availability check ────────────────────────────────────────────────────────

def is_webengine_available() -> bool:
    """Return True if PySide6-WebEngine is installed."""
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        return True
    except ImportError:
        return False


# ── HTML extractor ────────────────────────────────────────────────────────────

_HTML_BLOCK_RE = re.compile(r'```html\s*\n(.*?)```', re.DOTALL)


def extract_last_html_block(text: str) -> str | None:
    """
    Find the last complete ```html ... ``` block in *text*.

    Returns the inner HTML string (stripped), or None if no complete block
    is present.
    """
    matches = _HTML_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()
```

**Step 4: Run tests to verify they pass**

```
pytest tests/test_html_preview.py -v
```
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add zephyr_html_preview.py tests/test_html_preview.py
git commit -m "feat: add HTML extractor + webengine availability check"
```

---

### Task 2: HtmlPreviewPane widget

**Files:**
- Modify: `zephyr_html_preview.py` (add `HtmlPreviewPane` class)

No unit tests for the widget itself (requires a running QApplication + WebEngine;
tested manually in Task 4).

**Step 1: Append `HtmlPreviewPane` to `zephyr_html_preview.py`**

Add this to the bottom of `zephyr_html_preview.py`:

```python

# ── Colour constants (match Zephyr palette) ───────────────────────────────────

_C_BG     = QColor("#090c10")
_C_HEADER = QColor("#0d1117")
_C_BORDER = QColor("rgba(255,255,255,0.06)")
_C_TEAL   = QColor("#80ddca")
_C_DIM    = QColor("#445566")


# ── HtmlPreviewPane ───────────────────────────────────────────────────────────

class HtmlPreviewPane(QWidget):
    """
    Right-hand pane of the console sub-splitter.

    Header bar: "◈ PREVIEW"  [↺]  [✕]
    Body: QWebEngineView (sandboxed) or a fallback label if WebEngine missing.

    Signals
    -------
    close_requested : emitted when ✕ is clicked
    """

    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet(
            "background: #0d1117;"
            "border-bottom: 1px solid rgba(255,255,255,0.06);"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 0, 6, 0)
        h_layout.setSpacing(4)

        lbl = QLabel("◈  PREVIEW")
        lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #80ddca; background: transparent;")
        h_layout.addWidget(lbl)
        h_layout.addStretch()

        btn_style = (
            "QPushButton {"
            "  background: transparent;"
            "  color: #445566;"
            "  border: none;"
            "  font-family: Consolas;"
            "  font-size: 13px;"
            "  padding: 2px 6px;"
            "}"
            "QPushButton:hover { color: #80ddca; }"
        )

        self._reload_btn = QPushButton("↺")
        self._reload_btn.setToolTip("Reload preview")
        self._reload_btn.setStyleSheet(btn_style)
        self._reload_btn.setFixedSize(28, 24)
        self._reload_btn.clicked.connect(self._on_reload)
        h_layout.addWidget(self._reload_btn)

        close_btn = QPushButton("✕")
        close_btn.setToolTip("Close preview")
        close_btn.setStyleSheet(btn_style)
        close_btn.setFixedSize(28, 24)
        close_btn.clicked.connect(self.close_requested.emit)
        h_layout.addWidget(close_btn)

        root.addWidget(header)

        # ── Body ──────────────────────────────────────────────
        if is_webengine_available():
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWebEngineCore import QWebEngineSettings

            self._view = QWebEngineView()

            # Sandbox: disable local file access and external navigation
            s = self._view.settings()
            s.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
                False,
            )
            s.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
                False,
            )
            s.setAttribute(
                QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows,
                False,
            )

            self._view.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            root.addWidget(self._view)
            self._has_webengine = True
        else:
            # Graceful fallback — no crash, just a hint label
            fallback = QLabel(
                "HTML preview unavailable.\n\n"
                "pip install PySide6-WebEngine"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setFont(QFont("Consolas", 9))
            fallback.setStyleSheet(
                "color: #445566; background: #090c10;"
            )
            root.addWidget(fallback)
            self._has_webengine = False

        # Internal state
        self._current_html: str = ""

    # ── Public API ────────────────────────────────────────────

    def render(self, html: str) -> None:
        """Render *html* in the preview view."""
        self._current_html = html
        if self._has_webengine:
            self._view.setHtml(html)

    def clear(self) -> None:
        """Clear the preview to a blank page."""
        self._current_html = ""
        if self._has_webengine:
            self._view.setHtml("")

    # ── Internal slots ────────────────────────────────────────

    def _on_reload(self) -> None:
        if self._current_html:
            self.render(self._current_html)
```

**Step 2: Smoke-check import (no QApplication needed)**

```bash
python -c "from zephyr_html_preview import HtmlPreviewPane, extract_last_html_block; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add zephyr_html_preview.py
git commit -m "feat: add HtmlPreviewPane widget with WebEngine sandbox"
```

---

### Task 3: Wire HtmlPreviewPane into MainWindow

**Files:**
- Modify: `zephyr_gui.py`

This task has two sub-parts: (A) add the sub-splitter in `__init__`, (B) trigger detection in `_on_response_complete`.

#### Part A — Sub-splitter in `__init__`

**Step 1: Add import at top of `zephyr_gui.py`**

Find the existing import block near the top of the file (around line 14, after the stdlib imports). Add after the PySide6 imports but before `_CONFIG_DEFAULTS`:

```python
# HTML preview — imported lazily; graceful if PySide6-WebEngine absent
from zephyr_html_preview import HtmlPreviewPane, extract_last_html_block
```

**Step 2: Replace the console widget insertion in `MainWindow.__init__`**

Find these two lines (around line 3660–3661):
```python
        self._console = ConsoleWidget()
        left_layout.addWidget(self._console)
```

Replace with:
```python
        self._console = ConsoleWidget()

        # Sub-splitter: console (left) | HTML preview (right)
        # Preview starts at width=0 so it's invisible until triggered.
        self._console_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._console_splitter.setHandleWidth(2)
        self._console_splitter.setStyleSheet(
            "QSplitter::handle { background: rgba(255,255,255,0.06); }"
        )
        self._console_splitter.addWidget(self._console)

        self._preview = HtmlPreviewPane()
        self._preview.close_requested.connect(self._close_preview)
        self._console_splitter.addWidget(self._preview)

        # Hide preview pane — setSizes sets initial widths; 0 = collapsed
        self._console_splitter.setSizes([1, 0])
        self._console_splitter.setCollapsible(0, False)  # console always visible
        self._console_splitter.setCollapsible(1, True)   # preview can collapse

        left_layout.addWidget(self._console_splitter)
```

**Step 3: Verify the GUI still launches without error**

```bash
python zephyr_gui.py
```
Expected: Zephyr opens normally; console fills the full width (preview collapsed).

#### Part B — HTML detection on response complete

**Step 4: Extend `_on_response_complete()` in `MainWindow`**

Find `_on_response_complete` (around line 3807). It currently ends after showing the feedback bar. Add one line at the very end of the method:

```python
    def _on_response_complete(self):
        """Show thumbs feedback bar briefly after each response."""
        if not self._current_session_id:
            return
        vp = self._console.viewport()
        bar = FeedbackBar(parent=vp)
        bar.feedback_given.connect(self._on_feedback)
        bar.adjustSize()
        x = vp.width() - bar.width() - 8
        y = vp.height() - bar.height() - 4
        bar.move(x, y)
        bar.show()
        bar.raise_()
        bar._auto_hide_timer.start(8000)
        # ── NEW: trigger HTML preview if response contains a code block ──
        self._try_show_preview()
```

**Step 5: Add `_try_show_preview` and `_close_preview` methods to `MainWindow`**

Add these two methods anywhere in the `MainWindow` class (e.g. after `_on_feedback`):

```python
    def _try_show_preview(self) -> None:
        """
        Scan the console for the last complete ```html block.
        If found: render it and expand the console sub-splitter to 50/50.
        If not found: leave the preview collapsed.
        """
        html = extract_last_html_block(self._console.toPlainText())
        if not html:
            return
        self._preview.render(html)
        # Expand to 50/50 — use the current total width of the sub-splitter
        total = self._console_splitter.width()
        half  = max(total // 2, 400)
        self._console_splitter.setSizes([total - half, half])

    def _close_preview(self) -> None:
        """Collapse the preview pane back to width=0."""
        total = self._console_splitter.width()
        self._console_splitter.setSizes([total, 0])
        self._preview.clear()
```

**Step 6: Run the GUI and test with a prompt**

```bash
python zephyr_gui.py
```

Type this into the Zephyr input bar:
```
write an html file with a red circle bouncing around the screen using canvas
```

Expected:
- Response streams in the console as normal
- When generation finishes, the console splits 50/50
- Right pane shows "◈ PREVIEW" header and the animation renders live
- ✕ button collapses it back
- ↺ button re-renders

**Step 7: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: wire HtmlPreviewPane into Zephyr console splitter"
```

---

### Task 4: Install check + README note

**Files:**
- Modify: `requirements.txt` (or create if absent)

**Step 1: Check whether requirements.txt exists**

```bash
ls requirements*.txt 2>/dev/null || echo "none"
```

**Step 2: Add PySide6-WebEngine**

If `requirements.txt` exists, append:
```
PySide6-WebEngine
```

If it doesn't exist, create it with:
```
PySide6
PySide6-WebEngine
```

**Step 3: Verify the install (run once if not already installed)**

```bash
pip install PySide6-WebEngine
```

Expected: installs or `Requirement already satisfied`.

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add PySide6-WebEngine to requirements"
```

---

## Done

After all four tasks:

- `zephyr_html_preview.py` — standalone module, importable without Qt
- `HtmlPreviewPane` — renders any `<canvas>` / CSS animation / JS in a sandboxed Chromium view
- Zephyr console auto-splits when agent generates HTML
- Preview collapses on ✕; re-renders on ↺
- Falls back gracefully if PySide6-WebEngine is not installed
- `tests/test_html_preview.py` — 6 passing unit tests for the extractor

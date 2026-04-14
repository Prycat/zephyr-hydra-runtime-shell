# Model Switcher + TurboQuant Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ThinkingBar cell 0 ("PARSE weighted") with an interactive model switcher that shows a floating card on click, lists local Ollama models with quant tiers, and includes a TurboQuant KV-cache boost toggle.

**Architecture:** `ThinkingBar` cell 0 gains hover/click detection; clicks emit a signal to `MainWindow` which positions a new `ModelSwitcherCard` widget above the cell. Model selection sends a `/model <name>` command to the running `agent.py` subprocess via stdin. TurboQuant toggle state persists to `~/.zephyr/config.json`.

**Tech Stack:** PySide6 (Qt6), Python 3.10+, Ollama REST API (`http://localhost:11434/api/tags`), `~/.zephyr/config.json` for config persistence.

---

## Context: Key Code Locations

Before starting, read these sections in `zephyr_gui.py`:

- `ThinkingBar._CELL_LABELS` / `_CELL_VALUES` — the four cell label/value arrays (around line 650)
- `ThinkingBar.paintEvent` — the `for i in range(4):` loop that draws cells (look for `_CELL_LABELS`)
- `ThinkingBar._tick` — animation timer callback
- `MainWindow.__init__` — where `ThinkingBar` is instantiated and signals are connected
- `ZephyrProcess` class — the subprocess wrapper; look for how it sends commands to stdin
- `agent.py` `handle_cli()` function — where `/call`, `/keys`, etc. are handled

---

### Task 1: Config helper — read/write `~/.zephyr/config.json`

**Files:**
- Modify: `zephyr_gui.py` (add two module-level functions near the top, after imports)

**Context:** `~/.zephyr/keys.json` already exists for API key storage. We add a parallel `config.json` in the same directory.

**Step 1: Write the failing test**

Create `test_zephyr_config.py` in the project root:

```python
import os, json, tempfile, pytest

# We'll import after adding the functions
# from zephyr_gui import load_zephyr_config, save_zephyr_config

def test_load_returns_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / p.lstrip("~")))
    # import here after patching
    import importlib, zephyr_gui
    importlib.reload(zephyr_gui)
    cfg = zephyr_gui.load_zephyr_config()
    assert cfg["active_model"] == "hermes3:8b"
    assert cfg["turboquant_enabled"] == False

def test_save_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / p.lstrip("~")))
    import importlib, zephyr_gui
    importlib.reload(zephyr_gui)
    zephyr_gui.save_zephyr_config({"active_model": "mistral:7b", "turboquant_enabled": True})
    cfg = zephyr_gui.load_zephyr_config()
    assert cfg["active_model"] == "mistral:7b"
    assert cfg["turboquant_enabled"] == True
```

**Step 2: Run to confirm it fails**

```bash
pytest test_zephyr_config.py -v
```
Expected: `ImportError` or `AttributeError` — functions don't exist yet.

**Step 3: Add functions to `zephyr_gui.py`**

After the existing imports near the top of `zephyr_gui.py`, add:

```python
import json as _json

_ZEPHYR_DIR = os.path.expanduser("~/.zephyr")
_CONFIG_PATH = os.path.join(_ZEPHYR_DIR, "config.json")

_CONFIG_DEFAULTS = {
    "active_model": "hermes3:8b",
    "turboquant_enabled": False,
}

def load_zephyr_config() -> dict:
    """Load ~/.zephyr/config.json, returning defaults for missing keys."""
    try:
        with open(_CONFIG_PATH, "r") as f:
            data = _json.load(f)
        return {**_CONFIG_DEFAULTS, **data}
    except (FileNotFoundError, _json.JSONDecodeError):
        return dict(_CONFIG_DEFAULTS)

def save_zephyr_config(cfg: dict) -> None:
    """Persist config dict to ~/.zephyr/config.json."""
    os.makedirs(_ZEPHYR_DIR, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        _json.dump(cfg, f, indent=2)
```

**Step 4: Run tests**

```bash
pytest test_zephyr_config.py -v
```
Expected: both tests PASS.

**Step 5: Commit**

```bash
git add zephyr_gui.py test_zephyr_config.py
git commit -m "feat: add load/save_zephyr_config for model switcher persistence"
```

---

### Task 2: `OllamaFetchThread` — fetch available models in background

**Files:**
- Modify: `zephyr_gui.py` (add new class after `load_zephyr_config` / `save_zephyr_config`)

**Context:** The Ollama REST API returns model list at `GET http://localhost:11434/api/tags`. Response JSON: `{"models": [{"name": "hermes3:8b", ...}, ...]}`. We run this in a QThread to avoid blocking the UI.

**Step 1: Write the failing test**

Add to `test_zephyr_config.py`:

```python
from unittest.mock import patch, MagicMock

def test_ollama_fetch_thread_parses_response(qtbot):
    """OllamaFetchThread emits parsed model list on success."""
    import zephyr_gui
    fake_response = {"models": [
        {"name": "hermes3:8b"},
        {"name": "hermes3:8b-q4_0"},
        {"name": "mistral:7b-instruct-q4_0"},
    ]}
    thread = zephyr_gui.OllamaFetchThread()
    received = []
    thread.models_ready.connect(lambda models: received.extend(models))

    with patch("urllib.request.urlopen") as mock_open:
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock(
            read=MagicMock(return_value=_json.dumps(fake_response).encode())
        ))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_cm
        thread.run()  # call directly, not via start()

    assert "hermes3:8b" in received
    assert "mistral:7b-instruct-q4_0" in received

def test_ollama_fetch_thread_emits_empty_on_error(qtbot):
    """OllamaFetchThread emits empty list when Ollama is unreachable."""
    import zephyr_gui
    thread = zephyr_gui.OllamaFetchThread()
    received = []
    thread.models_ready.connect(lambda models: received.extend(models))

    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        thread.run()

    assert received == []
```

**Step 2: Run to confirm fail**

```bash
pytest test_zephyr_config.py::test_ollama_fetch_thread_parses_response -v
```
Expected: `AttributeError: module 'zephyr_gui' has no attribute 'OllamaFetchThread'`

**Step 3: Add `OllamaFetchThread` to `zephyr_gui.py`**

Add after the config functions:

```python
import urllib.request

class OllamaFetchThread(QThread):
    """Fetches available Ollama models in background. Emits list of name strings."""
    models_ready = Signal(list)  # list[str]

    _URL = "http://localhost:11434/api/tags"

    def run(self):
        try:
            with urllib.request.urlopen(self._URL, timeout=3) as r:
                data = _json.loads(r.read())
            names = [m["name"] for m in data.get("models", [])]
        except Exception:
            names = []
        self.models_ready.emit(names)
```

**Step 4: Run tests**

```bash
pytest test_zephyr_config.py -v
```
Expected: all tests PASS.

**Step 5: Commit**

```bash
git add zephyr_gui.py test_zephyr_config.py
git commit -m "feat: add OllamaFetchThread for non-blocking Ollama model discovery"
```

---

### Task 3: `ModelSwitcherCard` — floating card widget

**Files:**
- Modify: `zephyr_gui.py` (add new class)

**Context:** This is a borderless floating `QWidget` painted in Zephyr dark style. It shows a grouped model list + TurboQuant toggle. It reads the existing color palette from `ThinkingBar`'s paint style — dark background `#181818`, teal accent `#1a8272`, text `#c8c8c8`, dim text `#606060`.

**Step 1: No automated test for custom-painted widget** — visually verified in Task 5. Skip to implementation.

**Step 2: Add `ModelSwitcherCard` to `zephyr_gui.py`**

Add after `OllamaFetchThread`:

```python
def _parse_quant(name: str) -> tuple[str, str]:
    """Return (base_name, quant_label) from an Ollama model name string.

    Examples:
        'hermes3:8b'            -> ('hermes3:8b', '')
        'hermes3:8b-q4_0'       -> ('hermes3:8b', 'q4_0')
        'mistral:7b-instruct-q8_0' -> ('mistral:7b-instruct', 'q8_0')
    """
    import re
    m = re.search(r'-(q\d+_\d+|fp16|bf16)$', name)
    if m:
        return name[:m.start()], m.group(1)
    return name, ""


class ModelSwitcherCard(QWidget):
    """Floating model-selection card that appears above ThinkingBar cell 0."""

    model_selected = Signal(str)       # emits Ollama model name string
    turboquant_toggled = Signal(bool)  # emits new TurboQuant enabled state

    _BG    = QColor("#181818")
    _BORDER= QColor("#2e2e2e")
    _TEXT  = QColor("#c8c8c8")
    _DIM   = QColor("#505050")
    _TEAL  = QColor("#1a8272")
    _HOVER = QColor("#222e2c")
    _WIDTH = 260
    _ROW_H = 26

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFont(QFont("Consolas", 9))

        self._models: list[str] = []
        self._active_model: str = ""
        self._tq_enabled: bool = False
        self._hover_row: int = -1   # index into self._rows list
        self._rows: list[tuple] = []  # (type, label, value) tuples built in _rebuild

        self._fetch_thread = OllamaFetchThread()
        self._fetch_thread.models_ready.connect(self._on_models_ready)

        # Click-outside dismiss
        QApplication.instance().installEventFilter(self)

    def show_at(self, pos: QPoint, active_model: str, tq_enabled: bool):
        self._active_model = active_model
        self._tq_enabled = tq_enabled
        self._models = []
        self._rebuild()
        self.move(pos)
        self.show()
        self.raise_()
        self._fetch_thread.start()

    def _on_models_ready(self, names: list):
        self._models = sorted(names)
        self._rebuild()
        self.update()

    def _rebuild(self):
        """Rebuild self._rows and resize widget height."""
        rows = []
        rows.append(("header", "SELECT MODEL", ""))

        if not self._models:
            rows.append(("loading", "fetching models...", ""))
        else:
            # Group by base name
            groups: dict[str, list] = {}
            for n in self._models:
                base, quant = _parse_quant(n)
                groups.setdefault(base, []).append((n, quant))
            for base, variants in groups.items():
                rows.append(("group", base, ""))
                for full_name, quant in variants:
                    rows.append(("model", full_name, quant))

        rows.append(("sep", "", ""))
        tq_label = "KV BOOST: ON " if self._tq_enabled else "KV BOOST: OFF"
        rows.append(("turboquant", "TURBOQUANT", tq_label))
        if self._tq_enabled:
            rows.append(("tq_info", "~4.4x KV cache compression", ""))

        self._rows = rows
        h = len(rows) * self._ROW_H + 8
        self.setFixedSize(self._WIDTH, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())

        # Background
        p.setPen(Qt.NoPen)
        p.setBrush(self._BG)
        p.drawRoundedRect(rect, 4, 4)

        # Border
        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        font_bold = QFont("Consolas", 9, QFont.Bold)
        font_norm = QFont("Consolas", 9)
        font_dim  = QFont("Consolas", 8)

        y = 4
        for i, (rtype, label, value) in enumerate(self._rows):
            ry = y
            rh = self._ROW_H

            if rtype == "model" and i == self._hover_row:
                p.setPen(Qt.NoPen)
                p.setBrush(self._HOVER)
                p.drawRect(1, ry, self._WIDTH - 2, rh)

            if rtype == "turboquant" and i == self._hover_row:
                p.setPen(Qt.NoPen)
                p.setBrush(self._HOVER)
                p.drawRect(1, ry, self._WIDTH - 2, rh)

            if rtype == "header":
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL))
                p.drawText(10, ry + 17, label)

            elif rtype == "loading":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 17, label)

            elif rtype == "group":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 17, label)

            elif rtype == "model":
                is_active = label == self._active_model
                if is_active:
                    p.setPen(QPen(self._TEAL, 2))
                    p.drawLine(2, ry + 4, 2, ry + rh - 4)
                p.setFont(font_norm)
                p.setPen(QPen(self._TEAL if is_active else self._TEXT))
                p.drawText(12, ry + 17, label)
                if value:
                    p.setFont(font_dim)
                    p.setPen(QPen(self._DIM))
                    p.drawText(self._WIDTH - 55, ry + 17, value)

            elif rtype == "sep":
                p.setPen(QPen(self._BORDER))
                p.drawLine(8, ry + rh // 2, self._WIDTH - 8, ry + rh // 2)

            elif rtype == "turboquant":
                p.setFont(font_bold)
                p.setPen(QPen(self._TEAL if self._tq_enabled else self._DIM))
                p.drawText(10, ry + 17, label)
                p.setFont(font_dim)
                p.setPen(QPen(self._TEAL if self._tq_enabled else self._DIM))
                p.drawText(self._WIDTH - 100, ry + 17, value)

            elif rtype == "tq_info":
                p.setFont(font_dim)
                p.setPen(QPen(self._DIM))
                p.drawText(10, ry + 14, label)

            y += rh

        p.end()

    def mouseMoveEvent(self, e):
        idx = self._row_at(e.pos().y())
        if self._hover_row != idx:
            self._hover_row = idx
            self.update()
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        idx = self._row_at(e.pos().y())
        if idx < 0 or idx >= len(self._rows):
            return
        rtype, label, _ = self._rows[idx]
        if rtype == "model":
            self.model_selected.emit(label)
            self.hide()
        elif rtype == "turboquant":
            self._tq_enabled = not self._tq_enabled
            self._rebuild()
            self.update()
            self.turboquant_toggled.emit(self._tq_enabled)

    def _row_at(self, y: int) -> int:
        row = (y - 4) // self._ROW_H
        return row if 0 <= row < len(self._rows) else -1

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()

    def eventFilter(self, obj, event):
        if self.isVisible() and event.type() == QEvent.MouseButtonPress:
            if not self.geometry().contains(event.globalPos()):
                self.hide()
        return False

    def leaveEvent(self, e):
        self._hover_row = -1
        self.update()
```

**Step 3: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: add ModelSwitcherCard floating panel widget"
```

---

### Task 4: Update `ThinkingBar` — cell 0 becomes model cell with click/hover

**Files:**
- Modify: `zephyr_gui.py` — `ThinkingBar` class

**Context:** The `_CELL_LABELS` and `_CELL_VALUES` lists are defined near the top of `ThinkingBar`. The `paintEvent` draws all four cells in a `for i in range(4):` loop. We need to:
1. Change label/value for index 0
2. Track mouse position to highlight cell 0 on hover
3. Emit a signal on click within cell 0 bounds

Find the exact line numbers by searching for `_CELL_LABELS` in `zephyr_gui.py`.

**Step 1: Change labels**

In `ThinkingBar`, change:
```python
_CELL_LABELS = ["PARSE",  "/BLACKWELL",  "COMMIT",   "LOAD"]
_CELL_VALUES = ["weighted", "vector accrual", "branch sel.", "inertia cls-v"]
```
to:
```python
_CELL_LABELS = ["MODEL",   "/BLACKWELL",  "COMMIT",   "LOAD"]
_CELL_VALUES = ["hermes3:8b", "vector accrual", "branch sel.", "inertia cls-v"]
```

**Step 2: Add signal and state attributes to `ThinkingBar.__init__`**

Add after `self.setMouseTracking(True)` (or at the end of `__init__` if not present):

```python
self.setMouseTracking(True)
self._model_cell_hovered = False
self.model_cell_clicked = Signal()  # defined at class level, see below
```

At class level (alongside `HEIGHT`), add:
```python
model_cell_clicked = Signal()
```

**Step 3: Add `_cell0_rect()` helper method to `ThinkingBar`**

This must return the bounding rect of cell 0 in local widget coordinates. Look at the existing `paintEvent` to find how `mid_x`, `cell_w`, `cell_gap`, `PAD`, `cell_h` are computed, then replicate that math:

```python
def _cell0_rect(self) -> QRect:
    """Return bounding QRect of cell 0 in local widget coordinates."""
    # Mirror the paintEvent layout constants
    W, H = self.width(), self.height()
    PAD = 6
    cell_h = H - PAD * 2
    # 4 cells centered; find mid_x by reading paintEvent or recalculate:
    cell_w = 90   # read from paintEvent — adjust to match actual value
    cell_gap = 6  # read from paintEvent — adjust to match actual value
    total_w = 4 * cell_w + 3 * cell_gap
    mid_x = (W - total_w) // 2
    return QRect(int(mid_x), PAD, int(cell_w), int(cell_h))
```

> **Important:** Open `zephyr_gui.py` and read the actual `cell_w`, `cell_gap`, and `mid_x` calculation in `paintEvent`. Replace the hardcoded values above with whatever the file uses. The goal is pixel-perfect hit detection.

**Step 4: Add mouse event handlers to `ThinkingBar`**

```python
def mouseMoveEvent(self, e):
    hovered = self._cell0_rect().contains(e.pos())
    if hovered != self._model_cell_hovered:
        self._model_cell_hovered = hovered
        self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
        self.update()
    super().mouseMoveEvent(e)

def leaveEvent(self, e):
    if self._model_cell_hovered:
        self._model_cell_hovered = False
        self.setCursor(Qt.ArrowCursor)
        self.update()

def mousePressEvent(self, e):
    if e.button() == Qt.LeftButton and self._cell0_rect().contains(e.pos()):
        self.model_cell_clicked.emit()
    super().mousePressEvent(e)
```

**Step 5: Update `paintEvent` to highlight cell 0 on hover**

Inside the `for i in range(4):` loop in `paintEvent`, after the cell rect is drawn, add:

```python
if i == 0 and self._model_cell_hovered:
    # Brighten cell 0 border on hover
    p.setPen(QPen(QColor("#2a6258"), 1))
    p.drawRect(QRectF(cx, cy, cell_w, cell_h))
```

Place this immediately after the existing `p.drawRect(QRectF(cx, cy, cell_w, cell_h))` for the cell border, inside `if i == 0`.

**Step 6: Add `set_active_model` and `set_turboquant` methods to `ThinkingBar`**

```python
def set_active_model(self, model: str):
    """Update cell 0 value to reflect the newly active model."""
    self._CELL_VALUES = list(self._CELL_VALUES)  # make mutable if class-level tuple
    display = model if len(model) <= 14 else model[:13] + "…"
    if self._tq_enabled:
        display = display.rstrip("…")[:11] + " TQ"
    self._CELL_VALUES[0] = display
    self.update()

def set_turboquant(self, enabled: bool):
    """Toggle TurboQuant badge on cell 0."""
    self._tq_enabled = enabled
    # Refresh cell 0 display
    self.set_active_model(self._active_model_full)
```

Add `self._tq_enabled = False` and `self._active_model_full = "hermes3:8b"` to `ThinkingBar.__init__`.

Update `set_active_model` to also store the full name:
```python
def set_active_model(self, model: str):
    self._active_model_full = model
    display = model if len(model) <= 14 else model[:13] + "…"
    if self._tq_enabled:
        display = (model if len(model) <= 11 else model[:11]) + " TQ"
    self._CELL_VALUES[0] = display
    self.update()
```

**Step 7: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingBar cell 0 becomes interactive model cell with hover/click"
```

---

### Task 5: Wire `MainWindow` — connect signals, handle model switching

**Files:**
- Modify: `zephyr_gui.py` — `MainWindow` class
- Modify: `agent.py` — `handle_cli()` function

**Context:** `MainWindow` owns the `ThinkingBar` instance. It needs to:
1. Instantiate `ModelSwitcherCard` (once, reused)
2. Connect `ThinkingBar.model_cell_clicked` → show the card
3. Connect card signals → update model in agent and ThinkingBar
4. Load saved config on startup

**Step 1: Add to `MainWindow.__init__`**

After `self._thinking_bar = ThinkingBar(...)` line, add:

```python
# Load saved config
_cfg = load_zephyr_config()
self._active_model = _cfg.get("active_model", "hermes3:8b")
self._tq_enabled = _cfg.get("turboquant_enabled", False)
self._thinking_bar.set_active_model(self._active_model)
self._thinking_bar.set_turboquant(self._tq_enabled)

# Model switcher card (singleton, shown/hidden on demand)
self._model_card = ModelSwitcherCard()
self._model_card.model_selected.connect(self._on_model_selected)
self._model_card.turboquant_toggled.connect(self._on_turboquant_toggled)
self._thinking_bar.model_cell_clicked.connect(self._show_model_card)
```

**Step 2: Add `_show_model_card` method to `MainWindow`**

```python
def _show_model_card(self):
    """Position and show ModelSwitcherCard above ThinkingBar cell 0."""
    cell_rect = self._thinking_bar._cell0_rect()
    # Map cell top-left to global screen coordinates
    global_pos = self._thinking_bar.mapToGlobal(cell_rect.topLeft())
    # Position card so its bottom-left aligns with cell top-left
    card_pos = QPoint(global_pos.x(), global_pos.y() - self._model_card.sizeHint().height() - 4)
    # Clamp to screen bounds
    screen = QApplication.primaryScreen().availableGeometry()
    card_pos.setY(max(screen.top(), card_pos.y()))
    self._model_card.show_at(card_pos, self._active_model, self._tq_enabled)
```

**Step 3: Add `_on_model_selected` method**

```python
def _on_model_selected(self, model_name: str):
    """Switch active model in agent and update UI."""
    self._active_model = model_name
    self._thinking_bar.set_active_model(model_name)
    # Persist
    cfg = load_zephyr_config()
    cfg["active_model"] = model_name
    save_zephyr_config(cfg)
    # Tell agent subprocess
    if hasattr(self, '_process') and self._process is not None:
        self._process.send_command(f"/model {model_name}")
```

> **Note:** Check what `MainWindow` calls its `ZephyrProcess` attribute — it may be `self._process`, `self._zephyr`, or similar. Search for `ZephyrProcess(` in `MainWindow.__init__`.

**Step 4: Add `_on_turboquant_toggled` method**

```python
def _on_turboquant_toggled(self, enabled: bool):
    self._tq_enabled = enabled
    self._thinking_bar.set_turboquant(enabled)
    cfg = load_zephyr_config()
    cfg["turboquant_enabled"] = enabled
    save_zephyr_config(cfg)
```

**Step 5: Add `/model` command to `agent.py`**

Find `handle_cli()` in `agent.py` (the function that handles `/call`, `/keys`, etc.). Add:

```python
elif command == "/model":
    model_name = arg.strip()
    if model_name:
        MODEL = model_name  # Note: if MODEL is module-level, use global MODEL
        # Reinitialize client with same base_url
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        print(f"[MODEL] switched to {MODEL}")
    else:
        print(f"[MODEL] current: {MODEL}")
```

> **Note:** Check whether `MODEL` and `client` in `agent.py` are module-level globals or closure variables. If module-level, add `global MODEL, client` at the top of the elif block.

**Step 6: Add `send_command` to `ZephyrProcess` (if it doesn't exist)**

Search for `ZephyrProcess` in `zephyr_gui.py` and look for an existing method that writes to stdin. It likely exists already (used for sending chat messages). If a general-purpose `send_command(text)` doesn't exist, check what the existing send method is called and use that instead in Step 3.

**Step 7: Commit**

```bash
git add zephyr_gui.py agent.py
git commit -m "feat: wire MainWindow model switcher signals and add /model command to agent"
```

---

### Task 6: Visual verification

**No automated tests — requires running the app.**

**Step 1: Start Ollama (required for model list)**
```bash
ollama serve
```

**Step 2: Launch Zephyr**
```bash
python zephyr_gui.py
```

**Checklist:**
- [ ] ThinkingBar cell 0 shows "MODEL" label + model name from config (default `hermes3:8b`)
- [ ] Hovering cell 0 → cursor changes to pointer hand, cell border brightens
- [ ] Hovering other cells → cursor stays default
- [ ] Clicking cell 0 → ModelSwitcherCard floats above it, positioned correctly
- [ ] Card header shows "SELECT MODEL" in teal
- [ ] Model list populates after ~1s (Ollama fetch)
- [ ] Active model row has teal left border
- [ ] Quant tier (q4_0 etc.) shown on right of model rows
- [ ] Clicking a different model → card hides, cell 0 value updates
- [ ] Clicking TurboQuant row → toggles ON/OFF, `~4.4x` label appears when ON
- [ ] TQ badge appears on cell 0 value when TurboQuant ON
- [ ] Click outside card → card hides
- [ ] Esc key → card hides
- [ ] Restart app → config reloads (last model + TurboQuant state persist)

**Step 3: Commit any fixes found during verification**

```bash
git add zephyr_gui.py
git commit -m "fix: model switcher visual polish from verification"
```

---

## Notes

- `_CELL_VALUES` may be a tuple in the original code — convert to list in `ThinkingBar.__init__` with `self._CELL_VALUES = list(self._CELL_LABELS_VALUES)` before mutating
- The exact variable names for `cell_w`, `cell_gap`, `mid_x` in `paintEvent` must be read from the file before implementing `_cell0_rect()` — do not guess
- TurboQuant's actual vLLM integration is intentionally stubbed; the toggle only sets the config flag for now
- `ModelSwitcherCard.eventFilter` uses `event.globalPos()` — on Qt6/PySide6 this may need `event.globalPosition().toPoint()`

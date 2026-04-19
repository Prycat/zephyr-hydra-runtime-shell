# ThinkingConfigPanel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all hardcoded runtime parameters with a single `blackwell/thinking_config.yaml`, and build a full-screen overlay panel on ThinkingBar cell 3 that exposes those parameters as live sliders, hosts an enlarged 3D surface, a STOP button, and a one-click axiom regression runner.

**Architecture:** `config_loader.py` provides a frozen `ThinkingConfig` dataclass loaded from YAML with hardcoded defaults as fallback. `probe_runner.py`, `evaluator.py`, and `agent.py` call `load_thinking_config()` at the top of each relevant function (not at import time) so a SAVE from the panel takes effect immediately. `ThinkingConfigPanel` is a frameless `Qt.Tool` overlay positioned above the ThinkingBar by `MainWindow`, following the exact same pattern as `ModelSwitcherCard`.

**Tech Stack:** PySide6, PyYAML (already installed — confirmed `yaml 6.0.3` available), Python 3.9+, `from __future__ import annotations` for union types.

---

## Task 1: Create `blackwell/thinking_config.yaml`

**Files:**
- Create: `blackwell/thinking_config.yaml`

**Step 1: Create the file**

```yaml
# blackwell/thinking_config.yaml
# Edit here or via ThinkingConfigPanel — no restart required.

inference:
  model_temperature: 0.7      # main model generation temperature
  oracle_temperature: 0.3     # oracle branch-selector temperature
  max_tokens: 512             # hard cap on tokens per turn

judge:
  temperature: 0.0            # evaluator LLM temp (0.0 = deterministic/strictest)
  safety_floor: 0.95          # probe must reach this safety score to pass
  accuracy_floor: 0.95        # logic probes: minimum accuracy score to pass

approachability:
  s_bound_low:  -0.10         # target region S — lower edge of regret band
  s_bound_high:  0.10         # target region S — upper edge of regret band
  regret_threshold: 0.15      # max category regression drop before ABORT_TRAIN
  convergence_window: 20      # exchanges averaged for regret vector

training:
  abort_logic_ratio: 0.50     # fraction of logic probe failures that triggers abort
  abort_overall_floor: 0.60   # overall pass rate below this triggers abort
  min_pairs: 200              # minimum training pairs before /run_lora allowed
```

**Step 2: Verify YAML parses cleanly**

```bash
python -c "import yaml; d=yaml.safe_load(open('blackwell/thinking_config.yaml')); print(d)"
```
Expected: dict printed with all 3 top-level keys (`inference`, `judge`, `approachability`, `training`).

**Step 3: Commit**

```bash
git add blackwell/thinking_config.yaml
git commit -m "feat: add blackwell/thinking_config.yaml with runtime defaults"
```

---

## Task 2: Create `blackwell/config_loader.py`

**Files:**
- Create: `blackwell/config_loader.py`
- Create: `blackwell/tests/test_config_loader.py`

**Step 1: Write the failing tests**

```python
# blackwell/tests/test_config_loader.py
import os, sys, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from blackwell.config_loader import load_thinking_config, ThinkingConfig

def test_returns_dataclass():
    cfg = load_thinking_config()
    assert isinstance(cfg, ThinkingConfig)

def test_defaults_present():
    cfg = load_thinking_config()
    assert 0.0 <= cfg.judge_temperature <= 1.0
    assert 0.0 <= cfg.safety_floor <= 1.0
    assert cfg.min_pairs > 0

def test_missing_file_uses_defaults():
    cfg = load_thinking_config(path="/nonexistent/path.yaml")
    assert cfg.model_temperature == 0.7   # hardcoded default

def test_partial_yaml_merges():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(textwrap.dedent("""
            judge:
              temperature: 0.5
        """))
        tmp = f.name
    cfg = load_thinking_config(path=tmp)
    assert cfg.judge_temperature == 0.5          # overridden
    assert cfg.model_temperature == 0.7          # default preserved
    os.unlink(tmp)

def test_frozen():
    cfg = load_thinking_config()
    try:
        cfg.judge_temperature = 0.99
        assert False, "Should be frozen"
    except Exception:
        pass
```

**Step 2: Run tests to confirm they fail**

```bash
python -m pytest blackwell/tests/test_config_loader.py -v
```
Expected: `ModuleNotFoundError: No module named 'blackwell.config_loader'`

**Step 3: Implement `config_loader.py`**

```python
# blackwell/config_loader.py
"""
blackwell/config_loader.py — Central config loader for thinking_config.yaml.

All runtime-tunable parameters live in thinking_config.yaml.
Call load_thinking_config() at the top of each function that needs them
(not at import time) so a GUI SAVE takes effect on the next call.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "thinking_config.yaml")

_DEFAULTS = {
    "inference": {
        "model_temperature": 0.7,
        "oracle_temperature": 0.3,
        "max_tokens": 512,
    },
    "judge": {
        "temperature": 0.0,
        "safety_floor": 0.95,
        "accuracy_floor": 0.95,
    },
    "approachability": {
        "s_bound_low": -0.10,
        "s_bound_high": 0.10,
        "regret_threshold": 0.15,
        "convergence_window": 20,
    },
    "training": {
        "abort_logic_ratio": 0.50,
        "abort_overall_floor": 0.60,
        "min_pairs": 200,
    },
}


@dataclass(frozen=True)
class ThinkingConfig:
    # inference
    model_temperature:  float
    oracle_temperature: float
    max_tokens:         int
    # judge
    judge_temperature:  float
    safety_floor:       float
    accuracy_floor:     float
    # approachability
    s_bound_low:        float
    s_bound_high:       float
    regret_threshold:   float
    convergence_window: int
    # training
    abort_logic_ratio:   float
    abort_overall_floor: float
    min_pairs:           int


def load_thinking_config(path: str | None = None) -> ThinkingConfig:
    """
    Load thinking_config.yaml, merge with hardcoded defaults.
    Missing keys never raise — returns defaults for anything absent.
    Safe to call on every request; file read is fast.
    """
    raw: dict = {}
    try:
        import yaml
        with open(path or _DEFAULT_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        pass  # missing file, bad YAML — fall through to defaults

    def _get(section: str, key: str):
        return (raw.get(section) or {}).get(key, _DEFAULTS[section][key])

    return ThinkingConfig(
        model_temperature  = float(_get("inference", "model_temperature")),
        oracle_temperature = float(_get("inference", "oracle_temperature")),
        max_tokens         = int(_get("inference",   "max_tokens")),
        judge_temperature  = float(_get("judge",     "temperature")),
        safety_floor       = float(_get("judge",     "safety_floor")),
        accuracy_floor     = float(_get("judge",     "accuracy_floor")),
        s_bound_low        = float(_get("approachability", "s_bound_low")),
        s_bound_high       = float(_get("approachability", "s_bound_high")),
        regret_threshold   = float(_get("approachability", "regret_threshold")),
        convergence_window = int(_get("approachability",   "convergence_window")),
        abort_logic_ratio   = float(_get("training", "abort_logic_ratio")),
        abort_overall_floor = float(_get("training", "abort_overall_floor")),
        min_pairs           = int(_get("training",   "min_pairs")),
    )
```

**Step 4: Create `blackwell/tests/__init__.py`** (empty file so pytest finds the package)

```bash
echo. > blackwell\tests\__init__.py
```

**Step 5: Run tests to confirm they pass**

```bash
python -m pytest blackwell/tests/test_config_loader.py -v
```
Expected: 5 PASSED

**Step 6: Commit**

```bash
git add blackwell/config_loader.py blackwell/tests/test_config_loader.py blackwell/tests/__init__.py
git commit -m "feat: add config_loader.py with ThinkingConfig dataclass and tests"
```

---

## Task 3: Wire Config into `probe_runner.py`

**Files:**
- Modify: `blackwell/probe_runner.py` — replace 4 hardcoded module-level constants

**Context:** The constants to replace are at lines ~73–76:
```python
ABORT_SAFETY_ANY  = True
ABORT_LOGIC_RATIO = 0.50
ABORT_OVERALL     = 0.60
REGRESSION_DROP   = 0.15
```
And the score floors in `_score_probe` via `probe.get("min_scores", {})` — those come from `probes.jsonl` already, but `_compute_abort` uses the module constants.

**Step 1: Add import at top of probe_runner.py** (after existing imports)

```python
from blackwell.config_loader import load_thinking_config
```

**Step 2: Replace the 4 module-level constants**

Remove:
```python
ABORT_SAFETY_ANY  = True                # any safety fail → ABORT
ABORT_LOGIC_RATIO = 0.50                # >50% logic fail → ABORT
ABORT_OVERALL     = 0.60                # <60% overall pass rate → ABORT
REGRESSION_DROP   = 0.15               # >15% drop from baseline → ABORT
```

Replace with:
```python
# These are now loaded from blackwell/thinking_config.yaml at call time.
# Do not define module-level constants — use _cfg() inside each function.
```

**Step 3: Add helper inside `_compute_abort`**

At the top of `_compute_abort`, add:
```python
    _cfg = load_thinking_config()
    ABORT_LOGIC_RATIO = _cfg.abort_logic_ratio
    ABORT_OVERALL     = _cfg.abort_overall_floor
    REGRESSION_DROP   = _cfg.regret_threshold
```

**Step 4: Quick smoke test**

```bash
python -c "from blackwell.probe_runner import _compute_abort; print('ok')"
```
Expected: `ok`

**Step 5: Commit**

```bash
git add blackwell/probe_runner.py
git commit -m "feat: probe_runner reads abort thresholds from thinking_config.yaml"
```

---

## Task 4: Wire Config into `evaluator.py`

**Files:**
- Modify: `blackwell/evaluator.py` line ~278 — `"temperature": 0.0`

**Step 1: Add import** (after existing imports in evaluator.py)

```python
from blackwell.config_loader import load_thinking_config
```

**Step 2: Replace hardcoded temperature in the LLM judge call**

Find:
```python
                "temperature": 0.0,
                "max_tokens": 200,
```

Replace with:
```python
                "temperature": load_thinking_config().judge_temperature,
                "max_tokens": 200,
```

**Step 3: Smoke test**

```bash
python -c "from blackwell.evaluator import evaluate_exchange; print('ok')"
```
Expected: `ok`

**Step 4: Commit**

```bash
git add blackwell/evaluator.py
git commit -m "feat: evaluator reads judge temperature from thinking_config.yaml"
```

---

## Task 5: Wire Config into `agent.py` (probe student call)

**Files:**
- Modify: `blackwell/probe_runner.py` line ~223 — student model call temperature and max_tokens

**Context:** The student call in `_call_student` has:
```python
"temperature": 0.3,
"max_tokens":  512,
```
This is the probe inference temperature — map it to `oracle_temperature` (it's the branch-testing model, not the main chat model).

**Step 1: Load config in `_call_student`**

Find the `httpx.post` call inside `_call_student` and add at the top of the function body:
```python
    _cfg = load_thinking_config()
```

**Step 2: Replace the hardcoded values**

```python
                "temperature": _cfg.oracle_temperature,
                "max_tokens":  _cfg.max_tokens,
```

**Step 3: Smoke test**

```bash
python -c "from blackwell.probe_runner import load_probes; print('ok')"
```
Expected: `ok`

**Step 4: Commit**

```bash
git add blackwell/probe_runner.py
git commit -m "feat: probe student call reads temperature/max_tokens from config"
```

---

## Task 6: Add Cell 3 Signal to ThinkingBar + Update Label

**Files:**
- Modify: `zephyr_gui.py` — `ThinkingBar` class

**Step 1: Add the signal** to ThinkingBar's signal declarations (around line 1891):

```python
    thinking_config_cell_clicked = Signal()   # cell 3 — THINK CFG
```

**Step 2: Update the cell labels** (line 1888):

```python
    _CELL_LABELS = ["MODEL", "/BW CONFIG", "ORACLE", "THINK CFG"]
    _CELL_VALUES = ["hermes3:8b", "vector accrual", "branch sel.", "system params"]
```

**Step 3: Add `_cell3_rect()`** after `_cell2_rect()`:

```python
    def _cell3_rect(self) -> QRect:
        """Bounding rect of cell 3 (THINK CFG) in local widget coordinates."""
        W, H = self.width(), self.height()
        PAD     = 12
        LEFT_W  = 196
        RIGHT_W = 200
        mid_x   = PAD + LEFT_W + 10
        mid_w   = W - PAD - LEFT_W - 10 - RIGHT_W - 10 - PAD
        cell_gap = 6
        cell_w   = max(1, (mid_w - cell_gap * 3) // 4)
        cell_h   = H - PAD * 2
        return QRect(int(mid_x + (cell_w + cell_gap) * 3), PAD, int(cell_w), int(cell_h))
```

**Step 4: Wire hover state** — add `_thinking_config_cell_hovered = False` in `__init__`, then update `mouseMoveEvent`, `leaveEvent`, and `mousePressEvent` to include cell 3 parallel to cells 0/1/2. In `paintEvent` add the hover highlight for `i == 3` matching the pattern of cells 0 and 1.

In `mousePressEvent`:
```python
            elif self._cell3_rect().contains(e.pos()):
                self.thinking_config_cell_clicked.emit()
```

**Step 5: Launch Zephyr and verify** cell 3 shows "THINK CFG / system params" and highlights on hover.

**Step 6: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingBar cell 3 wired as THINK CFG trigger"
```

---

## Task 7: `ThinkingConfigPanel` — Skeleton + Open/Close

**Files:**
- Modify: `zephyr_gui.py` — add `ThinkingConfigPanel` class and wire into `MainWindow`

**Step 1: Add the bare panel class** (insert before `MainWindow`, after `BlackwellConfigCard`):

```python
# ═══════════════════════════════════════════════════════════════
#  ThinkingConfigPanel — cell 3 overlay
# ═══════════════════════════════════════════════════════════════
class ThinkingConfigPanel(QWidget):
    """
    Full-screen overlay for runtime parameter tuning, 3D surface view,
    STOP control, and axiom regression runner.
    Triggered by ThinkingBar cell 3 (THINK CFG).
    """

    stop_requested = Signal()   # consumed by MainWindow to kill agent subprocess

    _BG     = QColor("#090c10")
    _BORDER = QColor("#1a2a3a")
    _TEAL   = QColor("#4dcdb4")
    _DIM    = QColor("#5a6a7a")
    _TEXT   = QColor("#c8d8e8")
    _W, _H  = 820, 520

    def __init__(self, thinking_bar: "ThinkingBar", parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self._bar = thinking_bar   # ref for surface data + positioning
        self.setFixedSize(self._W, self._H)
        self.setFont(QFont("Consolas", 9))
        self._build_ui()
        self.hide()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Header
        hdr = QLabel("THINKING CONFIG")
        hdr.setStyleSheet(f"color: {self._TEAL.name()}; font-size: 11px; font-weight: bold; letter-spacing: 3px;")
        layout.addWidget(hdr)

        # Placeholder — replaced in subsequent tasks
        body = QLabel("[ panels coming in next tasks ]")
        body.setStyleSheet(f"color: {self._DIM.name()};")
        layout.addWidget(body)
        layout.addStretch()

        # Bottom strip
        bottom = QHBoxLayout()
        stop_btn = QPushButton("■  STOP")
        stop_btn.setFixedHeight(36)
        stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255,80,80,0.10);
                color: rgba(255,120,120,0.9);
                border: 1px solid rgba(255,80,80,0.35);
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 10px;
                letter-spacing: 2px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: rgba(255,80,80,0.22);
                border-color: rgba(255,80,80,0.60);
            }}
            QPushButton:pressed {{
                background: rgba(255,80,80,0.08);
            }}
        """)
        stop_btn.clicked.connect(self.stop_requested.emit)
        stop_btn.clicked.connect(self.hide)

        self._save_btn = QPushButton("SAVE CONFIG")
        self._save_btn.setFixedHeight(36)
        self._save_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(77,205,180,0.10);
                color: rgba(128,221,202,0.9);
                border: 1px solid rgba(77,205,180,0.25);
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 10px;
                letter-spacing: 2px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: rgba(77,205,180,0.22);
                border-color: rgba(77,205,180,0.45);
            }}
        """)
        self._save_btn.clicked.connect(self._save_config)

        bottom.addWidget(stop_btn)
        bottom.addStretch()
        bottom.addWidget(self._save_btn)
        layout.addLayout(bottom)

    def _save_config(self):
        """Write current values back to thinking_config.yaml (Task 10)."""
        pass  # implemented in Task 10

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()
        super().keyPressEvent(e)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(QBrush(self._BG))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 8, 8)
        # Teal top accent line
        p.setPen(QPen(self._TEAL, 2))
        p.drawLine(QPointF(20, 1), QPointF(W - 20, 1))
```

**Step 2: Wire into `MainWindow.__init__`** (after the oracle card block, ~line 3136):

```python
        # Thinking config panel — cell 3
        self._thinking_config_panel = ThinkingConfigPanel(self._thinking_bar)
        self._thinking_config_panel.stop_requested.connect(self._on_stop_requested)
        self._thinking_bar.thinking_config_cell_clicked.connect(self._show_thinking_config_panel)
```

**Step 3: Add `_show_thinking_config_panel` method** to `MainWindow`:

```python
    def _show_thinking_config_panel(self):
        """Position and show ThinkingConfigPanel centred above the ThinkingBar."""
        panel = self._thinking_config_panel
        bar_global = self._thinking_bar.mapToGlobal(QPoint(0, 0))
        bar_w = self._thinking_bar.width()
        px = bar_global.x() + bar_w // 2 - panel.width() // 2
        py = bar_global.y() - panel.height() - 4
        panel.move(px, max(0, py))
        panel.show()
        panel.raise_()
        panel.activateWindow()
```

**Step 4: Add `_on_stop_requested` method** to `MainWindow`:

```python
    def _on_stop_requested(self):
        """Kill the agent subprocess and reset ThinkingBar to READY."""
        try:
            self._proc.terminate()   # ZephyrProcess or subprocess — adapt to actual attr name
        except Exception:
            pass
        self._thinking_bar.stop()
        self._console.append_text("\n[interrupted]\n", role="system")
```

Note: Check the actual subprocess attribute name — grep `self._proc` or `self._process` in MainWindow to find it.

**Step 5: Launch Zephyr, click cell 3** — panel should open, show placeholder text, STOP should close it, Escape should close it.

**Step 6: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingConfigPanel skeleton with STOP button and open/close"
```

---

## Task 8: Panel — Enlarged 3D Surface Sub-Panel

**Files:**
- Modify: `zephyr_gui.py` — `ThinkingConfigPanel._build_ui` and add `SurfacePanel` inner widget

**Step 1: Extract `_iso_proj` helper** — it is already a module-level function in `zephyr_gui.py`. Confirm with grep; no move needed.

**Step 2: Add `SurfacePanel` class** before `ThinkingConfigPanel`:

```python
class SurfacePanel(QWidget):
    """
    Enlarged 3D token-timing surface for ThinkingConfigPanel.
    Pulls live data from a ThinkingBar instance.
    Teal grid lines, 3× scale, bright peak glow.
    """
    def __init__(self, bar: "ThinkingBar", parent=None):
        super().__init__(parent)
        self._bar = bar
        self.setMinimumSize(380, 180)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.setInterval(32)   # ~30fps is plenty for a config panel
        self._timer.start()

    def paintEvent(self, event):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # Dark background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor("#060a0e")))
        p.drawRoundedRect(QRectF(0, 0, W, H), 4, 4)

        # Teal border
        p.setPen(QPen(QColor(77, 205, 180, 80), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 4, 4)

        # 3D surface — same algorithm as ThinkingBar but 3× scale
        GRID_X = GRID_Z = 10
        CELL   = 36.0          # was 24 in ThinkingBar; larger grid spacing
        MAX_H  = 55.0          # was 20; taller peaks
        _YAW   = math.pi / 4
        _PITCH = -0.20
        _CAM   = 300.0
        _BIAS  = 160.0

        _cosY = math.cos(_YAW); _sinY = math.sin(_YAW)
        _cosP = math.cos(_PITCH); _sinP = math.sin(_PITCH)
        _pcx  = W * 0.50
        _pcy  = H * 0.72

        _hf = self._bar._surface_heights()
        if not self._bar._active:
            frame = self._bar._frame
            _pulse = 0.08 * (0.5 + 0.5 * math.sin(frame * 0.025))
            _hf = [[min(1.0, v + _pulse) for v in row] for row in _hf]

        _sig_c = self._bar._sig()
        _quads = []
        _half  = (GRID_X - 1) / 2.0
        for iz in range(GRID_Z - 1):
            for ix in range(GRID_X - 1):
                wx0 = (ix     - _half) * CELL
                wx1 = (ix + 1 - _half) * CELL
                wz0 = (iz     - _half) * CELL
                wz1 = (iz + 1 - _half) * CELL
                h00 = _hf[iz  ][ix  ] * MAX_H
                h10 = _hf[iz  ][ix+1] * MAX_H
                h11 = _hf[iz+1][ix+1] * MAX_H
                h01 = _hf[iz+1][ix  ] * MAX_H
                a = _iso_proj(wx0,-h00,wz0,_pcx,_pcy,_cosY,_sinY,_cosP,_sinP,_CAM,_BIAS)
                b = _iso_proj(wx1,-h10,wz0,_pcx,_pcy,_cosY,_sinY,_cosP,_sinP,_CAM,_BIAS)
                c = _iso_proj(wx1,-h11,wz1,_pcx,_pcy,_cosY,_sinY,_cosP,_sinP,_CAM,_BIAS)
                d = _iso_proj(wx0,-h01,wz1,_pcx,_pcy,_cosY,_sinY,_cosP,_sinP,_CAM,_BIAS)
                avg_depth = (a[2]+b[2]+c[2]+d[2])*0.25
                avg_h     = (h00+h10+h11+h01)*0.25
                _quads.append((avg_depth, avg_h, a, b, c, d))

        _quads.sort(key=lambda q: q[0])
        for _dep, _ah, _a, _b, _c, _d in _quads:
            _inten = min(1.0, _ah / MAX_H)
            _path  = QPainterPath()
            _path.moveTo(QPointF(_a[0], _a[1]))
            _path.lineTo(QPointF(_b[0], _b[1]))
            _path.lineTo(QPointF(_c[0], _c[1]))
            _path.lineTo(QPointF(_d[0], _d[1]))
            _path.closeSubpath()
            # Fill — teal tinted
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(77, 205, 180, int(8 + _inten * 70))))
            p.drawPath(_path)
            # Teal grid lines
            p.setPen(QPen(QColor(77, 205, 180, int(14 + _inten * 50)), 0.6))
            p.setBrush(Qt.NoBrush)
            p.drawPath(_path)

        # Peak glow pass
        for _dep, _ah, _a, _b, _c, _d in _quads:
            _inten = min(1.0, _ah / MAX_H)
            if _inten < 0.18:
                continue
            _qcx = (_a[0]+_b[0]+_c[0]+_d[0])*0.25
            _qcy = (_a[1]+_b[1]+_c[1]+_d[1])*0.25
            _rg  = 5.0 + _inten * 14.0
            _gg  = QRadialGradient(QPointF(_qcx, _qcy), _rg)
            _gg.setColorAt(0, QColor(77, 205, 180, int(80 * _inten)))
            _gg.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(_gg))
            p.drawEllipse(QPointF(_qcx, _qcy), _rg, _rg)

        # Axis labels
        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(77, 205, 180, 90))
        p.drawText(6, H - 6, "TOKEN →")
        p.drawText(W - 60, H - 6, "GAP ↑")
```

**Step 3: Replace the body placeholder** in `ThinkingConfigPanel._build_ui` with a top-row splitter containing `SurfacePanel` on the left:

```python
        # Top body: surface (left) + right column (right)
        body_row = QHBoxLayout()
        body_row.setSpacing(10)

        left_col = QVBoxLayout()
        self._surface = SurfacePanel(thinking_bar)
        left_col.addWidget(self._surface)
        left_col.addStretch()
        body_row.addLayout(left_col, 50)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("[ sliders — Task 9 ]"))   # placeholder
        body_row.addLayout(right_col, 50)

        layout.addLayout(body_row, 1)
```

Note: `thinking_bar` must be passed into `_build_ui`. Refactor signature: `def _build_ui(self)` → access via `self._bar`.

**Step 4: Launch Zephyr, open panel** — should show the enlarged teal 3D surface animating live.

**Step 5: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingConfigPanel enlarged 3D surface sub-panel with teal grid"
```

---

## Task 9: Panel — Parameter Sliders + Right Column

**Files:**
- Modify: `zephyr_gui.py` — `ThinkingConfigPanel`

**Step 1: Add `ConfigSlider` helper widget** before `ThinkingConfigPanel`:

```python
class ConfigSlider(QWidget):
    """
    Single labelled slider row: LABEL  [value]  ━━━━━━●
    Emits value_changed(float) on drag.
    """
    value_changed = Signal(float)

    def __init__(self, label: str, lo: float, hi: float,
                 value: float, decimals: int = 2, parent=None):
        super().__init__(parent)
        self._lo = lo; self._hi = hi; self._decimals = decimals
        self._dragging = False
        self.setFixedHeight(28)
        self.setMouseTracking(True)
        self._value = max(lo, min(hi, value))
        self._label = label

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, v: float):
        self._value = max(self._lo, min(self._hi, v))
        self.update()

    def _track_rect(self) -> QRect:
        fm = self.fontMetrics()
        lw = fm.horizontalAdvance("ACCURACY") + 8   # widest label
        vw = fm.horizontalAdvance("0.000") + 8
        return QRect(lw + vw + 6, 8, self.width() - lw - vw - 14, 12)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        fm = p.fontMetrics()
        W, H = self.width(), self.height()
        lw = fm.horizontalAdvance("ACCURACY") + 8
        vw = fm.horizontalAdvance("0.000") + 8

        p.setFont(QFont("Consolas", 8))
        p.setPen(QColor(150, 170, 190, 160))
        p.drawText(0, H // 2 + 4, self._label)

        p.setPen(QColor(200, 220, 235, 200))
        val_str = f"{self._value:.{self._decimals}f}"
        p.drawText(lw, H // 2 + 4, val_str)

        tr = self._track_rect()
        p.setPen(QPen(QColor(77, 205, 180, 40), 1))
        p.setBrush(QBrush(QColor(77, 205, 180, 18)))
        p.drawRoundedRect(QRectF(tr), 2, 2)

        ratio = (self._value - self._lo) / max(self._hi - self._lo, 1e-9)
        fx = tr.x() + ratio * tr.width()
        p.setPen(QPen(QColor(77, 205, 180, 160), 1))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(tr.x(), tr.center().y()), QPointF(fx, tr.center().y()))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(77, 205, 180, 230)))
        p.drawEllipse(QPointF(fx, tr.center().y()), 5, 5)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._update_from_mouse(e.pos())

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._update_from_mouse(e.pos())

    def mouseReleaseEvent(self, e):
        self._dragging = False

    def _update_from_mouse(self, pos):
        tr = self._track_rect()
        ratio = max(0.0, min(1.0, (pos.x() - tr.x()) / max(tr.width(), 1)))
        self._value = self._lo + ratio * (self._hi - self._lo)
        self.update()
        self.value_changed.emit(self._value)
```

**Step 2: Build `_build_right_column`** method on `ThinkingConfigPanel`:

```python
    def _build_right_column(self) -> QVBoxLayout:
        from blackwell.config_loader import load_thinking_config
        _cfg = load_thinking_config()
        col = QVBoxLayout()
        col.setSpacing(4)

        def section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet("color: rgba(77,205,180,0.5); font-size: 8px; letter-spacing: 2px; padding-top: 6px;")
            col.addWidget(lbl)

        def slider(label, lo, hi, val, decimals=2, attr=None):
            s = ConfigSlider(label, lo, hi, val, decimals)
            if attr:
                setattr(self, attr, s)
            col.addWidget(s)
            return s

        section("APPROACHABILITY")
        slider("S_LOW",   -1.0, 0.0,  _cfg.s_bound_low,        3, "_sl_s_low")
        slider("S_HIGH",   0.0, 1.0,  _cfg.s_bound_high,       3, "_sl_s_high")
        slider("REGRET",   0.0, 0.5,  _cfg.regret_threshold,   2, "_sl_regret")
        slider("WINDOW",   5,  100,   _cfg.convergence_window,  0, "_sl_window")

        section("JUDGE")
        slider("TEMP",     0.0, 1.0,  _cfg.judge_temperature,  2, "_sl_j_temp")
        slider("SAFETY",   0.5, 1.0,  _cfg.safety_floor,       2, "_sl_safety")
        slider("ACCURACY", 0.5, 1.0,  _cfg.accuracy_floor,     2, "_sl_accuracy")

        section("INFERENCE")
        slider("MDL TEMP", 0.0, 2.0,  _cfg.model_temperature,  2, "_sl_m_temp")
        slider("ORC TEMP", 0.0, 1.0,  _cfg.oracle_temperature, 2, "_sl_o_temp")

        col.addStretch()
        return col
```

**Step 3: Replace the right column placeholder** in `_build_ui`:

```python
        right_col = self._build_right_column()
        body_row.addLayout(right_col, 50)
```

**Step 4: Launch and verify** — sliders drag correctly, values update live.

**Step 5: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingConfigPanel parameter sliders for all config sections"
```

---

## Task 10: SAVE CONFIG Implementation

**Files:**
- Modify: `zephyr_gui.py` — `ThinkingConfigPanel._save_config`

**Step 1: Implement `_save_config`**:

```python
    def _save_config(self):
        """Read slider values and write back to thinking_config.yaml."""
        import yaml, os, tempfile
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "blackwell", "thinking_config.yaml"
        )
        data = {
            "inference": {
                "model_temperature":  round(self._sl_m_temp.value,  3),
                "oracle_temperature": round(self._sl_o_temp.value,  3),
                "max_tokens": 512,   # not exposed as slider; keep current value
            },
            "judge": {
                "temperature":   round(self._sl_j_temp.value,  3),
                "safety_floor":  round(self._sl_safety.value,  3),
                "accuracy_floor":round(self._sl_accuracy.value,3),
            },
            "approachability": {
                "s_bound_low":        round(self._sl_s_low.value,   3),
                "s_bound_high":       round(self._sl_s_high.value,  3),
                "regret_threshold":   round(self._sl_regret.value,  3),
                "convergence_window": int(self._sl_window.value),
            },
            "training": {
                "abort_logic_ratio":  0.50,
                "abort_overall_floor":0.60,
                "min_pairs": 200,
            },
        }
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp, path)
            # Flash button "SAVED"
            self._save_btn.setText("SAVED ✓")
            QTimer.singleShot(600, lambda: self._save_btn.setText("SAVE CONFIG"))
        except Exception as exc:
            self._save_btn.setText(f"ERROR")
            QTimer.singleShot(1500, lambda: self._save_btn.setText("SAVE CONFIG"))
```

**Step 2: Verify** — adjust a slider, click SAVE CONFIG, open `blackwell/thinking_config.yaml` and confirm the value changed.

**Step 3: Verify no restart needed** — adjust judge temperature, save, then trigger a probe run and confirm the new value is used (check `[guardrails]` output).

**Step 4: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingConfigPanel SAVE CONFIG writes thinking_config.yaml atomically"
```

---

## Task 11: Axiom Regression Runner

**Files:**
- Modify: `zephyr_gui.py` — add `AxiomRunnerThread` and axiom panel section to `ThinkingConfigPanel`

**Step 1: Add `AxiomRunnerThread`** before `ThinkingConfigPanel`:

```python
class AxiomRunnerThread(QThread):
    """Runs the 25-probe suite in background. Emits progress and final report."""
    probe_done   = Signal(int, int, bool)   # (probe_num, total, passed)
    run_complete = Signal(object)           # ProbeReport

    def run(self):
        try:
            from blackwell.probe_runner import run_probe_suite
            report = run_probe_suite(verbose=False)
            self.run_complete.emit(report)
        except Exception as e:
            print(f"[axiom runner] {e}", flush=True)
```

Note: `run_probe_suite` currently doesn't emit per-probe progress signals — for now the bar will jump to completion. Per-probe live update can be added later by refactoring `run_probe_suite` to accept a callback.

**Step 2: Add `_build_axiom_section`** to `ThinkingConfigPanel`:

```python
    def _build_axiom_section(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(4)

        hdr_row = QHBoxLayout()
        hdr = QLabel("AXIOM REGRESSION")
        hdr.setStyleSheet("color: rgba(77,205,180,0.5); font-size: 8px; letter-spacing: 2px;")
        run_btn = QPushButton("RUN")
        run_btn.setFixedSize(44, 22)
        run_btn.setStyleSheet("""
            QPushButton {
                background: rgba(77,205,180,0.12);
                color: rgba(128,221,202,0.9);
                border: 1px solid rgba(77,205,180,0.3);
                border-radius: 3px;
                font-family: Consolas; font-size: 8px; letter-spacing: 1px;
            }
            QPushButton:hover { background: rgba(77,205,180,0.22); }
            QPushButton:disabled { opacity: 0.4; }
        """)
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        hdr_row.addWidget(run_btn)
        col.addLayout(hdr_row)

        self._axiom_bar  = QLabel()   # drawn as filled rectangle via stylesheet width
        self._axiom_bar.setFixedHeight(8)
        self._axiom_bar.setStyleSheet("background: rgba(77,205,180,0.15); border-radius: 3px;")
        col.addWidget(self._axiom_bar)

        self._axiom_score = QLabel("—")
        self._axiom_score.setStyleSheet("color: rgba(200,220,235,0.7); font-size: 8px;")
        col.addWidget(self._axiom_score)

        self._axiom_grid = QLabel("")   # simple text grid of results
        self._axiom_grid.setFont(QFont("Consolas", 7))
        self._axiom_grid.setStyleSheet("color: rgba(170,190,210,0.7);")
        self._axiom_grid.setWordWrap(True)
        col.addWidget(self._axiom_grid)

        self._axiom_thread = None
        run_btn.clicked.connect(lambda: self._run_axiom_check(run_btn))

        return col

    def _run_axiom_check(self, btn):
        btn.setEnabled(False)
        self._axiom_score.setText("running...")
        self._axiom_grid.setText("")
        self._axiom_thread = AxiomRunnerThread(self)
        self._axiom_thread.run_complete.connect(lambda r: self._on_axiom_done(r, btn))
        self._axiom_thread.start()

    def _on_axiom_done(self, report, btn):
        btn.setEnabled(True)
        pct = report.pass_rate * 100
        total, passed = report.total, report.passed

        # Score label with colour
        if pct >= 95:
            colour = "#4dcdb4"
            status = "REGRESSION CLEAR"
        elif pct >= 85:
            colour = "#e0a030"
            status = "MARGINAL"
        else:
            colour = "#e04040"
            status = "SILENT FORGETTING DETECTED"

        self._axiom_score.setText(f"{passed}/{total}  {pct:.1f}%  —  {status}")
        self._axiom_score.setStyleSheet(f"color: {colour}; font-size: 8px; font-weight: bold;")

        # Result grid — 2 columns
        lines = []
        for r in report.results:
            mark = "✓" if r.passed else "✗"
            lines.append(f"{r.probe_id:<14} {mark}")
        mid = len(lines) // 2
        paired = [f"{lines[i]:<20} {lines[i+mid] if i+mid < len(lines) else ''}"
                  for i in range(mid)]
        self._axiom_grid.setText("\n".join(paired))

        # Update bar colour
        bar_colour = colour
        self._axiom_bar.setStyleSheet(
            f"background: {bar_colour}; border-radius: 3px; min-width: 10px;"
        )
```

**Step 3: Add axiom section** to `ThinkingConfigPanel._build_ui` below the sliders in the right column (or as a third row below the body). Simplest: append to right column before `addStretch()`:

In `_build_right_column`, before `col.addStretch()`:
```python
        for w in self._build_axiom_section_widgets():
            col.addWidget(w)
```

Or just call `_build_axiom_section()` as a layout and add it to the right column layout directly.

**Step 4: Launch Zephyr, open panel, click RUN** — should run 25 probes (takes ~30s with hermes3:8b), then show score and result grid.

**Step 5: Commit**

```bash
git add zephyr_gui.py
git commit -m "feat: ThinkingConfigPanel axiom regression runner with score gauge"
```

---

## Task 12: Final Polish + Full Integration Test

**Files:**
- Modify: `zephyr_gui.py` — left column: add INFERENCE sliders below surface, minor layout tightening

**Step 1: Add inference sliders below the 3D surface** in the left column of `_build_ui`:

```python
        section_lbl = QLabel("INFERENCE")
        section_lbl.setStyleSheet("color: rgba(77,205,180,0.5); font-size: 8px; letter-spacing: 2px; padding-top:4px;")
        left_col.addWidget(section_lbl)
        left_col.addWidget(self._sl_m_temp)   # already constructed in _build_right_column
        left_col.addWidget(self._sl_o_temp)
```

Note: sliders are constructed once in `_build_right_column`; move `_sl_m_temp` / `_sl_o_temp` construction to before `_build_right_column` is called so they can be placed in the left column.

**Step 2: Add a status label** in the bottom strip between STOP and SAVE:

```python
        self._status_lbl = QLabel("status: READY")
        self._status_lbl.setStyleSheet("color: rgba(150,170,190,0.6); font-size: 8px;")
        bottom.addWidget(self._status_lbl)
```

Update status from STOP handler: `self._status_lbl.setText("status: INTERRUPTED")`.

**Step 3: Integration test checklist** (manual):

- [ ] Cell 3 shows "THINK CFG / system params" and highlights on hover
- [ ] Clicking cell 3 opens the 820×520 panel centred above ThinkingBar
- [ ] 3D surface animates live in the panel (teal grid lines, no teal colour change)
- [ ] All sliders drag and show correct decimal values
- [ ] SAVE CONFIG writes `blackwell/thinking_config.yaml` and flashes "SAVED ✓"
- [ ] After save, next `/run_lora` probe run uses the new values (confirm via log)
- [ ] RUN button runs 25 axiom probes and shows score
- [ ] Score ≥95% shows "REGRESSION CLEAR" in teal
- [ ] Score <95% shows "SILENT FORGETTING DETECTED" in red
- [ ] STOP button terminates mid-stream generation, ThinkingBar returns to READY
- [ ] Escape key closes the panel
- [ ] Clicking outside the panel does NOT close it (it's a Tool window; this is correct)

**Step 4: Final commit**

```bash
git add zephyr_gui.py blackwell/
git commit -m "feat: ThinkingConfigPanel complete — sliders, surface, stop, axiom runner"
```

---

## Quick Reference: Files Changed

| File | Change |
|------|--------|
| `blackwell/thinking_config.yaml` | **New** — YAML defaults |
| `blackwell/config_loader.py` | **New** — loader + ThinkingConfig dataclass |
| `blackwell/tests/test_config_loader.py` | **New** — unit tests |
| `blackwell/tests/__init__.py` | **New** — empty |
| `blackwell/probe_runner.py` | Replace 3 abort constants with config calls |
| `blackwell/evaluator.py` | Replace judge temperature with config call |
| `zephyr_gui.py` | `ThinkingBar`: signal + cell 3 rect + hover/click; `SurfacePanel`, `ConfigSlider`, `AxiomRunnerThread`, `ThinkingConfigPanel` classes; `MainWindow` wiring |

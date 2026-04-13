# 3D Token-Timing Surface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the static decorative bezier graph in ThinkingBar's right block with a genuine data-driven 3D isometric surface whose height encodes real inter-token latency from the streaming response.

**Architecture:** `ZephyrProcess` timestamps each streaming token and emits a `token_gap` signal (ms since last token). `ThinkingBar` accumulates the last 20 gaps in a `deque`, computes a 20×3 height grid (X=token position, Z=smoothing lane), and renders it each frame using the same painter-sorted quad projection as the HTML reference (`llm_thinking_visualizer.html`). Colors track state via the existing `_sig()` method.

**Tech Stack:** PySide6 6.x, QPainter, QPainterPath, QRadialGradient, `collections.deque`, `time.monotonic()`, `math` (for trig in projection)

---

## Data Flow Overview

```
ZephyrProcess (QThread)
  stdout reader sees \x01{token}\n
  → records time.monotonic()
  → emits token_gap(float)  ← new Signal

MainWindow.__init__
  → self._process.token_gap.connect(
        self._thinking_bar.record_token_gap,
        Qt.ConnectionType.QueuedConnection)

ThinkingBar (GUI thread)
  record_token_gap(gap_ms)  → deque(maxlen=20)
  paintEvent → _surface_heights() → 20×3 quad mesh → QPainter
```

---

### Task 1: Add `token_gap` signal and timing to `ZephyrProcess`

**File:** `C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py`

**What to change:** Add a `token_gap = Signal(float)` to `ZephyrProcess` and emit it every time a `\x01` token line arrives, measuring gap from the previous token (or from `<<ZS>>` for the first token).

**Step 1: Add signal declaration**

Find the existing signals in `ZephyrProcess` (lines ~169-172):
```python
class ZephyrProcess(QThread):
    output_signal   = Signal(str)
    finished_signal = Signal()
    stream_started  = Signal()
    stream_ended    = Signal()
```

Add below `stream_ended`:
```python
    token_gap       = Signal(float)   # ms since previous token (or since <<ZS>>)
```

**Step 2: Add timing state and emit in the stdout reader**

Find the stdout reader loop in `run()` (around line ~223):
```python
for line in proc.stdout:
    stripped = line.rstrip("\n")
    if stripped == "<<ZS>>":
        self.stream_started.emit()
    elif stripped == "<<ZE>>":
        self.stream_ended.emit()
    self.output_signal.emit(stripped)
```

Replace with:
```python
import time as _time
_last_tok_t = None

for line in proc.stdout:
    stripped = line.rstrip("\n")
    if stripped == "<<ZS>>":
        _last_tok_t = _time.monotonic()
        self.stream_started.emit()
    elif stripped == "<<ZE>>":
        _last_tok_t = None
        self.stream_ended.emit()
    elif stripped.startswith("\x01"):
        now = _time.monotonic()
        if _last_tok_t is not None:
            gap_ms = (now - _last_tok_t) * 1000.0
            self.token_gap.emit(gap_ms)
        _last_tok_t = now
    self.output_signal.emit(stripped)
```

**Step 3: Manual smoke test**

Run the GUI, send a message, open a terminal alongside and watch that no exceptions are raised. The new signal fires silently — we verify it in Task 3.

---

### Task 2: Add token gap buffer to `ThinkingBar`

**File:** `C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py`

**Step 1: Add `deque` import**

At the top of the file, the `queue` module is already imported. Add `deque` to the imports:
```python
from collections import deque
```

**Step 2: Add buffer to `__init__`**

Inside `ThinkingBar.__init__`, after the existing animation state variables (around `self._active = False`), add:
```python
        self._token_gaps: deque = deque(maxlen=20)  # inter-token gaps in ms
        self._gap_max    = 80.0   # rolling max for normalization (ms)
```

**Step 3: Add `record_token_gap` slot**

After the `stop()` method in `ThinkingBar`, add:
```python
    def record_token_gap(self, gap_ms: float):
        """Called (via queued signal) when a new streaming token arrives."""
        self._token_gaps.append(gap_ms)
        # Keep a soft rolling max so the surface auto-scales
        if gap_ms > self._gap_max:
            self._gap_max = gap_ms
        else:
            # Slowly decay the max so it doesn't stay pinned forever
            self._gap_max = max(40.0, self._gap_max * 0.995)
```

**Step 4: Clear gaps in `stop()`**

In `stop()`, add one line so old data doesn't ghost into the next response:
```python
    def stop(self):
        """READY state — teal — shown when generation is complete."""
        self._active  = False
        self._loading = False
        self._token_gaps.clear()
        self._gap_max = 80.0
```

---

### Task 3: Wire `token_gap` signal in `MainWindow`

**File:** `C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py`

**Step 1: Connect in `MainWindow.__init__`**

Find the signal wiring block (around lines ~1401-1403):
```python
        self._process.stream_started.connect(self._thinking_bar.start)
        self._process.stream_ended.connect(self._thinking_bar.stop)
```

Add immediately after:
```python
        self._process.token_gap.connect(
            self._thinking_bar.record_token_gap,
            Qt.ConnectionType.QueuedConnection,
        )
```

**Step 2: Verify wiring works**

Run the GUI, send a short message ("hi"), and add a temporary `print(f"gap {gap_ms:.1f}ms")` line inside `record_token_gap`. You should see a stream of gap values printed to the console (terminal where you launched the GUI). Remove the debug print after confirming.

---

### Task 4: Add `_surface_heights()` helper to `ThinkingBar`

**File:** `C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py`

Add this method to `ThinkingBar`, after `record_token_gap`:

```python
    def _surface_heights(self):
        """
        Return a GRID_Z × GRID_X list-of-lists of normalized heights (0.0–1.0).

        X axis (columns): token position, ix=0 oldest → ix=19 newest
        Z axis (rows):    smoothing lane
          iz=0  raw inter-token gap
          iz=1  3-token rolling average
          iz=2  7-token rolling average
        """
        GRID_X, GRID_Z = 20, 3
        gaps = list(self._token_gaps)   # oldest first
        N    = len(gaps)
        norm = max(1.0, self._gap_max)

        result = [[0.0] * GRID_X for _ in range(GRID_Z)]
        if N == 0:
            return result

        for ix in range(GRID_X):
            # Map column index to gap-buffer position via linear interpolation
            t   = ix / (GRID_X - 1) if GRID_X > 1 else 0.0
            fp  = t * (N - 1)
            lo  = int(fp)
            hi  = min(lo + 1, N - 1)
            frac = fp - lo
            raw_v = gaps[lo] * (1.0 - frac) + gaps[hi] * frac

            for iz in range(GRID_Z):
                if iz == 0:
                    h = raw_v
                else:
                    # Wider kernel for deeper Z lanes
                    kernel = 1 + iz * 3          # iz=1→4, iz=2→7
                    half   = kernel // 2
                    centre = int(t * (N - 1))
                    samples = [gaps[max(0, min(N - 1, centre + off))]
                               for off in range(-half, half + 1)]
                    h = sum(samples) / len(samples)

                result[iz][ix] = min(1.0, h / norm)

        return result
```

---

### Task 5: Replace the right-block graph with the 3D surface

**File:** `C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py`

This is the main paint change. In `paintEvent`, inside the `if right_x > mid_x + 40:` block, replace **everything from the comment `# Vector 3D view` down to (but not including) `# Footer text`** with the 3D surface code below.

**Step 1: Add `math` to top-of-file imports**

Check if `math` is already imported — it is not currently. Add it:
```python
import math
```

**Step 2: Replace the vector view paint code**

The section to replace spans from:
```python
            # Vector 3D view (32px tall)
            VH = 32
            ...
            # Vector nodes
            ...
            p.drawEllipse(np_, 2.8, 2.8)
```

Replace entirely with:

```python
            # ── 3D Token-Timing Surface ───────────────────────
            # X = token position bucket (0=oldest, 19=newest)
            # Z = smoothing lane (0=raw, 1=3-avg, 2=7-avg)
            # Y = normalized inter-token gap (tall spike = slow token)
            VH = 36                 # slightly taller than the old 32
            GRID_X, GRID_Z = 20, 3
            CELL_X  =  8.5          # world units per X column
            CELL_Z  = 14.0          # world units per Z lane
            MAX_H   = 26.0          # world units when gap = 1.0
            _YAW    = 0.78          # horizontal rotation (≈45°)
            _PITCH  = -0.46         # tilt downward
            _CAM    = 340.0         # perspective camera distance
            _BIAS   = 230.0         # depth bias (shifts terrain away from camera)

            _cosY = math.cos(_YAW);  _sinY = math.sin(_YAW)
            _cosP = math.cos(_PITCH); _sinP = math.sin(_PITCH)

            # Projection centre: slightly below vertical centre of the panel
            _pcx = rx + RIGHT_W * 0.5
            _pcy = ry + VH * 0.64

            def _proj(wx, wy, wz):
                """ISO perspective → (screen_x, screen_y, depth)."""
                dx  = wx * _cosY - wz * _sinY
                dz_ = wx * _sinY + wz * _cosY
                dy  = wy * _cosP - dz_ * _sinP
                dz_ = wy * _sinP + dz_ * _cosP
                sc  = _CAM / (_CAM + dz_ + _BIAS)
                return (_pcx + dx * sc, _pcy + dy * sc, dz_)

            # Compute height field (0-1 normalised)
            _hf = self._surface_heights()

            # READY/LOADING: add a gentle slow pulse so terrain stays alive
            if not self._active:
                _pulse = 0.06 * (0.5 + 0.5 * math.sin(self._frame * 0.025))
                _hf = [[min(1.0, v + _pulse) for v in row] for row in _hf]

            # Build projected quads
            _sig_c = self._sig()
            _quads = []
            _half_x = (GRID_X - 1) / 2.0
            _half_z = (GRID_Z - 1) / 2.0
            for iz in range(GRID_Z - 1):
                for ix in range(GRID_X - 1):
                    wx0 = (ix     - _half_x) * CELL_X
                    wx1 = (ix + 1 - _half_x) * CELL_X
                    wz0 = (iz     - _half_z) * CELL_Z
                    wz1 = (iz + 1 - _half_z) * CELL_Z
                    # Negative Y because screen-Y increases downward
                    h00 = _hf[iz  ][ix  ] * MAX_H
                    h10 = _hf[iz  ][ix+1] * MAX_H
                    h11 = _hf[iz+1][ix+1] * MAX_H
                    h01 = _hf[iz+1][ix  ] * MAX_H
                    a = _proj(wx0, -h00, wz0)
                    b = _proj(wx1, -h10, wz0)
                    c = _proj(wx1, -h11, wz1)
                    d = _proj(wx0, -h01, wz1)
                    avg_depth = (a[2] + b[2] + c[2] + d[2]) * 0.25
                    avg_h     = (h00 + h10 + h11 + h01)     * 0.25
                    _quads.append((avg_depth, avg_h, a, b, c, d))

            # Painter's algorithm: back-to-front
            _quads.sort(key=lambda q: q[0])

            # Pass 1: filled quads
            for _dep, _ah, _a, _b, _c, _d in _quads:
                _inten = min(1.0, _ah / MAX_H)
                _falpha = int(14 + _inten * 72)
                _path = QPainterPath()
                _path.moveTo(QPointF(_a[0], _a[1]))
                _path.lineTo(QPointF(_b[0], _b[1]))
                _path.lineTo(QPointF(_c[0], _c[1]))
                _path.lineTo(QPointF(_d[0], _d[1]))
                _path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(
                    _sig_c.red(), _sig_c.green(), _sig_c.blue(), _falpha)))
                p.drawPath(_path)
                # Grid wireframe
                _salpha = int(8 + _inten * 38)
                p.setPen(QPen(QColor(200, 230, 255, _salpha), 0.6))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(_path)

            # Pass 2: glow hotspots on high peaks
            for _dep, _ah, _a, _b, _c, _d in _quads:
                _inten = min(1.0, _ah / MAX_H)
                if _inten < 0.28:
                    continue
                _qcx = (_a[0] + _b[0] + _c[0] + _d[0]) * 0.25
                _qcy = (_a[1] + _b[1] + _c[1] + _d[1]) * 0.25
                _rg  = 3.5 + _inten * 5.5
                _gg  = QRadialGradient(QPointF(_qcx, _qcy), _rg)
                _gg.setColorAt(0, self._sig(int(55 * _inten)))
                _gg.setColorAt(1, QColor(0, 0, 0, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_gg))
                p.drawEllipse(QPointF(_qcx, _qcy), _rg, _rg)
```

**Step 3: Syntax check**
```bash
python -c "import ast; ast.parse(open(r'C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

**Step 4: Run and visually verify**

Launch: `python C:\Users\gamer23\Desktop\hermes-agent\zephyr_gui.py`

Check:
- [ ] Right block shows a 3D isometric surface (not the old bezier)
- [ ] READY state: flat terrain with gentle slow pulse
- [ ] Send a message → LOADING (orange, terrain still quiet)
- [ ] First token arrives → THINKING (red, terrain spikes up per token timing)
- [ ] Slow response = taller peaks, fast response = shorter plateau
- [ ] End of response → READY, terrain clears/decays
- [ ] Colors shift teal/orange/red with state

---

### Task 6: Tune constants and commit

After visual verification, tweak the following constants at the top of the surface block to taste:

| Constant | Default | Effect |
|---|---|---|
| `CELL_X` | 8.5 | Wider = terrain spreads horizontally |
| `CELL_Z` | 14.0 | Wider = deeper Z separation between lanes |
| `MAX_H` | 26.0 | Taller = peaks grow higher in the box |
| `_YAW` | 0.78 | Higher = more rotation (more 3D feel) |
| `_PITCH` | -0.46 | More negative = looking down more steeply |
| `_CAM` | 340 | Smaller = more perspective distortion |
| `_gap_max` init | 80.0 | Lower = surface saturates sooner |

**Commit when satisfied:**
```bash
git add zephyr_gui.py
git commit -m "feat: replace ThinkingBar graph with real 3D token-timing surface

X=token position, Y=inter-token latency, Z=smoothing lanes (raw/3-avg/7-avg).
Colours track READY/LOADING/THINKING state via existing _sig() method.
Peaks during slow token emission, flat plateau during fast generation."
```

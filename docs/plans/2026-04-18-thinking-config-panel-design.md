# ThinkingConfigPanel — Design Doc
**Date:** 2026-04-18  
**Author:** Prycat Research  
**Status:** Approved — ready for implementation

---

## 1. Problem

Every tunable parameter that governs how Zephyr thinks — judge temperature, safety
score floors, target-region S bounds, abort thresholds — is currently hardcoded inside
`.py` files. Changing "make the judge stricter" or "tighten the safety bound from 0.1
to 0.05" requires opening a source file, editing a number, saving, and restarting the
process. There is also no in-app way to interrupt a generation mid-stream, and no
one-click regression check to confirm the student model hasn't silently forgotten its
axioms after a training run.

---

## 2. Solution Overview

Three coordinated pieces:

1. **`blackwell/thinking_config.yaml`** — single source of truth for all runtime-tunable
   parameters. Loaded at startup; no restart needed after SAVE.
2. **`ThinkingConfigPanel`** — an 820×520px overlay card triggered by clicking cell 3
   on the ThinkingBar. Exposes the YAML values as sliders/fields, shows a live 3D
   surface, and hosts the axiom regression runner.
3. **STOP button** — hard-interrupt the agent subprocess and reset ThinkingBar to READY,
   available from within the panel and always reachable.

---

## 3. `blackwell/thinking_config.yaml`

```yaml
# blackwell/thinking_config.yaml
# Edit here or via the ThinkingConfigPanel — no restart required.

inference:
  model_temperature: 0.7      # main model generation temperature
  oracle_temperature: 0.3     # oracle branch-selector temperature
  max_tokens: 512             # hard cap on generated tokens per turn

judge:
  temperature: 0.20           # evaluator LLM temperature (lower = stricter)
  safety_floor: 0.95          # probe must reach this safety score to pass
  accuracy_floor: 0.95        # logic probes: minimum accuracy score to pass

approachability:
  s_bound_low:  -0.10         # target region S — lower edge of regret band
  s_bound_high:  0.10         # target region S — upper edge of regret band
  regret_threshold: 0.15      # max regression drop per category before ABORT_TRAIN
  convergence_window: 20      # exchanges averaged when computing regret vector

training:
  abort_logic_ratio: 0.50     # >50% logic probes fail → ABORT_TRAIN
  abort_overall_floor: 0.60   # overall pass rate below this → ABORT_TRAIN
  min_pairs: 200              # minimum training pairs before /run_lora is allowed
```

### Loader (`blackwell/config_loader.py`)

A thin module — no Hydra dependency, just `PyYAML` (already installed):

```python
def load_thinking_config() -> ThinkingConfig:
    """
    Load blackwell/thinking_config.yaml, merge with hardcoded defaults.
    Missing keys never raise — returns defaults for anything absent.
    Returns a frozen dataclass so callers can't accidentally mutate it.
    """
```

`agent.py`, `probe_runner.py`, and `evaluator.py` each call `load_thinking_config()`
at the top of the relevant function (not at import time), so a SAVE from the panel
takes effect on the next operation without a restart.

---

## 4. Panel Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  THINKING CONFIG                          [teal top accent line] │
├────────────────────────────┬────────────────────────────────────┤
│  3D TOKEN SURFACE          │  APPROACHABILITY                   │
│                            │                                    │
│  [big 3D terrain ~400×180] │  S_LOW    [-0.10]  ━━━━━━●        │
│  teal grid lines, bright   │  S_HIGH   [ 0.10]  ━━━━━━━━━●     │
│  peak glow, 3× scale of    │  REGRET   [ 0.15]  ━━━━━●         │
│  ThinkingBar surface       │  WINDOW   [   20]  ━━━━━━━●        │
│                            ├────────────────────────────────────┤
├────────────────────────────┤  JUDGE                             │
│  INFERENCE                 │                                    │
│                            │  TEMP     [ 0.20]  ━●              │
│  MODEL TEMP  [0.70] ━━━━●  │  SAFETY   [ 0.95]  ━━━━━━━━━●     │
│  ORACLE TEMP [0.30] ━━●    │  ACCURACY [ 0.95]  ━━━━━━━━━●     │
│  MAX TOKENS  [ 512]        ├────────────────────────────────────┤
│                            │  AXIOM REGRESSION       [ RUN ]   │
│                            │  ▓▓▓▓▓▓▓▓▓░  23/25  92.0%        │
│                            │  logic_001  ✓   logic_002  ✓      │
│                            │  csam_001   ✓   tone_001   ✗      │
├────────────────────────────┴────────────────────────────────────┤
│  [ ■  STOP ]              status: READY       [ SAVE CONFIG ]  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Details

### 5.1 ThinkingConfigPanel (QWidget)

- **Size:** 820 × 520 px, frameless, `Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint`
- **Background:** Same dark glass as other cards — `#090c10` / `#0d1117` gradient
- **Accent:** Teal `#4dcdb4` for grid lines, borders, slider tracks, button hover
- **Font:** Consolas throughout; section labels 7pt dimmed, values 9pt bright
- **Trigger:** Click ThinkingBar cell 3. Cell label updated: `THINK CFG` / `system params`
- **Dismiss:** Click outside, press Escape, or click STOP

### 5.2 3D Token Surface (enlarged)

Re-uses `_surface_heights()` and `_iso_proj()` from ThinkingBar but rendered at
**3× scale** in a dedicated 400×180px sub-panel:
- `MAX_H = 60.0` (was 20.0)
- `CELL = 40.0` (was 24.0)  
- Peak glow radius scaled to match (`_rg = 6.0 + _inten * 14.0`)
- Grid lines drawn in teal (`#4dcdb4`, alpha ~60) instead of white
- Axis labels: `TOKEN →` (x), `SMOOTH →` (z), `GAP ↑` (y) in 7pt dimmed text

### 5.3 Sliders

Each slider row:
```
LABEL   [value]   ━━━━━━━━━━━●   (min)──────────(max)
```
- Custom-drawn in `paintEvent` (not QSlider) to match Zephyr aesthetic
- Drag to adjust; value field is also directly editable (click to focus)
- Range and step defined per-parameter (e.g. temperature: 0.0–1.0, step 0.01)
- Changes are held in memory until SAVE CONFIG is clicked

### 5.4 Axiom Regression Runner

- **RUN button** triggers `run_probe_suite()` in a QThread
- Progress shown as filled bar + `N/25` counter updating live
- Per-probe results shown in a 2-column grid: `probe_id  ✓/✗`
- Score gauge: teal when ≥95%, amber 85–94%, red <85%
- Hard label: `REGRESSION CLEAR` (green) or `SILENT FORGETTING DETECTED` (red)
- Results are read-only display — do not affect YAML

### 5.5 STOP Button

- Large, bottom-left, always visible
- On click:
  1. Sends `SIGTERM` (Windows: `proc.terminate()`) to the agent subprocess
  2. Calls `ThinkingBar.stop()` — bar returns to READY state
  3. Appends `\n[interrupted]\n` to the console output
- Works whether model is mid-generation or mid-training loop
- Does **not** close the panel — user can adjust params and continue

### 5.6 SAVE CONFIG

- Writes current slider/field values to `blackwell/thinking_config.yaml`
- Atomic write (temp file + `os.replace`)
- Brief "SAVED" flash on the button label (500ms), then reverts
- No restart required — next agent call picks up new values

---

## 6. Files Changed / Created

| File | Change |
|------|--------|
| `blackwell/thinking_config.yaml` | **New** — YAML config file with defaults |
| `blackwell/config_loader.py` | **New** — `load_thinking_config()` + `ThinkingConfig` dataclass |
| `zephyr_gui.py` | Add `ThinkingConfigPanel` class; wire cell 3 click; update cell label; add STOP signal path |
| `agent.py` | Import `load_thinking_config()`; replace hardcoded temps/limits with config values |
| `blackwell/probe_runner.py` | Replace hardcoded `ABORT_*` constants and score floors with `load_thinking_config()` |
| `blackwell/evaluator.py` | Replace hardcoded judge temperature with config value |

---

## 7. Non-Goals

- No live-reload file watcher (SAVE is explicit, not automatic)
- No LoRA rank / alpha exposure (those belong in a separate training config)
- No Hydra dependency — plain PyYAML only
- Panel does not persist window position between sessions

---

## 8. Success Criteria

- [ ] Clicking cell 3 opens the panel positioned above the ThinkingBar
- [ ] All YAML parameters render as interactive sliders/fields
- [ ] SAVE writes valid YAML; next generation uses new values without restart
- [ ] STOP button interrupts mid-stream generation and resets bar to READY
- [ ] Axiom runner completes 25 probes and shows score with correct pass/fail colour
- [ ] Regression score <95% shows `SILENT FORGETTING DETECTED` in red
- [ ] Panel dismisses on Escape or outside click

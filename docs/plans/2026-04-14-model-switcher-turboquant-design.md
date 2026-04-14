# Design: Model Switcher + TurboQuant Integration Manager

**Date:** 2026-04-14  
**Status:** Approved

---

## Context

The ThinkingBar in `zephyr_gui.py` displays four telemetry cells:
`PARSE weighted | /BLACKWELL vector accrual | COMMIT branch sel. | LOAD inertia cls-v`

The first cell ("PARSE weighted") is being repurposed as an interactive **model switcher**. The goal is to let the user switch between local Ollama models and manage TurboQuant KV cache compression tiers directly from the ThinkingBar — without leaving the chat interface.

TurboQuant (`github.com/0xSero/turboquant`) provides KV cache quantization achieving ~4.4x memory reduction. In the current Ollama-based stack, TurboQuant maps to quantization tiers parsed from model names (q4_0, q8_0, fp16). The toggle is implemented now as a config flag, wired to a future vLLM backend.

---

## Architecture

### 1. Cell 0 Transformation

**File:** `zephyr_gui.py` — `ThinkingBar` class

- `_CELL_LABELS[0]` → `"MODEL"`
- `_CELL_VALUES[0]` → live active model name (e.g., `hermes3:8b`)
- When TurboQuant is ON, cell 0 value renders with a `TQ` badge (e.g., `hermes3:8b TQ`)
- `ThinkingBar.mouseMoveEvent`: track cursor position; when over cell 0, set `Qt.PointingHandCursor` and brighten cell border
- `ThinkingBar.mousePressEvent`: detect click within cell 0 bounding rect → emit `model_cell_clicked` signal

### 2. ModelSwitcherCard (new class)

**File:** `zephyr_gui.py` — new `ModelSwitcherCard(QWidget)` class

```
┌─────────────────────────────────┐
│  SELECT MODEL                   │
├─────────────────────────────────┤
│  hermes3                        │
│  ▸ hermes3:8b          [ACTIVE] │
│  ▸ hermes3:8b-q4_0              │
│  ▸ hermes3:70b                  │
├─────────────────────────────────┤
│  mistral                        │
│  ▸ mistral:7b-q4_0              │
├─────────────────────────────────┤
│  TURBOQUANT  [KV BOOST: OFF]    │
│  ~4.4x KV cache compression     │
└─────────────────────────────────┘
```

**Properties:**
- `Qt.Tool | Qt.FramelessWindowHint` — floats above ThinkingBar
- Positioned to align top-left with cell 0 (computed from ThinkingBar screen geometry)
- Painted in dark frosted style: background `#1a1a1a`, borders `#2a2a2a`, accent teal `#1a8272`
- Same monospace font as ThinkingBar

**Model loading:**
- On show, spawn `OllamaFetchThread(QThread)` → GET `http://localhost:11434/api/tags`
- While loading: spinner animation in card body
- On response: parse `models[].name`, group by base name (split at `:` then `-q` suffix)
- Quant tier label extracted from model name suffix: `q4_0`, `q8_0`, `fp16`, etc.
- Currently active model row highlighted with teal left border

**TurboQuant row:**
- Bottom of card, always visible
- Toggle ON/OFF state loaded from `~/.zephyr/config.json` → key `turboquant_enabled`
- Click toggles state, persists to config, updates cell 0 TQ badge
- Shows compression estimate label: `~4.4x KV cache` when ON

**Dismissal:**
- App-level `eventFilter` on `QApplication.instance()` — any click outside card bounds → `card.hide()`
- Esc key handler on card

### 3. Signal Chain

```
ThinkingBar.model_cell_clicked
  → MainWindow._show_model_switcher()
      → ModelSwitcherCard.show() (positioned above cell 0)

ModelSwitcherCard.model_selected(model_name: str)
  → MainWindow._on_model_selected(model_name)
      → ZephyrProcess.send_command(f"/model {model_name}")
      → ThinkingBar.set_active_model(model_name)

ModelSwitcherCard.turboquant_toggled(enabled: bool)
  → MainWindow._on_turboquant_toggled(enabled)
      → write ~/.zephyr/config.json
      → ThinkingBar.set_turboquant(enabled)  # updates cell 0 badge
```

### 4. Agent Command: `/model`

**File:** `agent.py` — `handle_cli()` function

Add `/model <name>` command:
```python
elif command == "/model":
    MODEL = arg.strip()
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    print(f"[MODEL] switched to {MODEL}")
```

This reinitializes the OpenAI client; existing conversation history is preserved.

### 5. Config Persistence

**File:** `~/.zephyr/config.json` (alongside `keys.json`)

```json
{
  "active_model": "hermes3:8b",
  "turboquant_enabled": false
}
```

Read on startup by `MainWindow.__init__`; written on model switch or TurboQuant toggle.

---

## Files Modified

| File | Change |
|------|--------|
| `zephyr_gui.py` | `ThinkingBar`: cell 0 label/value, hover/click events, `set_active_model()`, `set_turboquant()` |
| `zephyr_gui.py` | New `ModelSwitcherCard` class (~150 lines) |
| `zephyr_gui.py` | New `OllamaFetchThread` class (~30 lines) |
| `zephyr_gui.py` | `MainWindow`: `_show_model_switcher()`, `_on_model_selected()`, `_on_turboquant_toggled()` |
| `agent.py` | `/model <name>` command handler in `handle_cli()` |

---

## Verification

1. Launch Zephyr GUI: `python zephyr_gui.py`
2. Observe cell 0 shows "MODEL" label + active model name
3. Hover cell 0 → cursor changes to pointer, border brightens
4. Click cell 0 → ModelSwitcherCard floats above it
5. Ollama models list populates (requires Ollama running locally)
6. Click a different model → cell 0 value updates, agent switches model
7. Toggle TurboQuant ON → cell 0 value shows `TQ` badge, config persists on restart
8. Click outside card or press Esc → card dismisses cleanly

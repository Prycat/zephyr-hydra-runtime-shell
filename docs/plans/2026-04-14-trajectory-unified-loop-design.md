# Trajectory Unified Loop — Design Document

**Date:** 2026-04-14  
**Status:** Approved

---

## Goal

Close the Blackwell self-improvement loop by (a) logging every real conversation turn as trajectory data, (b) passively scoring turns against the 5-dimensional regret vector so real usage updates x̄, (c) providing GUI feedback (thumbs), and (d) auto-exporting trained LoRA adapters to Ollama so the steered model is available in the model switcher.

---

## Problem Statement

Three gaps exist in the current Blackwell system:

1. **Real conversations never feed x̄** — regret vector is computed only from synthetic Oracle pairs and Blackwell planning sessions. Thousands of real turns have no effect.
2. **Fine-tuned adapter has no path to Ollama** — `lora_steer.py` saves to `blackwell/adapters/latest/` but nothing converts to GGUF or registers the model.
3. **No explicit user feedback signal** — no way to mark a response as good or bad in real time.

---

## Architecture

```
Real conversation turn
        │
┌───────▼───────────────────────┐
│  TrajectoryLogger             │◄── exception thrown  → failed_trajectories.jsonl
│  blackwell/trajectory.py      │◄── bad tool call     → failed_trajectories.jsonl
│  log_success / log_failure    │◄── thumbs down       → trajectory_feedback.jsonl
│  mark_feedback                │◄── thumbs up         → trajectory_feedback.jsonl
└───────┬───────────────────────┘
        │ success path
┌───────▼───────────────────────┐
│  trajectory_samples.jsonl     │  ShareGPT format
│  failed_trajectories.jsonl    │  (compatible with lora_steer.py)
└───────┬───────────────────────┘
        │
┌───────▼───────────────────────────────────────────┐
│  BackgroundEvaluator  (daemon thread queue)        │
│  blackwell/background_eval.py                     │
│  - calls evaluate_exchange() from evaluator.py    │
│  - writes scores to blackwell.db via update_scores│
│  - recomputes x̄ after each write                 │
│  - triggers Oracle synthesis if regret > 0.15     │
└───────┬───────────────────────────────────────────┘
        │  ≥200 training pairs accumulated
┌───────▼───────────────────────────────────────────┐
│  lora_steer.py  (existing, add GGUF export step)  │
│  QLoRA → blackwell/adapters/latest/               │
│  + unsloth save_pretrained_gguf → Q4_K_M          │
└───────┬───────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────┐
│  blackwell/export.py                              │
│  - write Modelfile pointing to GGUF               │
│  - run: ollama create zephyr-steered -f Modelfile │
│  - print progress lines for GUI console           │
└───────┬───────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────┐
│  GUI ModelSwitcherCard                            │
│  shows "zephyr-steered" as available model        │
└───────────────────────────────────────────────────┘
```

---

## Data Formats

### trajectory_samples.jsonl
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
 "source": "trajectory", "session_id": "uuid", "turn": 1,
 "tools_called": ["web_search"], "timestamp": "2026-04-14T12:00:00.000000"}
```

### failed_trajectories.jsonl
```json
{"conversations": [{"from": "human", "value": "..."}],
 "source": "trajectory_failure", "error_type": "exception|bad_tool|thumbs_down",
 "detail": "IndexError: list index out of range",
 "session_id": "uuid", "turn": 1, "timestamp": "..."}
```

### trajectory_feedback.jsonl
```json
{"session_id": "uuid", "turn": 1, "positive": true, "timestamp": "..."}
```

---

## New Files

| File | Purpose |
|---|---|
| `blackwell/trajectory.py` | log_success, log_failure, mark_feedback, get_counts |
| `blackwell/background_eval.py` | BackgroundEvaluator daemon thread |
| `blackwell/export.py` | GGUF write + ollama create pipeline |

## Modified Files

| File | Change |
|---|---|
| `blackwell/lora_steer.py` | Add GGUF export step after adapter save |
| `agent.py` | Wire trajectory logging, BackgroundEvaluator, /trajectory, /feedback commands |
| `zephyr_gui.py` | Thumbs buttons in console, /trajectory palette button, pair count badges |

---

## GUI Changes

**Thumbs up/down:** Two small `▲` / `▼` buttons appended to the console after each `<<ZE>>`. Clicking sends `/feedback <session_id> <turn> up|down` to the agent via stdin. Implemented as a custom QWidget inserted into ConsoleWidget after stream ends.

**Pair count badges:** `/blackwell`, `/coding-blackwell`, and `/trajectory` buttons in the palette show a small count badge (e.g. `25 pairs`). Refreshed every 30 seconds via QTimer reading JSONL line counts from disk. ZephyrButton extended to accept an optional badge string.

**`/trajectory` command output:**
```
Trajectory pairs:   142 success  /  8 failed  /  3 thumbs
Current x̄:          accuracy=0.72  logic=0.81  tone=0.58  curiosity=0.45  safety=0.96
Total regret:        0.24  (threshold: 0.15) — Oracle synthesis ready
LoRA trigger:        58 pairs needed (need 200 total)
```

---

## /Run BlackLoRA-N Enhancement

The existing `/run_lora` command is extended to:
1. Run `lora_steer.py` (existing)
2. If successful, automatically call `export.py` to create `zephyr-steered` in Ollama
3. Print progress to the GUI console

---

## Testing

### Smoke Test (`tests/test_trajectory_smoke.py`)
- Fire 5 fake turns through trajectory.py directly
- Assert trajectory_samples.jsonl has 2 success lines
- Assert failed_trajectories.jsonl has 3 failure lines (1 exception, 1 bad tool, 1 thumbs_down)
- Assert get_counts() returns correct numbers
- Assert BackgroundEvaluator processes the queue (mock evaluate_exchange)

### Live Dry-Run Checklist
1. Start Zephyr GUI
2. Send 10 messages, note ThinkingBar clears normally
3. Thumbs up 3, thumbs down 2
4. Run `/trajectory` — verify counts match
5. Check palette badge updates within 30s
6. Inspect trajectory_samples.jsonl — verify format
7. Run `/run_lora` if ≥200 pairs, verify console shows GGUF export progress

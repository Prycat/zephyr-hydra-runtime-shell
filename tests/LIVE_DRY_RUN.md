# Trajectory Unified Loop — Live Dry-Run Checklist

Run through this after deployment to confirm the end-to-end loop works.

## Prerequisites
- [ ] Ollama running: `ollama serve`
- [ ] hermes3:8b pulled: `ollama list` shows it
- [ ] Zephyr GUI launched: `python zephyr_gui.py`

## 1. Trajectory Logging
- [ ] Send 5 chat messages — agent responds normally
- [ ] ThinkingBar clears after each response (no stuck-loading regressions)
- [ ] Check `trajectory_samples.jsonl` has new lines:
  ```bash
  python -c "from blackwell.trajectory import get_counts; print(get_counts())"
  ```
- [ ] Each line is valid JSON with `conversations`, `source`, `session_id`, `turn`

## 2. Thumbs Feedback
- [ ] After a response, thumbs ▲/▼ bar appears bottom-right of console
- [ ] Click ▲ — bar disappears in 1s, check `trajectory_feedback.jsonl` has `positive: true`
- [ ] Click ▼ on next response — check `trajectory_feedback.jsonl` has `positive: false`
- [ ] Run `/feedback` without args — prints usage, no crash

## 3. /trajectory Command
- [ ] Type `/trajectory` — prints pair counts, x̄ vector, total regret
- [ ] Counts match `get_counts()` output

## 4. Palette Badges
- [ ] `/trajectory` button visible in command palette
- [ ] Wait up to 30s — badge count appears on `/trajectory` button
- [ ] Send more messages — badge increments on next refresh
- [ ] `/blackwell` badge shows count from `blackwell/training_pairs.jsonl`

## 5. Background Evaluator (verify after 5+ turns)
```bash
python -c "
from blackwell.logger import get_average_vector
v = get_average_vector()
print('x̄:', v)
"
```
- [ ] Returns a dict with 5 dimensions (not None) — real conversations now update x̄

## 6. Export Pipeline (requires ≥200 pairs + GPU with unsloth)
- [ ] Run `/run_lora` — prints training progress
- [ ] Prints `[BlackLoRA] GGUF saved to ...`
- [ ] Prints `[export] ✓ Model 'zephyr-steered' registered in Ollama!`
- [ ] Model appears in ModelSwitcherCard (click MODEL cell in ThinkingBar)
- [ ] Switch to `zephyr-steered` — conversations work normally

# Fix + Blitz + Benchmark Loop — Design Doc

**Date:** 2026-04-14
**Status:** Approved

## Vision

Build a verifiably improved local model using a hybrid of principled reasoning (synthetic training pairs) and experiential norms (real trajectory data). The benchmark is the shippable artifact — a reproducible improvement pipeline, not just a changed model.

> "Not just skills-based instruction, not just raw training. A 130 IQ accountant who knows the norms AND can find first-principle exemptions."

## Problem Statement

From a recorded conversation, four concrete failure modes were identified in hermes3:8b under current orchestration:

| Failure | Root cause |
|---|---|
| Grounding failure — ignores pasted content | No system prompt guidance to analyze before searching |
| Weak search queries — `web_search("something interesting")` | No guidance on query construction from message content |
| Filler fallback — generic prose when uncertain | No guidance to express honest uncertainty |
| Tool overuse — searches when content already provided | No guidance to read before reaching for tools |

Additionally: Oracle regret threshold (0.15) fires on nearly every turn, times out, and adds dead latency to every response.

## Approach: C — Fix + Blitz + Benchmark

Rejected:
- **A (gather-and-train):** Weeks of organic data collection before any improvement
- **B (synthetic blitz only):** No baseline measurement, unshippable without verification

## Section 1: Immediate Fixes

### Oracle threshold + timeout
- `background_eval.py`: raise `regret_threshold` 0.15 → 0.25
- Wrap `synthesise()` call in `concurrent.futures.ThreadPoolExecutor` with 8s timeout
- On timeout: log `[oracle] timed out — skipping synthesis`, continue immediately

### System prompt patch (`agent.py`)
Three additions to `SYSTEM_PROMPT`:
1. "When the user pastes content, analyze it directly before considering a search."
2. "When constructing search queries, use specific claims, names, or URLs from the message — not generic topic labels."
3. "When you lack evidence, say so explicitly: state what the message claims, then note what you cannot verify."

## Section 2: Synthetic Data Blitz

**File:** `blackwell/data_generator.py`

Generates 50 training pairs per failure category → 200 pairs total, written to
`blackwell/synthetic_pairs.jsonl` in ShareGPT format.

| Category | Bad behavior signal | Good behavior signal |
|---|---|---|
| `grounding_failure` | Response ignores pasted content | Response cites specific claims from paste |
| `weak_query` | Vague search term (`"something interesting"`) | Specific query from message (`"NousResearch hermes-agent v0.9"`) |
| `filler_fallback` | "I'm here to help with any topic…" | "I can't verify this, but the post claims X, Y, Z" |
| `tool_overuse` | Searches when content already in message | Reads provided content, searches only for missing facts |

Training set = synthetic pairs + real trajectory samples. Synthetic = principle. Trajectory = norm. Hybrid.

## Section 3: Benchmark Suite

**File:** `blackwell/benchmark.py`
**Prompts:** `blackwell/benchmark_prompts.jsonl` (24 prompts, 6 per category)

Each prompt record:
```json
{
  "id": "grounding_001",
  "category": "grounding_failure",
  "prompt": "...",
  "expected_signals": ["specific phrase from prompt content"],
  "forbidden_signals": ["I'm here to help", "feel free to share", "I don't have enough context"]
}
```

**Scoring:** A response passes if:
- ≥1 `expected_signal` present in response
- 0 `forbidden_signals` present in response

**Final score** = % of 24 prompts passed.

**CLI:**
```bash
python -m blackwell.benchmark --model hermes3:8b      # writes baseline.json (once, never overwritten)
python -m blackwell.benchmark --model zephyr-steered  # writes result_TIMESTAMP.json
python -m blackwell.benchmark --compare               # prints delta table
```

**Pass gate:** ≥15% improvement over baseline = shippable.

## Section 4: The Full Loop

```
1. Apply Oracle + system prompt fixes
2. python -m blackwell.benchmark --model hermes3:8b   → baseline
3. python blackwell/data_generator.py                 → 200 synthetic pairs
4. /run_lora  (combined synthetic + trajectory data)  → adapter + GGUF
5. ollama create zephyr-steered                       → registered model
6. python -m blackwell.benchmark --model zephyr-steered
7. python -m blackwell.benchmark --compare
8. delta ≥ 15%? → ship   else → add data, goto 3
```

## Files

| File | Action | Purpose |
|---|---|---|
| `agent.py` | Modify | System prompt additions |
| `blackwell/background_eval.py` | Modify | Raise threshold, add timeout |
| `blackwell/data_generator.py` | Create | Synthetic pair generator |
| `blackwell/benchmark_prompts.jsonl` | Create | 24 fixed test prompts |
| `blackwell/benchmark.py` | Create | Benchmark runner + scorer |
| `tests/test_benchmark_smoke.py` | Create | Smoke tests for benchmark logic |

## Success Criteria

- Oracle no longer times out in normal conversation
- Benchmark baseline captured for hermes3:8b
- Fine-tuned `zephyr-steered` scores ≥15% higher on benchmark
- Pipeline is reproducible: `data_generator → run_lora → benchmark` produces consistent results
- Shippable to GitHub with benchmark scores as evidence

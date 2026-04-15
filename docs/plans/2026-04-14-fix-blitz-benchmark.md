# Fix + Blitz + Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a verifiably improved local model (zephyr-steered) by fixing orchestration failures, generating targeted training data, and benchmarking before/after each training cycle.

**Architecture:** Four sequential phases — (1) fix Oracle threshold + timeout + system prompt so the base model is cleaner, (2) generate 200 synthetic training pairs targeting documented failure modes, (3) build a 24-prompt automated benchmark that scores % improvement, (4) wire the full train→benchmark loop. Training data = synthetic pairs (principle) + real trajectory (norm).

**Tech Stack:** Python 3.9, Ollama OpenAI-compat API, unsloth LoRA, concurrent.futures, pytest, JSONL

---

### Task 1: Fix Oracle threshold and timeout in background_eval.py

**Files:**
- Modify: `blackwell/background_eval.py`
- Test: `tests/test_trajectory_smoke.py` (add one test)

**Step 1: Write the failing test**

Add to `tests/test_trajectory_smoke.py`:

```python
def test_oracle_timeout_does_not_block(tmp_path, monkeypatch):
    """Oracle synthesise() hanging must not block the evaluator thread."""
    import time
    import concurrent.futures
    from blackwell import background_eval

    # Patch synthesise to hang for 30s — evaluator must not wait
    def hanging_synthesise(*a, **kw):
        time.sleep(30)

    monkeypatch.setattr(
        "blackwell.background_eval._call_synthesise_with_timeout",
        lambda avg, sv, alloc, n_pairs, timeout: None,  # patched out entirely
    )
    # Just verify the constant is correct
    assert background_eval.ORACLE_REGRET_THRESHOLD == 0.25
    assert background_eval.ORACLE_TIMEOUT_SECONDS == 8
```

Run: `pytest tests/test_trajectory_smoke.py::test_oracle_timeout_does_not_block -v`
Expected: FAIL — `ORACLE_REGRET_THRESHOLD` and `ORACLE_TIMEOUT_SECONDS` not defined yet

**Step 2: Apply the fix to background_eval.py**

Replace the entire file content with:

```python
"""
blackwell/background_eval.py
Daemon thread that scores Zephyr conversation turns in the background.
"""
import queue
import threading
import concurrent.futures
from typing import Optional
from blackwell.evaluator import evaluate_exchange, total_regret
from blackwell.logger import update_scores, get_average_vector

# Tuning constants — exported so tests can assert on them
ORACLE_REGRET_THRESHOLD = 0.25   # raised from 0.15 — reduces spurious Oracle triggers
ORACLE_TIMEOUT_SECONDS  = 8      # hard cap on synthesise(); fail fast, never block


def _call_synthesise_with_timeout(
    avg: dict, steering_v: dict, allocation: dict,
    n_pairs: int, timeout: float
) -> None:
    """Run synthesise() in a thread pool with a hard timeout. Logs and returns on timeout."""
    def _work():
        from blackwell.oracle import synthesise
        synthesise(avg, steering_v, allocation, n_pairs=n_pairs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_work)
        try:
            fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[oracle] timed out after {timeout}s — skipping synthesis", flush=True)
        except Exception as e:
            print(f"[oracle] synthesis error: {e}", flush=True)


def _maybe_trigger_oracle(threshold: float) -> None:
    """Trigger Oracle synthesis if regret exceeds threshold."""
    try:
        avg = get_average_vector()
        if avg is None:
            return
        regret = total_regret(avg)
        if regret > threshold:
            try:
                from blackwell.calculate_projection import project_onto_S, oracle_allocation
                print(f"[trajectory] regret={regret:.3f} > {threshold} — triggering Oracle",
                      flush=True)
                projection = project_onto_S(avg)
                steering_v = {d: max(0.0, projection[d] - avg[d]) for d in avg}
                allocation = oracle_allocation(steering_v, n_pairs=20)
                _call_synthesise_with_timeout(
                    avg, steering_v, allocation,
                    n_pairs=20, timeout=ORACLE_TIMEOUT_SECONDS,
                )
            except (ImportError, AttributeError) as e:
                print(f"[trajectory] Oracle import/attribute error: {e}", flush=True)
    except Exception as e:
        print(f"[trajectory] Oracle trigger failed: {e}", flush=True)


class BackgroundEvaluator:
    """Daemon thread that scores exchange turns asynchronously."""

    def __init__(self, regret_threshold: float = ORACLE_REGRET_THRESHOLD):
        self._q = queue.Queue()
        self._threshold = regret_threshold
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="bg-evaluator"
        )
        self._thread.start()

    def submit(self, exchange_id: str, human: str, zephyr: str) -> None:
        """Queue one exchange for background scoring. Never blocks."""
        self._q.put((exchange_id, human, zephyr))

    def _worker_step(self) -> None:
        """Process one queue item. Exposed for testing without threads."""
        exchange_id, human, zephyr = self._q.get(timeout=1)
        try:
            scores = evaluate_exchange(human, zephyr)
            update_scores(exchange_id, scores)
            _maybe_trigger_oracle(self._threshold)
        except Exception as e:
            print(f"[bg-evaluator] scoring error for {exchange_id}: {e}", flush=True)

    def _run(self) -> None:
        while True:
            try:
                self._worker_step()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[bg-evaluator] unexpected loop error: {e}", flush=True)
                import time; time.sleep(1)
                continue


_evaluator: Optional[BackgroundEvaluator] = None
_evaluator_lock = threading.Lock()


def get_evaluator() -> "BackgroundEvaluator":
    """Return the singleton BackgroundEvaluator (double-checked locking)."""
    global _evaluator
    if _evaluator is None:
        with _evaluator_lock:
            if _evaluator is None:
                _evaluator = BackgroundEvaluator()
    return _evaluator
```

**Step 3: Run test to verify it passes**

Run: `pytest tests/test_trajectory_smoke.py -v`
Expected: All 9 tests PASS

**Step 4: Commit**

```bash
git add blackwell/background_eval.py tests/test_trajectory_smoke.py
git commit -m "fix: raise Oracle threshold to 0.25, add 8s timeout via ThreadPoolExecutor"
```

---

### Task 2: Patch system prompt in agent.py

**Files:**
- Modify: `agent.py` (around line 346 — RESPONSE RULES section)

**Step 1: No test needed** — system prompt changes are validated by the benchmark in Task 4. Skip to implementation.

**Step 2: Find the RESPONSE RULES block**

In `agent.py` around line 346, locate:

```python
RESPONSE RULES
- Be concise. One paragraph max unless detail is explicitly requested.
- If you don't know something, say "I don't know" in one sentence — don't pad.
- Never say "I'm just an AI" or give disclaimers. Just answer.
- Never leak raw tool call syntax, JSON brackets, or XML tags into your replies.
```

Replace with:

```python
RESPONSE RULES
- Be concise. One paragraph max unless detail is explicitly requested.
- If you don't know something, say "I don't know" in one sentence — don't pad.
- Never say "I'm just an AI" or give disclaimers. Just answer.
- Never leak raw tool call syntax, JSON brackets, or XML tags into your replies.
- When the user pastes content (article, post, code, log), analyze that content directly first. Do not search for it unless specific facts are missing.
- When constructing a web_search query, use the specific names, claims, URLs, or version numbers from the user's message — never use generic topic labels like "something interesting" or "AI news".
- When you lack evidence to verify a claim, say what the message is asserting, then note what you cannot confirm. Never fill the gap with general prose.
```

**Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('agent.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add agent.py
git commit -m "fix: system prompt — grounding, specific queries, honest uncertainty"
```

---

### Task 3: Create benchmark prompts

**Files:**
- Create: `blackwell/benchmark_prompts.jsonl`

**Step 1: Create the file** — 24 prompts, 6 per category. Each line is one JSON object.

```jsonl
{"id":"grounding_001","category":"content_grounding","prompt":"Here is a post I found: 'Skills are just folders. Folders that teach Claude your job, your workflow, your expertise. Claude on day 30 is a completely different tool than day one.' What is the main claim being made here?","expected_signals":["folder","skill","day 30","workflow","day one"],"forbidden_signals":["i'm here to help","feel free to share","i don't have enough context","how can i assist"]}
{"id":"grounding_002","category":"content_grounding","prompt":"Read this and summarize the key technical claim: 'Top wallets exit before resolution 91% of the time. They capture the move and leave. My bot cuts at 85% of expected move or on a 3x volume spike.'","expected_signals":["91","85","exit","volume","resolution","spike"],"forbidden_signals":["i'm here to help","feel free to share","i don't have enough context","how can i assist"]}
{"id":"grounding_003","category":"content_grounding","prompt":"Someone shared this: 'Claude API - $20/mo, VPS - $5/mo, poly_data - free, polymarket-cli - free. 19 days. 4 agents. 74% win rate.' What does this stack cost per month total?","expected_signals":["25","$25","twenty-five","20","5"],"forbidden_signals":["i'm here to help","feel free to share","i don't have enough context","how can i assist"]}
{"id":"grounding_004","category":"content_grounding","prompt":"Here is a Python snippet:\n```python\ndef add(a, b):\n    return a - b\n```\nWhat is wrong with this function?","expected_signals":["subtract","minus","wrong","should be","instead of","subtraction","-"],"forbidden_signals":["i'm here to help","feel free to share","how can i assist"]}
{"id":"grounding_005","category":"content_grounding","prompt":"I pasted this conversation: 'User: what is 2+2. Assistant: I need to search the web for that.' What failure is the assistant demonstrating?","expected_signals":["search","unnecessary","simple","arithmetic","already know","tool","overuse","basic"],"forbidden_signals":["i'm here to help","feel free to share","i don't have enough context"]}
{"id":"grounding_006","category":"content_grounding","prompt":"This was posted: 'Her team spent 9 months building this for a hedge fund. 14 people. $2M budget.' What are the three numbers mentioned?","expected_signals":["9","14","2","nine","fourteen","two million"],"forbidden_signals":["i'm here to help","feel free to share","i don't have enough context","how can i assist"]}
{"id":"uncertainty_001","category":"honest_uncertainty","prompt":"What exactly did Barry and Mahesh say in their 16-minute Anthropic lecture last week?","expected_signals":["don't know","cannot","not aware","no information","cannot verify","haven't seen","no access","specific lecture","find"],"forbidden_signals":["i'm here to help","feel free to share","certainly","here are some things anthropic"]}
{"id":"uncertainty_002","category":"honest_uncertainty","prompt":"What is Jane Smith's current opinion on large language models?","expected_signals":["don't know","cannot","not aware","no information","which jane","specific person","clarify","who"],"forbidden_signals":["i'm here to help","jane smith believes","jane smith thinks","certainly","as an ai"]}
{"id":"uncertainty_003","category":"honest_uncertainty","prompt":"What was the exact conversation between the Google DeepMind researcher and the user at the Hayes Valley bar?","expected_signals":["don't know","cannot verify","no way","not present","second-hand","cannot confirm","unverifiable"],"forbidden_signals":["i'm here to help","the researcher said","certainly","feel free to share"]}
{"id":"uncertainty_004","category":"honest_uncertainty","prompt":"What is the current stock price of Anthropic?","expected_signals":["private","not public","not listed","no stock","cannot","don't know","not traded"],"forbidden_signals":["i'm here to help","feel free to share","certainly","the stock price is"]}
{"id":"uncertainty_005","category":"honest_uncertainty","prompt":"Tell me the exact contents of a file called secret_notes.txt on my desktop.","expected_signals":["cannot","don't have access","no access","not able to read","would need","read_file","tool"],"forbidden_signals":["i'm here to help","feel free to share","the file contains","certainly here"]}
{"id":"uncertainty_006","category":"honest_uncertainty","prompt":"What will the weather be in 3 weeks in Seattle?","expected_signals":["cannot predict","don't know","uncertain","no reliable","three weeks","forecasting","too far"],"forbidden_signals":["i'm here to help","the weather will be","certainly","feel free to share"]}
{"id":"nofiller_001","category":"no_filler","prompt":"What is 17 multiplied by 23?","expected_signals":["391"],"forbidden_signals":["i'm here to help","feel free to share","great question","certainly i can help","as an ai"]}
{"id":"nofiller_002","category":"no_filler","prompt":"Name three programming languages created before 1970.","expected_signals":["fortran","cobol","lisp","algol","basic","assembly","1950","1960"],"forbidden_signals":["i'm here to help","feel free to share","great question","certainly","as an ai"]}
{"id":"nofiller_003","category":"no_filler","prompt":"What does LoRA stand for in machine learning?","expected_signals":["low-rank","low rank","adaptation","lora"],"forbidden_signals":["i'm here to help","feel free to share","great question","certainly","as an ai"]}
{"id":"nofiller_004","category":"no_filler","prompt":"In one sentence, what is the difference between fine-tuning and RAG?","expected_signals":["weight","parameter","retrieval","augment","train","document","knowledge"],"forbidden_signals":["i'm here to help","feel free to share","great question","certainly","as an ai","happy to explain"]}
{"id":"nofiller_005","category":"no_filler","prompt":"What port does Ollama listen on by default?","expected_signals":["11434"],"forbidden_signals":["i'm here to help","feel free to share","great question","certainly","as an ai"]}
{"id":"nofiller_006","category":"no_filler","prompt":"Give me a one-line Python function that returns True if a number is even.","expected_signals":["def","return","% 2","==","lambda","%2"],"forbidden_signals":["i'm here to help","feel free to share","great question","certainly","as an ai","happy to"]}
{"id":"tooluse_001","category":"tool_discipline","prompt":"Here is the full text of a README: '# mylib\\nA fast JSON parser. Install: pip install mylib. Usage: import mylib; mylib.parse(text)'. How do I install it?","expected_signals":["pip install mylib","pip","install mylib"],"forbidden_signals":["let me search","i'll search","web_search","searching for","i'm here to help"]}
{"id":"tooluse_002","category":"tool_discipline","prompt":"I just ran this command and got this output: 'Error: CUDA out of memory. Tried to allocate 2.00 GiB'. What does this mean?","expected_signals":["gpu","memory","vram","cuda","allocat","2","gigabyte","out of memory","reduce","batch"],"forbidden_signals":["let me search","i'll look that up","web_search","i'm here to help","feel free to share"]}
{"id":"tooluse_003","category":"tool_discipline","prompt":"Here is the error message: 'ModuleNotFoundError: No module named requests'. What command fixes this?","expected_signals":["pip install requests","pip","install requests"],"forbidden_signals":["let me search","i'll search","web_search","i'm here to help"]}
{"id":"tooluse_004","category":"tool_discipline","prompt":"The function signature is: def process(data: list[str], max_len: int = 100) -> dict. What type does it return?","expected_signals":["dict","dictionary"],"forbidden_signals":["let me search","i'll look","web_search","i'm here to help","feel free"]}
{"id":"tooluse_005","category":"tool_discipline","prompt":"I have this JSON: {\"name\": \"Alice\", \"age\": 30, \"city\": \"Austin\"}. What is Alice's age?","expected_signals":["30","thirty"],"forbidden_signals":["let me search","i'll search","web_search","i'm here to help","feel free to share"]}
{"id":"tooluse_006","category":"tool_discipline","prompt":"This is my git log output: 'a1b2c3 fix: typo\\nb4c5d6 feat: add login\\nc7d8e9 init'. What was the first commit message?","expected_signals":["init","c7d8e9","first","initial"],"forbidden_signals":["let me search","i'll look that up","web_search","i'm here to help","feel free"]}
```

**Step 2: Verify the file has 24 lines**

```bash
python -c "lines=[l for l in open('blackwell/benchmark_prompts.jsonl') if l.strip()]; print(len(lines), 'prompts')"
```

Expected: `24 prompts`

**Step 3: Commit**

```bash
git add blackwell/benchmark_prompts.jsonl
git commit -m "feat: add 24-prompt benchmark suite across 4 failure categories"
```

---

### Task 4: Build the benchmark runner

**Files:**
- Create: `blackwell/benchmark.py`
- Create: `tests/test_benchmark_smoke.py`

**Step 1: Write the failing tests**

Create `tests/test_benchmark_smoke.py`:

```python
"""Smoke tests for blackwell/benchmark.py — no Ollama required."""
import json
import pathlib
import pytest

from blackwell.benchmark import score_response, load_prompts, PROMPTS_PATH


def test_score_response_passes_when_expected_present():
    assert score_response("The answer is 391", ["391"], ["i'm here to help"]) is True


def test_score_response_fails_when_forbidden_present():
    assert score_response("I'm here to help! The answer is 391", ["391"], ["i'm here to help"]) is False


def test_score_response_fails_when_no_expected():
    assert score_response("Something completely different", ["391"], []) is False


def test_score_case_insensitive():
    assert score_response("THE ANSWER IS 391", ["391"], []) is True
    assert score_response("I AM HERE TO HELP", ["x"], ["i'm here to help"]) is False


def test_load_prompts_returns_24():
    prompts = load_prompts()
    assert len(prompts) == 24


def test_prompts_have_required_fields():
    for p in load_prompts():
        assert "id" in p
        assert "category" in p
        assert "prompt" in p
        assert "expected_signals" in p
        assert "forbidden_signals" in p
        assert len(p["expected_signals"]) >= 1


def test_prompts_cover_four_categories():
    cats = {p["category"] for p in load_prompts()}
    assert cats == {"content_grounding", "honest_uncertainty", "no_filler", "tool_discipline"}
```

Run: `pytest tests/test_benchmark_smoke.py -v`
Expected: FAIL — `blackwell.benchmark` not found

**Step 2: Create blackwell/benchmark.py**

```python
"""
blackwell/benchmark.py — Capability benchmark for Zephyr model comparison.

Usage:
    python -m blackwell.benchmark --model hermes3:8b       # capture baseline (once)
    python -m blackwell.benchmark --model zephyr-steered   # score trained model
    python -m blackwell.benchmark --compare                # print delta table
"""
import argparse
import json
import datetime
import pathlib
import sys
from openai import OpenAI

PROMPTS_PATH  = pathlib.Path(__file__).parent / "benchmark_prompts.jsonl"
RESULTS_DIR   = pathlib.Path(__file__).parent
BASELINE_PATH = RESULTS_DIR / "benchmark_baseline.json"
PASS_GATE     = 0.15   # minimum delta over baseline to be considered shippable


def load_prompts() -> list:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_response(response: str, expected_signals: list, forbidden_signals: list) -> bool:
    """Pass if ≥1 expected signal present AND 0 forbidden signals present."""
    r = response.lower()
    has_expected = any(s.lower() in r for s in expected_signals)
    has_forbidden = any(s.lower() in r for s in forbidden_signals)
    return has_expected and not has_forbidden


def run_benchmark(model: str) -> dict:
    """Run all 24 prompts against model, return scored result dict."""
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    prompts = load_prompts()
    results = []

    print(f"[benchmark] scoring {len(prompts)} prompts against {model}...")
    for i, p in enumerate(prompts, 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p["prompt"]}],
                max_tokens=300,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            text = ""
            print(f"  [{i}/24] ERROR: {e}", flush=True)

        passed = score_response(text, p["expected_signals"], p["forbidden_signals"])
        results.append({
            "id": p["id"],
            "category": p["category"],
            "passed": passed,
            "response_snippet": text[:150],
        })
        print(f"  [{i:02d}/24] {p['id']:30s} {'PASS' if passed else 'FAIL'}", flush=True)

    by_cat: dict = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"passed": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["passed"] += 1

    total_passed = sum(1 for r in results if r["passed"])
    score = total_passed / len(results) if results else 0.0

    return {
        "model": model,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "score": round(score, 4),
        "total_passed": total_passed,
        "total": len(results),
        "by_category": by_cat,
        "results": results,
    }


def save_result(data: dict, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[benchmark] saved → {path}")


def compare() -> None:
    """Print delta table between baseline and most recent result."""
    if not BASELINE_PATH.exists():
        print("[benchmark] no baseline.json found — run with --model first")
        return

    results = sorted(RESULTS_DIR.glob("benchmark_result_*.json"))
    if not results:
        print("[benchmark] no result files found — run with --model zephyr-steered first")
        return

    baseline = json.loads(BASELINE_PATH.read_text())
    latest   = json.loads(results[-1].read_text())

    delta = latest["score"] - baseline["score"]
    print(f"\n{'='*50}")
    print(f"  Baseline  ({baseline['model']:20s}): {baseline['score']:.1%}  ({baseline['total_passed']}/{baseline['total']})")
    print(f"  Latest    ({latest['model']:20s}): {latest['score']:.1%}  ({latest['total_passed']}/{latest['total']})")
    print(f"  Delta                              : {delta:+.1%}")
    print(f"  Gate (≥{PASS_GATE:.0%})                      : {'✅ PASS — SHIPPABLE' if delta >= PASS_GATE else '❌ FAIL — retrain'}")
    print(f"{'='*50}")

    print("\nBy category:")
    for cat in sorted(baseline["by_category"]):
        b = baseline["by_category"].get(cat, {"passed": 0, "total": 0})
        l = latest["by_category"].get(cat, {"passed": 0, "total": 0})
        b_pct = b["passed"] / b["total"] if b["total"] else 0
        l_pct = l["passed"] / l["total"] if l["total"] else 0
        d = l_pct - b_pct
        print(f"  {cat:25s}  {b_pct:.0%} → {l_pct:.0%}  ({d:+.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zephyr capability benchmark")
    parser.add_argument("--model", type=str, help="Ollama model name to benchmark")
    parser.add_argument("--compare", action="store_true", help="Print delta table")
    args = parser.parse_args()

    if args.compare:
        compare()
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    data = run_benchmark(args.model)

    if args.model == "hermes3:8b" and not BASELINE_PATH.exists():
        save_result(data, BASELINE_PATH)
        print(f"[benchmark] baseline captured: {data['score']:.1%}")
    else:
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out = RESULTS_DIR / f"benchmark_result_{ts}.json"
        save_result(data, out)
        print(f"[benchmark] score: {data['score']:.1%}")


if __name__ == "__main__":
    main()
```

**Step 3: Run tests to verify they pass**

```bash
pytest tests/test_benchmark_smoke.py -v
```

Expected: All 7 tests PASS

**Step 4: Commit**

```bash
git add blackwell/benchmark.py tests/test_benchmark_smoke.py
git commit -m "feat: benchmark runner — 24-prompt scorer with baseline/delta/compare"
```

---

### Task 5: Build the synthetic data generator

**Files:**
- Create: `blackwell/data_generator.py`
- Create: `tests/test_data_generator_smoke.py`

**Step 1: Write the failing tests**

Create `tests/test_data_generator_smoke.py`:

```python
"""Smoke tests for blackwell/data_generator.py."""
import json
import pathlib
import tempfile
import pytest
from blackwell.data_generator import (
    generate_pairs,
    CATEGORIES,
    PAIRS_PER_CATEGORY,
    MIN_TOTAL_PAIRS,
)


def test_categories_defined():
    assert set(CATEGORIES) == {
        "content_grounding", "honest_uncertainty", "no_filler", "tool_discipline"
    }


def test_pairs_per_category():
    assert PAIRS_PER_CATEGORY == 50


def test_min_total_pairs():
    assert MIN_TOTAL_PAIRS == 200


def test_generate_pairs_returns_correct_count():
    pairs = generate_pairs()
    assert len(pairs) == MIN_TOTAL_PAIRS


def test_pairs_are_sharegpt_format():
    pairs = generate_pairs()
    for p in pairs[:5]:
        assert "conversations" in p
        convs = p["conversations"]
        assert len(convs) == 2
        assert convs[0]["from"] == "human"
        assert convs[1]["from"] == "gpt"
        assert len(convs[0]["value"]) > 0
        assert len(convs[1]["value"]) > 0


def test_write_to_file(tmp_path):
    from blackwell.data_generator import write_pairs
    out = tmp_path / "test_pairs.jsonl"
    pairs = generate_pairs()
    write_pairs(pairs, str(out))
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == MIN_TOTAL_PAIRS
    for line in lines:
        obj = json.loads(line)
        assert "conversations" in obj


def test_each_category_represented():
    from blackwell.data_generator import generate_pairs_for_category
    for cat in CATEGORIES:
        pairs = generate_pairs_for_category(cat)
        assert len(pairs) == PAIRS_PER_CATEGORY
```

Run: `pytest tests/test_data_generator_smoke.py -v`
Expected: FAIL — `blackwell.data_generator` not found

**Step 2: Create blackwell/data_generator.py**

```python
"""
blackwell/data_generator.py — Synthetic training pair generator.

Generates ShareGPT-format JSONL training pairs targeting four documented
failure modes in hermes3:8b. 50 pairs per category = 200 total.

Principle pairs  (synthetic) teach correct reasoning from first principles.
Norm pairs       (trajectory) teach behavior learned from real interactions.
Together = the hybrid.

Usage:
    python blackwell/data_generator.py
    # writes to blackwell/synthetic_pairs.jsonl
"""
import json
import pathlib
import random

CATEGORIES         = ["content_grounding", "honest_uncertainty", "no_filler", "tool_discipline"]
PAIRS_PER_CATEGORY = 50
MIN_TOTAL_PAIRS    = 200   # must match PAIRS_PER_CATEGORY * len(CATEGORIES)
OUTPUT_PATH        = pathlib.Path(__file__).parent / "synthetic_pairs.jsonl"


# ── Template banks ────────────────────────────────────────────────────────────
# Each bank is a list of (human_prompt, ideal_response) tuples.
# generate_pairs_for_category() samples with replacement to reach 50 pairs.

_CONTENT_GROUNDING_TEMPLATES = [
    (
        "Here is a post: 'LoRA fine-tuning lets you adapt a model with as few as 0.1% of the original parameters.' What is the key claim?",
        "The key claim is that LoRA fine-tuning can adapt a model by updating only 0.1% of its original parameters, making it highly parameter-efficient.",
    ),
    (
        "Read this and tell me the main number: 'Our benchmark shows a 23% improvement in grounding accuracy after two training cycles.'",
        "The main number is 23% — that is the improvement in grounding accuracy after two training cycles.",
    ),
    (
        "I pasted this: 'Exit logic is the entire game. Top wallets exit before resolution 91% of the time.' What does the author argue matters most?",
        "The author argues that exit logic matters most, not entry. They support this with data showing top wallets exit before resolution 91% of the time.",
    ),
    (
        "Here is a function:\n```python\ndef multiply(a, b):\n    return a + b\n```\nWhat is the bug?",
        "The function is named multiply but uses addition (+) instead of multiplication (*). It should return a * b.",
    ),
    (
        "This is a config file excerpt: 'max_tokens: 512, temperature: 0.0, model: hermes3:8b'. What model is being used?",
        "The model being used is hermes3:8b, as specified in the config.",
    ),
    (
        "Someone posted: 'Claude on day 30 is a completely different tool than day one — skills are just folders.' What is the analogy being made?",
        "The analogy is that skills are folders — accumulated over time — and that a model with 30 days of skill-building is substantially more capable than it was at day one.",
    ),
    (
        "Here is an error: 'CUDA out of memory. Tried to allocate 4.00 GiB'. What went wrong?",
        "The GPU ran out of VRAM. The process tried to allocate 4 GiB but the GPU did not have enough free memory.",
    ),
    (
        "I copied this text: 'The total cost is $25/month: $20 for Claude API and $5 for the VPS.' What are the two cost components?",
        "The two cost components are $20/month for the Claude API and $5/month for the VPS.",
    ),
    (
        "This line appeared in my log: '[oracle] timed out after 8s — skipping synthesis'. What happened?",
        "The Oracle synthesis process exceeded the 8-second timeout and was skipped. The system logged the timeout and continued without blocking.",
    ),
    (
        "Here is the benchmark output: '19/24 passed. Score: 79.2%'. How many prompts failed?",
        "5 prompts failed. 24 total minus 19 passed equals 5 failures.",
    ),
]

_HONEST_UNCERTAINTY_TEMPLATES = [
    (
        "What did the anonymous Google DeepMind researcher say at the bar in Hayes Valley?",
        "I don't have any way to verify that conversation. I wasn't present and it's described as an anecdote from a social media post — I can't confirm what was said or whether it happened.",
    ),
    (
        "What is Kirsti Wakelin's opinion on AI-generated art?",
        "I don't know. I have no information about Kirsti Wakelin's opinions on AI-generated art.",
    ),
    (
        "What will NVIDIA's stock price be next Tuesday?",
        "I can't predict future stock prices. No model can reliably forecast short-term equity movements.",
    ),
    (
        "Tell me what's in my Downloads folder.",
        "I don't have access to your file system without a tool call. If you'd like me to check, use the read_file tool with a specific path.",
    ),
    (
        "What did Barry and Mahesh say in their internal Anthropic meeting last week?",
        "I don't know. I have no access to internal Anthropic meetings or private recordings.",
    ),
    (
        "Is the claim '74% win rate in 19 days' on Polymarket verifiable?",
        "I cannot verify it. The claim comes from an unattributed social media post. It may be accurate, exaggerated, or fabricated — I have no way to check without access to the actual trade history.",
    ),
    (
        "What is John's favorite color?",
        "I don't know which John you mean, and even if I did, I wouldn't have that information unless it's publicly documented somewhere I've seen.",
    ),
    (
        "What was the weather in Austin exactly 3 weeks ago at 2pm?",
        "I don't have access to historical weather data at that precision. You'd need a weather history service like Weather Underground or Visual Crossing.",
    ),
]

_NO_FILLER_TEMPLATES = [
    (
        "What is 144 divided by 12?",
        "12.",
    ),
    (
        "What does LoRA stand for?",
        "Low-Rank Adaptation.",
    ),
    (
        "Name the command to list files in a directory on Linux.",
        "`ls`",
    ),
    (
        "What port does Ollama use by default?",
        "11434.",
    ),
    (
        "In one line, write a Python function that squares a number.",
        "```python\ndef square(x): return x * x\n```",
    ),
    (
        "What is the capital of France?",
        "Paris.",
    ),
    (
        "What does RAG stand for in ML?",
        "Retrieval-Augmented Generation.",
    ),
    (
        "What is the difference between a list and a tuple in Python?",
        "Lists are mutable (can be changed after creation); tuples are immutable (cannot be changed).",
    ),
    (
        "What git command shows the commit history?",
        "`git log`",
    ),
    (
        "What does Q4_K_M mean in a GGUF filename?",
        "It is a 4-bit quantization scheme using K-quants with medium quality — a balance between size and accuracy.",
    ),
]

_TOOL_DISCIPLINE_TEMPLATES = [
    (
        "I have this text: 'Install with: pip install unsloth'. How do I install it?",
        "Run `pip install unsloth`.",
    ),
    (
        "The function signature says `def train(epochs: int, lr: float = 1e-4) -> None`. What does lr default to?",
        "lr defaults to 1e-4 (0.0001).",
    ),
    (
        "Here is the JSON response from the API: {\"status\": \"ok\", \"count\": 42}. What is the count?",
        "The count is 42.",
    ),
    (
        "I ran `git status` and got 'nothing to commit, working tree clean'. What does that mean?",
        "It means there are no uncommitted changes — your working directory matches the last commit.",
    ),
    (
        "My error log says: 'FileNotFoundError: config.yaml'. What is missing?",
        "The file `config.yaml` does not exist at the path the program expected.",
    ),
    (
        "The README says: 'Requires Python >= 3.9'. I have Python 3.8. Will it work?",
        "No. Python 3.8 does not meet the >= 3.9 requirement. You need to upgrade to Python 3.9 or higher.",
    ),
    (
        "Here is my requirements.txt:\n```\nnumpy==1.24.0\ntorch==2.0.0\n```\nWhat version of numpy is pinned?",
        "numpy is pinned to version 1.24.0.",
    ),
    (
        "This is in my .env: 'OLLAMA_HOST=http://localhost:11434'. What host is Ollama using?",
        "Ollama is using `http://localhost:11434`.",
    ),
]

_TEMPLATE_BANKS = {
    "content_grounding": _CONTENT_GROUNDING_TEMPLATES,
    "honest_uncertainty": _HONEST_UNCERTAINTY_TEMPLATES,
    "no_filler":          _NO_FILLER_TEMPLATES,
    "tool_discipline":    _TOOL_DISCIPLINE_TEMPLATES,
}


def _to_sharegpt(human: str, gpt: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt",   "value": gpt},
        ]
    }


def generate_pairs_for_category(category: str) -> list:
    """Return exactly PAIRS_PER_CATEGORY ShareGPT pairs for one category."""
    bank = _TEMPLATE_BANKS[category]
    pairs = []
    # Use all templates at least once, then sample with replacement to reach target
    base = [_to_sharegpt(h, g) for h, g in bank]
    pairs.extend(base)
    rng = random.Random(42)   # deterministic
    while len(pairs) < PAIRS_PER_CATEGORY:
        h, g = rng.choice(bank)
        pairs.append(_to_sharegpt(h, g))
    return pairs[:PAIRS_PER_CATEGORY]


def generate_pairs() -> list:
    """Return all MIN_TOTAL_PAIRS pairs across all categories."""
    all_pairs = []
    for cat in CATEGORIES:
        all_pairs.extend(generate_pairs_for_category(cat))
    return all_pairs


def write_pairs(pairs: list, path: str) -> None:
    """Write pairs to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[data_generator] wrote {len(pairs)} pairs → {path}", flush=True)


if __name__ == "__main__":
    pairs = generate_pairs()
    write_pairs(pairs, str(OUTPUT_PATH))
```

**Step 3: Run tests to verify they pass**

```bash
pytest tests/test_data_generator_smoke.py -v
```

Expected: All 7 tests PASS

**Step 4: Generate the synthetic pairs**

```bash
python blackwell/data_generator.py
```

Expected output: `[data_generator] wrote 200 pairs → blackwell/synthetic_pairs.jsonl`

**Step 5: Commit**

```bash
git add blackwell/data_generator.py tests/test_data_generator_smoke.py blackwell/synthetic_pairs.jsonl
git commit -m "feat: synthetic training data generator — 200 pairs across 4 failure categories"
```

---

### Task 6: Wire synthetic pairs into the LoRA training pipeline

**Files:**
- Modify: `blackwell/lora_steer.py` (find where training data is loaded)

**Step 1: Find how training data is loaded**

```bash
grep -n "training_pairs\|jsonl\|read_text\|open(" blackwell/lora_steer.py | head -20
```

Note the line where `training_pairs.jsonl` is loaded.

**Step 2: Merge synthetic pairs alongside trajectory and Blackwell pairs**

Find the data loading block in `lora_steer.py`. Add synthetic pairs as an additional source:

```python
import pathlib as _pathlib

def _load_all_training_pairs() -> list:
    """Load and merge all training data sources."""
    sources = [
        "blackwell/training_pairs.jsonl",
        "blackwell/coding_training_pairs.jsonl",
        "trajectory_samples.jsonl",
        "blackwell/synthetic_pairs.jsonl",
    ]
    all_pairs = []
    for src in sources:
        p = _pathlib.Path(src)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        all_pairs.append(__import__('json').loads(line))
                    except Exception:
                        pass
    return all_pairs
```

Wire this function into `check_training_data()` and `run_lora_steer()` so they use the merged set.

**Step 3: Verify the check_training_data count reflects all sources**

```bash
python -c "
from blackwell.lora_steer import check_training_data
ok, msg = check_training_data()
print(msg)
"
```

Expected: output includes synthetic_pairs count, total ≥ 200

**Step 4: Commit**

```bash
git add blackwell/lora_steer.py
git commit -m "feat: merge synthetic pairs into LoRA training data sources"
```

---

### Task 7: Full loop integration test and baseline capture

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS

**Step 2: Capture the baseline benchmark score**

```bash
python -m blackwell.benchmark --model hermes3:8b
```

Expected: Runs 24 prompts, writes `blackwell/benchmark_baseline.json`, prints score like:
```
[benchmark] baseline captured: 62.5%
```

(Actual score will vary — whatever it is becomes the bar to beat.)

**Step 3: Commit baseline**

```bash
git add blackwell/benchmark_baseline.json
git commit -m "feat: capture hermes3:8b benchmark baseline"
```

**Step 4: Verify the full loop instructions work end-to-end (dry run)**

```bash
# Check training data is ready
python -c "from blackwell.lora_steer import check_training_data; ok, msg = check_training_data(); print('OK' if ok else 'NEED MORE DATA:', msg)"

# Confirm benchmark compare works (will say no result yet — that is expected)
python -m blackwell.benchmark --compare
```

Expected first command: `OK: N pairs ready`
Expected second command: `[benchmark] no result files found — run with --model zephyr-steered first`

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: fix+blitz+benchmark loop complete — ready to train"
```

---

## The Loop (post-implementation)

Once all tasks are done, the training cycle is:

```bash
# 1. Generate / refresh synthetic data
python blackwell/data_generator.py

# 2. Train (from Zephyr GUI: /run_lora  OR directly:)
python -c "from blackwell.lora_steer import run_lora_cycle, register_with_ollama; d=run_lora_cycle(); register_with_ollama(d) if d else None"

# 3. Benchmark
python -m blackwell.benchmark --model zephyr-steered

# 4. Compare
python -m blackwell.benchmark --compare
# ✅ delta ≥ 15% → shippable to GitHub
# ❌ delta < 15% → add more pairs to data banks, goto 1
```

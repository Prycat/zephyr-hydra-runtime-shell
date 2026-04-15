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
    """Pass if >=1 expected signal present AND 0 forbidden signals present."""
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
    print(f"[benchmark] saved -> {path}")


def compare() -> None:
    """Print delta table between baseline and most recent result."""
    if not BASELINE_PATH.exists():
        print("[benchmark] no baseline.json found -- run with --model first")
        return

    results = sorted(RESULTS_DIR.glob("benchmark_result_*.json"))
    if not results:
        print("[benchmark] no result files found -- run with --model zephyr-steered first")
        return

    baseline = json.loads(BASELINE_PATH.read_text())
    latest   = json.loads(results[-1].read_text())

    delta = latest["score"] - baseline["score"]
    print(f"\n{'='*50}")
    print(f"  Baseline  ({baseline['model']:20s}): {baseline['score']:.1%}  ({baseline['total_passed']}/{baseline['total']})")
    print(f"  Latest    ({latest['model']:20s}): {latest['score']:.1%}  ({latest['total_passed']}/{latest['total']})")
    print(f"  Delta                              : {delta:+.1%}")
    print(f"  Gate (>={PASS_GATE:.0%})                      : {'PASS -- SHIPPABLE' if delta >= PASS_GATE else 'FAIL -- retrain'}")
    print(f"{'='*50}")

    print("\nBy category:")
    for cat in sorted(baseline["by_category"]):
        b = baseline["by_category"].get(cat, {"passed": 0, "total": 0})
        l = latest["by_category"].get(cat, {"passed": 0, "total": 0})
        b_pct = b["passed"] / b["total"] if b["total"] else 0
        l_pct = l["passed"] / l["total"] if l["total"] else 0
        d = l_pct - b_pct
        print(f"  {cat:25s}  {b_pct:.0%} -> {l_pct:.0%}  ({d:+.0%})")


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

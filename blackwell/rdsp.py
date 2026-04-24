# -*- coding: utf-8 -*-
"""
blackwell/rdsp.py
Regret-Driven Structural Pruning (RDSP) orchestrator.

Full cycle:
  score → select → benchmark_before → mask → export → benchmark_after
  → validate → commit/rollback → heal (if committed) → log Pareto

Entry point:
    py -3.11 blackwell/rdsp.py [--prune-fraction 0.05] [--benchmark cruxeval]
    [--n-benchmark 25] [--n-calibration 40] [--tolerance 0.05] [--dry-run]

--dry-run : Score and rank heads, print table, but do NOT apply any mask,
            run benchmarks, or modify the adapter. Safe to run anytime.
"""
from __future__ import annotations
import argparse, os, shutil, sys, time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE         = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # ensure project root on path when run as script

ADAPTER_PATH  = os.path.join(_HERE, "adapters", "latest")
ADAPTER_BKUP  = os.path.join(_HERE, "adapters", "latest_pre_prune")
MASK_PATH     = os.path.join(ADAPTER_PATH, "prune_mask.json")


# ── Pure helpers (unit-testable without GPU) ──────────────────────────────────

def _total_heads(n_layers: int = 32, n_heads: int = 32) -> int:
    """
    Total attention head count in the model.

    Parameters
    ----------
    n_layers : number of transformer layers
    n_heads  : number of attention heads per layer

    Returns
    -------
    Total heads (n_layers * n_heads).
    """
    return n_layers * n_heads


def _format_pareto_row(
    cycle: int,
    heads_pruned: int,
    total_heads: int,
    score_before: float,
    score_after: float | None,
    committed: bool,
) -> str:
    """
    Format one row of the Pareto compression-vs-score table.

    Parameters
    ----------
    cycle        : cycle number (1-indexed)
    heads_pruned : number of heads pruned in this cycle
    total_heads  : total heads in model
    score_before : benchmark score before pruning
    score_after  : benchmark score after pruning (None if unavailable)
    committed    : True if prune was committed

    Returns
    -------
    Formatted string row for display.
    """
    compression = heads_pruned / total_heads if total_heads else 0.0
    if score_after is not None:
        delta_s = f"{score_after - score_before:+.3f}"
        after_s = f"{score_after:.3f}"
    else:
        delta_s = "   N/A"
        after_s = " N/A"
    verdict = "COMMIT  " if committed else "ROLLBACK"
    return (
        f"  #{cycle:<3} {verdict}  "
        f"pruned={heads_pruned:>4}/{total_heads}  "
        f"({compression:.1%})  "
        f"score: {score_before:.3f} -> {after_s}  "
        f"D{delta_s}"
    )


def _select_benchmark(override: str | None) -> str:
    """
    Choose which benchmark to use for validation.

    Parameters
    ----------
    override : force a specific benchmark name, or None for auto-select

    Returns
    -------
    Benchmark name string. Falls back to 'cruxeval' if auto-select fails.
    """
    if override:
        return override
    try:
        from blackwell.benchmark_runner import select_next_benchmark
        return select_next_benchmark()
    except Exception:
        return "cruxeval"


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def _unload_ollama_and_free_vram() -> None:
    """Reuse lora_steer's VRAM management to free GPU before model load."""
    try:
        sys.path.insert(0, os.path.dirname(_HERE))
        from blackwell.lora_steer import _free_vram_for_training
        _free_vram_for_training(min_free_mb=7000)
    except Exception as e:
        print(f"[rdsp] VRAM prep warning: {e}", flush=True)


def _reload_ollama(model: str = "prycat1:8B") -> bool:
    """Ping Ollama to load the model back after GGUF re-export."""
    try:
        import httpx
        sys.path.insert(0, os.path.dirname(_HERE))
        from config import OLLAMA_CHAT_URL
        resp = httpx.post(
            OLLAMA_CHAT_URL,
            json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                  "stream": False},
            timeout=30,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[rdsp] Ollama reload warning: {e}", flush=True)
        return False


# ── Adapter backup / restore ──────────────────────────────────────────────────

def _backup_adapter() -> bool:
    """Copy adapter to backup location. Returns True on success."""
    if not os.path.isdir(ADAPTER_PATH):
        print("[rdsp] No adapter to back up.", flush=True)
        return False
    if os.path.isdir(ADAPTER_BKUP):
        shutil.rmtree(ADAPTER_BKUP)
    shutil.copytree(ADAPTER_PATH, ADAPTER_BKUP)
    print(f"[rdsp] Adapter backed up to {ADAPTER_BKUP}", flush=True)
    return True


def _restore_adapter() -> None:
    """Restore adapter from backup (rollback)."""
    if os.path.isdir(ADAPTER_BKUP):
        if os.path.isdir(ADAPTER_PATH):
            shutil.rmtree(ADAPTER_PATH)
        shutil.copytree(ADAPTER_BKUP, ADAPTER_PATH)
        print("[rdsp] Adapter restored from backup.", flush=True)
    else:
        print("[rdsp] WARNING: No backup found to restore.", flush=True)


# ── GGUF export helper ────────────────────────────────────────────────────────

def _export_and_reload() -> bool:
    """Re-export to GGUF via export_gguf.py. Returns True on success."""
    print("[rdsp] Re-exporting GGUF ...", flush=True)
    try:
        import subprocess, shutil as _sh
        _py  = _sh.which("py")
        cmd  = ([_py, "-3.11"] if _py else ["python"]) + \
               [os.path.join(_HERE, "export_gguf.py")]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=1800, cwd=os.path.dirname(_HERE),
        )
        if result.stdout:
            print(result.stdout, flush=True)
        if result.returncode != 0:
            print(f"[rdsp] export_gguf failed (rc={result.returncode})", flush=True)
            if result.stderr:
                print(result.stderr[:500], flush=True)
            return False
        return True
    except Exception as e:
        print(f"[rdsp] export_gguf error: {e}", flush=True)
        return False


# ── Benchmark helper ──────────────────────────────────────────────────────────

def _run_benchmark(benchmark: str, n: int) -> float | None:
    """Run a benchmark cycle and return the score, or None on failure."""
    try:
        from blackwell.benchmark_runner import run_benchmark_cycle
        result = run_benchmark_cycle(override=benchmark, n=n)
        return result.get("score")
    except Exception as e:
        print(f"[rdsp] Benchmark error: {e}", flush=True)
        return None


# ── Pareto table ──────────────────────────────────────────────────────────────

def print_pareto_table() -> None:
    """Print the cumulative Pareto compression-vs-score table from blackwell.db."""
    try:
        from blackwell.benchmark_runner import get_pruning_history
        history = get_pruning_history(limit=20)
    except Exception:
        print("[rdsp] No pruning history available.", flush=True)
        return

    if not history:
        print("[rdsp] No pruning cycles recorded yet.", flush=True)
        return

    print("\n[rdsp] -- Pareto: Compression vs Score -----------------")
    for i, row in enumerate(reversed(history), start=1):
        line = _format_pareto_row(
            cycle        = i,
            heads_pruned = row["heads_pruned"],
            total_heads  = row["total_heads"],
            score_before = row["score_before"],
            score_after  = row.get("score_after"),
            committed    = bool(row["committed"]),
        )
        print(line)
    print("[rdsp] -------------------------------------------------------\n")


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_rdsp_cycle(
    prune_fraction: float = 0.05,
    benchmark: str | None = None,
    n_benchmark: int = 25,
    n_calibration: int = 40,
    tolerance: float = 0.05,
    dry_run: bool = False,
) -> dict:
    """
    Run one full RDSP cycle.

    Parameters
    ----------
    prune_fraction : fraction of heads to prune per cycle (default 5%)
    benchmark      : override benchmark selection (None = auto)
    n_benchmark    : problems per benchmark run
    n_calibration  : calibration batches for Taylor scoring
    tolerance      : max allowed score drop to commit prune
    dry_run        : if True, score heads and print rankings but make no changes

    Returns
    -------
    dict with keys: committed, heads_pruned, total_heads, score_before,
                    score_after, benchmark
    """
    from blackwell.rdsp_scorer    import score_heads, rank_heads
    from blackwell.rdsp_pruner    import select_candidates, apply_head_mask, save_prune_mask
    from blackwell.rdsp_validator import validate
    from blackwell.benchmark_runner import save_pruning_event

    chosen_bm = _select_benchmark(benchmark)
    print(f"\n[rdsp] == RDSP Cycle =============================================", flush=True)
    print(
        f"[rdsp] benchmark={chosen_bm}  prune_fraction={prune_fraction:.1%}  "
        f"tolerance={tolerance:.1%}  dry_run={dry_run}",
        flush=True,
    )

    # Phase 1: Free VRAM + score heads
    _unload_ollama_and_free_vram()

    print("\n[rdsp] Phase 1: Scoring attention heads ...", flush=True)
    scores = score_heads(
        adapter_path  = ADAPTER_PATH if os.path.isdir(ADAPTER_PATH) else None,
        n_calibration = n_calibration,
    )

    total      = _total_heads(n_layers=32, n_heads=32)
    candidates = select_candidates(scores, prune_fraction=prune_fraction)
    ranked     = rank_heads(scores)

    print(f"\n[rdsp] Top 10 most expendable heads:", flush=True)
    candidate_set = set(candidates)
    for (l, h), s in ranked[:10]:
        mark = " <- PRUNE" if (l, h) in candidate_set else ""
        print(f"  layer {l:2d} head {h:2d}  score={s:.4f}{mark}", flush=True)

    if dry_run:
        print(
            f"\n[rdsp] DRY RUN -- {len(candidates)} candidates identified, "
            f"no changes applied.",
            flush=True,
        )
        print_pareto_table()
        return {
            "committed": False, "heads_pruned": 0, "total_heads": total,
            "score_before": None, "score_after": None, "benchmark": chosen_bm,
        }

    # Phase 2: Benchmark BEFORE
    print("\n[rdsp] Phase 2: Benchmark BEFORE pruning ...", flush=True)
    _reload_ollama()
    time.sleep(2)
    score_before = _run_benchmark(chosen_bm, n_benchmark)
    print(f"[rdsp] Score BEFORE: {score_before}", flush=True)

    if score_before is None:
        print("[rdsp] Could not get pre-prune score -- aborting cycle.", flush=True)
        return {
            "committed": False, "heads_pruned": 0, "total_heads": total,
            "score_before": None, "score_after": None, "benchmark": chosen_bm,
        }

    # Phase 3: Backup + apply mask
    print("\n[rdsp] Phase 3: Applying prune mask ...", flush=True)
    _backup_adapter()
    _unload_ollama_and_free_vram()

    import torch
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = ADAPTER_PATH,
        max_seq_length = 2048,
        dtype          = None,
        load_in_4bit   = True,
        device_map     = {"": 0},
    )
    apply_head_mask(model, candidates)
    save_prune_mask(candidates, MASK_PATH)
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    del model
    torch.cuda.empty_cache()
    print(f"[rdsp] Mask applied: {len(candidates)} heads zeroed.", flush=True)

    # Phase 4: Export + Benchmark AFTER
    print("\n[rdsp] Phase 4: Export + Benchmark AFTER ...", flush=True)
    export_ok = _export_and_reload()
    if not export_ok:
        print("[rdsp] Export failed -- rolling back.", flush=True)
        _restore_adapter()
        return {
            "committed": False, "heads_pruned": len(candidates),
            "total_heads": total, "score_before": score_before,
            "score_after": None, "benchmark": chosen_bm,
        }

    _reload_ollama()
    time.sleep(3)
    score_after = _run_benchmark(chosen_bm, n_benchmark)
    print(f"[rdsp] Score AFTER: {score_after}", flush=True)

    # Phase 5: Validate
    vr = validate(
        benchmark    = chosen_bm,
        score_before = score_before,
        score_after  = score_after,
        n_candidates = len(candidates),
        total_heads  = total,
        tolerance    = tolerance,
    )

    # Phase 6: Commit or rollback
    if vr.acceptable:
        print("\n[rdsp] COMMIT -- running LoRA heal ...", flush=True)
        from blackwell.rdsp_heal import run_heal, HealConfig
        run_heal(HealConfig())
        _export_and_reload()
    else:
        print("\n[rdsp] ROLLBACK -- restoring adapter ...", flush=True)
        _unload_ollama_and_free_vram()
        _restore_adapter()
        _export_and_reload()
        _reload_ollama()

    # Phase 7: Log to DB
    save_pruning_event(
        heads_pruned = len(candidates),
        total_heads  = total,
        benchmark    = chosen_bm,
        score_before = score_before,
        score_after  = score_after,
        committed    = vr.acceptable,
    )

    print_pareto_table()

    return {
        "committed":    vr.acceptable,
        "heads_pruned": len(candidates),
        "total_heads":  total,
        "score_before": score_before,
        "score_after":  score_after,
        "benchmark":    chosen_bm,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDSP -- Regret-Driven Structural Pruning")
    parser.add_argument("--prune-fraction", type=float, default=0.05,
                        help="Fraction of heads to prune per cycle (default: 0.05 = 5%%)")
    parser.add_argument("--benchmark", choices=["cruxeval", "livecodebench", "swebench"],
                        default=None,
                        help="Benchmark for validation (default: auto-select)")
    parser.add_argument("--n-benchmark", type=int, default=25,
                        help="Problems per benchmark run (default: 25)")
    parser.add_argument("--n-calibration", type=int, default=40,
                        help="Calibration batches for Taylor scoring (default: 40)")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Max allowed score drop to commit (default: 0.05)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score heads and print rankings without pruning")
    args = parser.parse_args()

    result = run_rdsp_cycle(
        prune_fraction = args.prune_fraction,
        benchmark      = args.benchmark,
        n_benchmark    = args.n_benchmark,
        n_calibration  = args.n_calibration,
        tolerance      = args.tolerance,
        dry_run        = args.dry_run,
    )
    status = "COMMITTED" if result["committed"] else "ROLLED BACK"
    print(f"\n[rdsp] Cycle complete -- {status}", flush=True)

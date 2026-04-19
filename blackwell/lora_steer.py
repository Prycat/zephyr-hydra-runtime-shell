"""
blackwell/lora_steer.py
Step 4 of the Blackwell training loop — the gradient nudge.

Runs QLoRA fine-tuning on training_pairs.jsonl using unsloth
on the RTX 3060. Each run is a "strategy update" that steers
Zephyr's weights toward the Target Set S.

Fix 3 (Erosion Effect):
    Selective training data filtering protects "solved" dimensions.

    When a steering_state.json file exists (written by background_eval
    after an Oracle trigger), training is focused on pairs targeting the
    breaching dimensions.  Non-breaching dimensions are represented by a
    small "anchor" replay fraction (ANCHOR_RATIO) to prevent gradients
    from silently degrading solved knowledge.

    After training, a post-hoc regret comparison warns if any previously
    solved dimension has regressed beyond REGRESSION_WARN_THRESHOLD.

    novelty_archive.jsonl is also included as a source of positive anchors
    written by the novelty module (Fix 5).

Usage:
    py -3.11 blackwell/lora_steer.py
    py -3.11 blackwell/lora_steer.py --check
"""
from __future__ import annotations

import sys
import os
import json
import random
import argparse
import time
from typing import Optional

TRAINING_PATH    = os.path.join(os.path.dirname(__file__), "training_pairs.jsonl")
ADAPTER_PATH     = os.path.join(os.path.dirname(__file__), "adapters", "latest")
STEERING_STATE   = os.path.join(os.path.dirname(__file__), "steering_state.json")
NOVELTY_ARCHIVE  = os.path.join(os.path.dirname(__file__), "novelty_archive.jsonl")

# Axiom pairs — Fixed Probe Set training injection (Fix C extension).
# These 25 human-written pairs are ALWAYS included in every training run.
# They NEVER pass through the erosion guard.  They NEVER get filtered.
# They are the gravitational constant of the training loop:
# the model cannot drift away from "17×23=391" if it sees that pair
# in every single gradient update.
AXIOM_PAIRS_PATH = os.path.join(os.path.dirname(__file__), "axiom_pairs.jsonl")

MODEL_ID         = "NousResearch/Hermes-3-Llama-3.1-8B"
MAX_SEQ_LENGTH   = 2048
LORA_RANK        = 32       # increased from 16 to give more capacity for partitioning
LORA_ALPHA       = 64       # scale with rank
BATCH_SIZE       = 2
GRAD_ACCUM       = 4
MAX_STEPS        = 200

# Fix 3 constants
ANCHOR_RATIO              = 0.25   # fraction of non-target pairs kept as anchors
MIN_ANCHOR_PAIRS          = 10     # always keep at least this many anchors
REGRESSION_WARN_THRESHOLD = 0.08   # warn if a solved dim regresses by this much


# ── VRAM management ───────────────────────────────────────────────────────────

def _vram_free_mb() -> int:
    """Return free VRAM in MB, or 0 if torch not available."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free // (1024 * 1024)
    except Exception:
        pass
    return 0


def _unload_ollama_model(model: str = "hermes3:8b") -> bool:
    """
    Ask Ollama to evict its loaded model from VRAM.
    keep_alive=0 requires a prompt field to trigger a valid generate request.
    """
    try:
        import httpx
        # Must include a prompt — Ollama ignores keep_alive on empty requests
        resp = httpx.post(
            __import__("config").OLLAMA_GEN_URL,
            json={"model": model, "prompt": " ", "keep_alive": 0, "stream": False},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _free_vram_for_training(min_free_mb: int = 7000) -> None:
    """
    Unload Ollama's model and wait until at least min_free_mb VRAM is available.
    7000 MB (~7 GB) gives headroom for 8B model in 4-bit + LoRA overhead.
    """
    before = _vram_free_mb()
    print(
        f"[BlackLoRA] VRAM free before unload: {before} MB — "
        f"need {min_free_mb} MB for training",
        flush=True,
    )

    if before >= min_free_mb:
        print("[BlackLoRA] Enough VRAM available. Skipping Ollama unload.", flush=True)
        return

    print("[BlackLoRA] Unloading Ollama model from VRAM...", flush=True)
    _unload_ollama_model()

    # Poll until VRAM is free (up to 20s)
    for i in range(20):
        time.sleep(1)
        free = _vram_free_mb()
        if free >= min_free_mb:
            print(f"[BlackLoRA] VRAM free after unload: {free} MB. Ready.", flush=True)
            return
        if i % 5 == 4:
            print(f"[BlackLoRA] Waiting for VRAM... {free} MB free so far", flush=True)

    free = _vram_free_mb()
    print(
        f"[BlackLoRA] VRAM after 20s: {free} MB. "
        f"{'Proceeding anyway.' if free > 5000 else 'WARNING: may OOM — close Zephyr GUI to free more.'}",
        flush=True,
    )


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    """Check if Phase 2 dependencies are installed."""
    missing = []
    for pkg in ["unsloth", "trl", "transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Phase 2 dependencies not yet installed: {', '.join(missing)}")
        print("\nTo install (run in your terminal):")
        print("  py -3.11 -m pip install torch==2.6.0 torchvision "
              "--index-url https://download.pytorch.org/whl/cu124")
        print("  py -3.11 -m pip install triton-windows==3.2.0.post21 torchao==0.13.0")
        print("  py -3.11 -m pip install unsloth trl transformers datasets accelerate "
              "peft bitsandbytes")
        return False
    return True


# ── Fix 3: Steering state ─────────────────────────────────────────────────────

def _load_steering_state() -> dict:
    """
    Load the steering state written by background_eval after an Oracle trigger.
    Returns empty dict if no state file exists (full training mode).
    """
    if not os.path.exists(STEERING_STATE):
        return {}
    try:
        with open(STEERING_STATE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _select_training_pairs(
    all_pairs: list[dict],
    target_dims: list[str],
    anchor_ratio: float = ANCHOR_RATIO,
) -> list[dict]:
    """
    Fix 3 — Selective pair filtering for erosion protection.

    Splits all_pairs into:
      - targeted:  pairs whose target_dim is in target_dims (all included)
      - anchors:   a random sample of non-target pairs (prevents forgetting)
      - novelty:   pairs from novelty_archive.jsonl (always included fully)

    The anchor fraction keeps gradients from drifting away from solved
    knowledge while concentrating training power on failing dimensions.

    Parameters
    ----------
    all_pairs    : all loaded training pairs
    target_dims  : dimensions currently breaching the Target Set
    anchor_ratio : fraction of non-target pairs to include as anchors

    Returns
    -------
    Filtered list ready for training.
    """
    # Axiom pairs must never reach this function — they're injected in
    # load_training_data() after filtering.  This guard catches any
    # accidental path where they get included in all_pairs.
    non_axiom = [p for p in all_pairs if not p.get("immutable", False)]
    if len(non_axiom) < len(all_pairs):
        print(
            f"[lora] Erosion guard: expelled {len(all_pairs) - len(non_axiom)} "
            "axiom pairs (they are injected separately).",
            flush=True,
        )
    all_pairs = non_axiom

    if not target_dims:
        # No steering state → use everything (full general training)
        print("[lora] No target dims specified — training on full dataset", flush=True)
        return all_pairs

    target_set = set(target_dims)

    targeted = [
        p for p in all_pairs
        if p.get("target_dim") in target_set
        or p.get("source") == "novelty_archive"
    ]
    non_targeted = [
        p for p in all_pairs
        if p.get("target_dim") not in target_set
        and p.get("source") != "novelty_archive"
    ]

    # Anchor sample: random subset of non-targeted pairs
    n_anchors = max(MIN_ANCHOR_PAIRS, int(len(non_targeted) * anchor_ratio))
    anchors = random.sample(non_targeted, min(n_anchors, len(non_targeted)))

    selected = targeted + anchors
    random.shuffle(selected)

    print(
        f"[lora] Erosion guard: {len(targeted)} targeted ({', '.join(sorted(target_set))}) "
        f"+ {len(anchors)} anchors / {len(non_targeted)} non-target available",
        flush=True,
    )
    return selected


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_axiom_pairs() -> list[dict]:
    """
    Load the immutable Fixed Probe Set training pairs (Fix C extension).

    These are ALWAYS returned in full and NEVER subject to any filtering.
    If axiom_pairs.jsonl does not exist, the system prints a warning —
    training proceeds but lacks the Ground Truth Anchor.
    """
    if not os.path.exists(AXIOM_PAIRS_PATH):
        print(
            "[lora] WARNING: axiom_pairs.jsonl not found. "
            "Training will proceed without Ground Truth Anchor pairs. "
            "Run /FixedProbeSet to generate them.",
            flush=True,
        )
        return []

    axioms = []
    with open(AXIOM_PAIRS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("conversations") and len(obj["conversations"]) >= 2:
                    # Tag so erosion guard can identify and skip them
                    obj["source"] = "axiom"
                    obj["immutable"] = True
                    axioms.append(obj)
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"  [data] axiom_pairs.jsonl: {len(axioms)} pairs (immutable, always included)",
          flush=True)
    return axioms


def _load_all_pairs() -> list[dict]:
    """
    Load and merge training pairs from all available sources.

    Sources:
      - blackwell/axiom_pairs.jsonl          Fixed Probe Set (immutable, always included)
      - blackwell/training_pairs.jsonl       Oracle pairs
      - blackwell/coding_training_pairs.jsonl Coding Blackwell
      - trajectory_samples.jsonl             Real conversation turns
      - blackwell/synthetic_pairs.jsonl      data_generator output
      - blackwell/novelty_archive.jsonl      Fix 5 positive anchors

    Axiom pairs are loaded separately and never filtered by _select_training_pairs.
    All other pairs go through the erosion guard.
    """
    base_dir     = os.path.dirname(__file__)
    project_root = os.path.dirname(base_dir)

    sources = [
        TRAINING_PATH,
        os.path.join(base_dir, "coding_training_pairs.jsonl"),
        os.path.join(project_root, "trajectory_samples.jsonl"),
        os.path.join(base_dir, "synthetic_pairs.jsonl"),
        NOVELTY_ARCHIVE,
    ]

    all_records = []
    for src in sources:
        if not os.path.exists(src):
            continue
        count_before = len(all_records)
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("conversations") and len(obj["conversations"]) >= 2:
                        all_records.append(obj)
                except (json.JSONDecodeError, KeyError):
                    pass
        added = len(all_records) - count_before
        if added:
            print(f"  [data] {os.path.basename(src)}: {added} pairs", flush=True)

    return all_records


def load_training_data(target_dims: list[str] = None):
    """
    Load, filter (Fix 3), and format training pairs for SFTTrainer.

    Parameters
    ----------
    target_dims : list of dimension names currently breaching Target Set.
                  If None, loads all pairs (general training run).

    Data pipeline
    -------------
    1. Load axiom pairs (immutable Ground Truth Anchors — NEVER filtered).
    2. Load all other pairs and apply erosion guard (Fix 3).
    3. Merge axiom pairs AFTER filtering so they cannot be dropped.
    4. Shuffle the merged set so axioms are distributed across batches.
    """
    from datasets import Dataset

    # Step 1 — axiom pairs bypass everything
    axiom_objs = _load_axiom_pairs()

    # Step 2 — all other data sources, subject to erosion guard
    all_objs = _load_all_pairs()
    if not all_objs and not axiom_objs:
        raise FileNotFoundError(
            "No training data found. Run: py -3.11 blackwell/data_generator.py"
        )

    # Fix 3 — apply erosion guard to non-axiom pairs only
    filtered_objs = _select_training_pairs(all_objs, target_dims or [])

    # Step 3 — merge axioms back in AFTER filtering
    # Axioms are appended first so they're visible in the log, then the
    # full list is shuffled so they're distributed evenly across batches
    merged = axiom_objs + filtered_objs
    random.shuffle(merged)

    records = []
    for obj in merged:
        convos = obj.get("conversations", [])
        if len(convos) >= 2:
            records.append({
                "human":  convos[0]["value"],
                "zephyr": convos[1]["value"],
                "dim":    obj.get("target_dim", obj.get("category", "unknown")),
            })

    n_axiom   = len(axiom_objs)
    n_other   = len(records) - n_axiom
    print(
        f"Training on {len(records)} pairs total "
        f"({n_axiom} axiomatic + {n_other} dynamic)",
        flush=True,
    )

    def format_row(row):
        return {
            "text": (
                f"<|im_start|>user\n{row['human']}<|im_end|>\n"
                f"<|im_start|>assistant\n{row['zephyr']}<|im_end|>"
            )
        }

    dataset = Dataset.from_list(records)
    return dataset.map(format_row)


# ── Pre/post regret comparison (Fix 3) ───────────────────────────────────────

def _snapshot_regret() -> dict | None:
    """Capture current per-dimension regret before training."""
    try:
        from blackwell.logger import get_average_vector
        from blackwell.evaluator import regret_from_scores
        avg = get_average_vector()
        if avg is None:
            return None
        return regret_from_scores(avg)
    except Exception:
        return None


def _check_regression(pre: dict, post: dict, target_dims: list[str]) -> None:
    """
    Compare pre- and post-training regret.
    Warn if any dimension outside target_dims has regressed by more than
    REGRESSION_WARN_THRESHOLD.
    """
    if pre is None or post is None:
        return
    target_set = set(target_dims)
    for dim, pre_val in pre.items():
        if dim in target_set:
            continue   # expected to change
        post_val = post.get(dim, pre_val)
        regression = post_val - pre_val
        if regression > REGRESSION_WARN_THRESHOLD:
            print(
                f"[lora] ⚠ REGRESSION WARNING: {dim} regressed "
                f"{pre_val:.3f} → {post_val:.3f} (+{regression:.3f}). "
                f"Consider reducing learning_rate or increasing ANCHOR_RATIO.",
                flush=True,
            )
        else:
            arrow = "↓" if regression < -0.01 else "→"
            print(
                f"[lora] {dim}: {pre_val:.3f} {arrow} {post_val:.3f}",
                flush=True,
            )


# ── Main training loop ────────────────────────────────────────────────────────

def _run_probe_gate() -> bool:
    """
    Fix C — Ground Truth Anchor gate.
    Run the fixed probe suite against the current student model.
    Returns True if training should proceed, False if it should be aborted.

    Probe failures are the ONLY signal in the system that is fully
    decoupled from the Oracle-Evaluator feedback loop.  If this gate
    returns False, no amount of good internal metric scores overrides it.
    """
    print("\n[BlackLoRA] ── Probe Gate (Fix C) ──────────────────────────────")
    print("[BlackLoRA] Running fixed probe suite before training...")
    try:
        from blackwell.probe_runner import probe_gate
        ok, reasons = probe_gate()
        if not ok:
            print("\n[BlackLoRA] !! PROBE GATE BLOCKED TRAINING !!", flush=True)
            for r in reasons:
                print(f"  → {r}", flush=True)
            print("[BlackLoRA] Training aborted. Fix the issues above and re-run.",
                  flush=True)
        return ok
    except Exception as e:
        print(f"[BlackLoRA] Probe gate error: {e}", flush=True)
        print("[BlackLoRA] WARNING: Probe gate failed to run — proceeding with "
              "training (manual audit recommended).", flush=True)
        return True   # Fail-open: don't block training on probe runner bugs


def _run_drift_gate() -> bool:
    """
    Fix B — Drift Monitor gate.
    Check if the Oracle-Evaluator gap has grown beyond the drift threshold.
    Returns True if training should proceed, False if drift is detected.

    Drift check is advisory unless sustained (>= MIN_SAMPLES).
    """
    print("[BlackLoRA] Checking Oracle-Evaluator drift state (Fix B)...")
    try:
        from blackwell.drift_monitor import check_drift, print_drift_report
        state = check_drift()
        if state.drift_detected and state.abort_train:
            print_drift_report(state)
            print("[BlackLoRA] !! DRIFT GATE BLOCKED TRAINING !!", flush=True)
            print("[BlackLoRA] The LLM judge is inflating scores above the rule "
                  "layer baseline. This suggests Oracle-Evaluator co-evolution.",
                  flush=True)
            print("[BlackLoRA] Options: audit last 50 exchanges, run "
                  "'py -3.11 blackwell/drift_monitor.py', or reset baseline.",
                  flush=True)
            return False
        elif state.drift_detected:
            print(f"[BlackLoRA] ⚠ Drift detected in {state.drifting_dims} but "
                  f"sample count ({state.n_samples}) below minimum — monitoring "
                  "only, not blocking.", flush=True)
        else:
            print(f"[BlackLoRA] Drift check passed (n={state.n_samples}). "
                  "Oracle-Evaluator gap within bounds.", flush=True)
        return True
    except Exception as e:
        print(f"[BlackLoRA] Drift gate error: {e} — proceeding (advisory only).",
              flush=True)
        return True   # Fail-open: advisory only


def run_lora_steer(steps: int = MAX_STEPS) -> Optional[str]:
    """
    Run one LoRA steering cycle.

    Reads steering_state.json (if present) to focus training on breaching
    dimensions while anchoring solved ones (Fix 3).

    Returns path to GGUF directory on success, None otherwise.
    """
    if not check_dependencies():
        return None

    # ── Fix C: Ground Truth Probe Gate (runs BEFORE touching weights) ──────────
    # The probe suite is the only signal fully decoupled from the Oracle-Evaluator
    # loop.  If it blocks, we do not proceed regardless of internal metric state.
    if not _run_probe_gate():
        return None

    # ── Fix B: Drift Monitor Gate ──────────────────────────────────────────────
    # Check if Oracle-Evaluator gap has grown beyond threshold.
    # Advisory until MIN_SAMPLES is reached; blocks only on sustained drift.
    if not _run_drift_gate():
        return None

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Fix 3 — load steering state
    state       = _load_steering_state()
    target_dims = state.get("target_dims", [])

    print("\n=== Blackwell LoRA Steering Cycle ===")
    print(f"Model      : {MODEL_ID}")
    print(f"Adapter    : {ADAPTER_PATH}")
    print(f"Steps      : {steps}")
    print(f"LoRA rank  : {LORA_RANK}")
    if target_dims:
        print(f"Target dims: {', '.join(target_dims)}  (erosion guard active)")
    else:
        print("Target dims: all  (general training)")
    print()

    # Fix 3 — capture pre-training regret snapshot
    pre_regret = _snapshot_regret()

    # Free Ollama's VRAM before loading training model.
    # Need ~9 GB clear: 4-bit weights (~4.5 GB) + LoRA + optimizer states + buffers.
    _free_vram_for_training(min_free_mb=9000)

    # Load model — device_map={"": 0} forces all layers onto GPU 0.
    # Without this unsloth may spill layers to CPU when VRAM appears tight.
    import torch
    torch.cuda.empty_cache()
    print("Loading model in 4-bit (QLoRA)...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = MODEL_ID,
        max_seq_length= MAX_SEQ_LENGTH,
        dtype         = None,
        load_in_4bit  = True,
        device_map    = {"": 0},
    )

    # torch 2.6 compat: float8_e8m0fnu (MX scaling dtype) added in torch 2.7.
    # unsloth_zoo references it during LoRA attachment; stub with nearest float8.
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0,     # must be 0 for unsloth fast-patch (torch 2.6 compat)
        bias                       = "none",
        use_gradient_checkpointing = True,
        random_state               = 42,
    )

    dataset = load_training_data(target_dims=target_dims)

    os.makedirs(ADAPTER_PATH, exist_ok=True)

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LENGTH,
        args               = TrainingArguments(
            output_dir                  = ADAPTER_PATH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps                = 10,
            max_steps                   = steps,
            learning_rate               = 2e-4,
            fp16                        = False,
            bf16                        = True,   # RTX 3060 Ampere supports bf16
            logging_steps               = 10,
            optim                       = "adamw_8bit",
            save_strategy               = "no",
            report_to                   = "none",
        ),
    )

    print("Training...", flush=True)
    trainer.train()

    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"\nAdapter saved to: {ADAPTER_PATH}", flush=True)

    # Fix 3 — post-training regression check
    if target_dims and pre_regret:
        print("\n[lora] Post-training regret comparison:", flush=True)
        post_regret = _snapshot_regret()
        _check_regression(pre_regret, post_regret, target_dims)

    # Export to GGUF — try three escalating strategies
    gguf_dir   = os.path.join(os.path.dirname(ADAPTER_PATH), "gguf")
    merged_dir = os.path.join(os.path.dirname(ADAPTER_PATH), "merged")

    # Strategy 1: native unsloth GGUF export (fastest, sometimes fails on
    # certain unsloth versions with a 'dict has no attribute replace' error
    # caused by an internal quantization method lookup returning a dict)
    print("[BlackLoRA] Exporting to GGUF (Q4_K_M)...", flush=True)
    try:
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(f"[BlackLoRA] GGUF saved to {gguf_dir}", flush=True)
        return gguf_dir
    except Exception as e1:
        print(f"[BlackLoRA] Strategy 1 failed ({e1}), trying f16 GGUF...", flush=True)

    # Strategy 2: f16 GGUF — avoids the quantization lookup that causes the dict error
    try:
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="f16")
        print(f"[BlackLoRA] f16 GGUF saved to {gguf_dir}", flush=True)
        return gguf_dir
    except Exception as e2:
        print(f"[BlackLoRA] Strategy 2 failed ({e2}), falling back to merged HF save...", flush=True)

    # Strategy 3: save merged HuggingFace weights — Ollama can't load these
    # directly but they can be converted with llama.cpp convert_hf_to_gguf.py
    # The export.py script will detect no .gguf and print the manual steps.
    try:
        os.makedirs(merged_dir, exist_ok=True)
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[BlackLoRA] Merged HF weights saved to {merged_dir}", flush=True)
        print("[BlackLoRA] GGUF conversion requires llama.cpp. Run:", flush=True)
        print(f"  py -3.11 blackwell/export_gguf.py", flush=True)
        # Return merged_dir so the caller knows where weights are,
        # even though register_with_ollama will fail without a .gguf
        return merged_dir
    except Exception as e3:
        print(f"[BlackLoRA] All export strategies failed: {e3}", flush=True)
        print("[BlackLoRA] Adapter is intact. Run  py -3.11 blackwell/export_gguf.py  to retry.", flush=True)
        return None


# Canonical alias
run_lora_cycle = run_lora_steer


# ── Training data report ──────────────────────────────────────────────────────

def check_training_data() -> tuple:
    """Report on training data. Returns (ok: bool, message: str)."""
    records_raw = _load_all_pairs()
    if not records_raw:
        msg = "No training data yet. Run: py -3.11 run_oracle.py"
        print(msg)
        return False, msg

    dim_counts: dict[str, int] = {}
    for obj in records_raw:
        dim = obj.get("target_dim", "unknown")
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    print("\n=== Training Data Report ===")
    print(f"Total pairs : {len(records_raw)}")
    print("By dimension:")
    for dim, count in sorted(dim_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 60)
        print(f"  {dim:<18} {count:>4}  {bar}")

    lengths = []
    for r in records_raw:
        try:
            convs = r.get("conversations", [])
            if len(convs) >= 2:
                lengths.append(len(convs[1]["value"].split()))
        except (KeyError, TypeError, AttributeError):
            continue
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"Avg Zephyr response length: {avg_len:.0f} words")

    state = _load_steering_state()
    if state.get("target_dims"):
        print(f"\nCurrent steering target: {state['target_dims']}")
        print(f"  Gaps: {state.get('gaps', {})}")

    needed = 200 - len(records_raw)
    if needed > 0:
        msg = f"Only {len(records_raw)} pairs — need {needed} more before LoRA training."
        print(f"\n{msg}")
        return False, msg
    else:
        msg = f"{len(records_raw)} pairs — ready for LoRA training."
        print(f"\n{msg}")
        return True, msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackwell LoRA Steering")
    parser.add_argument("--check", action="store_true", help="Check data and deps")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if args.check:
        check_dependencies()
        check_training_data()
    else:
        run_lora_steer(steps=args.steps)

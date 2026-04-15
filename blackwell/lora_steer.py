"""
blackwell/lora_steer.py
Step 4 of the Blackwell training loop — the gradient nudge.

Runs QLoRA fine-tuning on training_pairs.jsonl using unsloth
on the RTX 3060. Each run is a "strategy update" that steers
Zephyr's weights toward the Target Set S.

Requirements (install when ready for Phase 2):
    pip install unsloth trl transformers datasets

Usage:
    python blackwell/lora_steer.py
    python blackwell/lora_steer.py --check    # just verify setup
"""

import sys
import os
import json
import argparse
from typing import Optional

TRAINING_PATH  = os.path.join(os.path.dirname(__file__), "training_pairs.jsonl")
ADAPTER_PATH   = os.path.join(os.path.dirname(__file__), "adapters", "latest")
MODEL_ID       = "NousResearch/Hermes-3-Llama-3.1-8B"  # base model
MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16      # r — higher = more capacity, more VRAM
LORA_ALPHA     = 32      # scaling factor
BATCH_SIZE     = 2       # RTX 3060 safe batch size
GRAD_ACCUM     = 4       # effective batch = 8
MAX_STEPS      = 200     # one steering cycle — short by design


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
        print("  pip install unsloth trl transformers datasets")
        print("\nNote: unsloth requires torch with CUDA already installed.")
        print("Your RTX 3060 (12.9GB) is sufficient for QLoRA on 8B models.\n")
        return False
    return True


def _load_all_pairs() -> list:
    """Load and merge training pairs from all available sources.

    Sources (all optional except training_pairs.jsonl):
      - blackwell/training_pairs.jsonl      (Blackwell oracle pairs)
      - blackwell/coding_training_pairs.jsonl (coding-specific pairs)
      - trajectory_samples.jsonl             (real conversation trajectory)
      - blackwell/synthetic_pairs.jsonl      (synthetic failure-mode pairs)
    """
    base_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(base_dir)

    sources = [
        TRAINING_PATH,  # blackwell/training_pairs.jsonl (existing constant)
        os.path.join(base_dir, "coding_training_pairs.jsonl"),
        os.path.join(project_root, "trajectory_samples.jsonl"),
        os.path.join(base_dir, "synthetic_pairs.jsonl"),
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


def load_training_data():
    """Load and format training pairs for SFTTrainer."""
    from datasets import Dataset

    all_objs = _load_all_pairs()
    if not all_objs:
        raise FileNotFoundError(
            f"No training data found. Run: python blackwell/data_generator.py"
        )

    records = []
    for obj in all_objs:
        convos = obj.get("conversations", [])
        if len(convos) >= 2:
            records.append({
                "human":  convos[0]["value"],
                "zephyr": convos[1]["value"],
                "dim":    obj.get("target_dim", "unknown"),
            })

    print(f"Loaded {len(records)} training pairs total", flush=True)

    # Format as instruction-following
    def format_row(row):
        return {
            "text": (
                f"<|im_start|>user\n{row['human']}<|im_end|>\n"
                f"<|im_start|>assistant\n{row['zephyr']}<|im_end|>"
            )
        }

    dataset = Dataset.from_list(records)
    return dataset.map(format_row)


def run_lora_steer(steps: int = MAX_STEPS) -> Optional[str]:
    """Run one LoRA steering cycle.

    Returns
    -------
    str | None
        Path to the GGUF directory on success, None if export was skipped or
        training could not start.
    """
    if not check_dependencies():
        return None

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print("\n=== Blackwell LoRA Steering Cycle ===")
    print(f"Model      : {MODEL_ID}")
    print(f"Adapter    : {ADAPTER_PATH}")
    print(f"Steps      : {steps}")
    print(f"LoRA rank  : {LORA_RANK}")
    print()

    # Load base model with 4-bit quantisation (QLoRA)
    print("Loading model in 4-bit (QLoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = MODEL_ID,
        max_seq_length= MAX_SEQ_LENGTH,
        dtype         = None,   # auto-detect
        load_in_4bit  = True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r                 = LORA_RANK,
        target_modules    = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        lora_alpha        = LORA_ALPHA,
        lora_dropout      = 0.05,
        bias              = "none",
        use_gradient_checkpointing = True,
        random_state      = 42,
    )

    dataset = load_training_data()

    os.makedirs(ADAPTER_PATH, exist_ok=True)

    trainer = SFTTrainer(
        model            = model,
        tokenizer        = tokenizer,
        train_dataset    = dataset,
        dataset_text_field = "text",
        max_seq_length   = MAX_SEQ_LENGTH,
        args             = TrainingArguments(
            output_dir              = ADAPTER_PATH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps            = 10,
            max_steps               = steps,
            learning_rate           = 2e-4,
            fp16                    = True,
            logging_steps           = 10,
            optim                   = "adamw_8bit",
            save_strategy           = "no",
            report_to               = "none",
        ),
    )

    print("Training...")
    trainer.train()

    # Save adapter
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"\nAdapter saved to: {ADAPTER_PATH}")

    # Export merged model to GGUF for Ollama
    print("[BlackLoRA] Exporting to GGUF (Q4_K_M) — this takes a few minutes...", flush=True)
    gguf_dir = os.path.join(os.path.dirname(ADAPTER_PATH), "gguf")
    try:
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"[BlackLoRA] GGUF saved to {gguf_dir}", flush=True)
        return gguf_dir
    except Exception as e:
        print(f"[BlackLoRA] GGUF export failed: {e}", flush=True)
        print("[BlackLoRA] Adapter saved but GGUF skipped. Run export manually.", flush=True)
        return None


# Canonical alias used by agent.py and export pipeline
run_lora_cycle = run_lora_steer


def check_training_data() -> tuple:
    """Report on what's in the training data.

    Returns
    -------
    (ok: bool, message: str)
        ok is True when there are enough pairs to train (>=200).
    """
    records_raw = _load_all_pairs()
    if not records_raw:
        msg = f"No training data yet. Run: python run_oracle.py"
        print(msg)
        return False, msg

    records = records_raw
    dim_counts = {}
    for obj in records:
        dim = obj.get("target_dim", "unknown")
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    print(f"\n=== Training Data Report ===")
    print(f"Total pairs : {len(records)}")
    print(f"By dimension:")
    for dim, count in sorted(dim_counts.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"  {dim:<14} {count:>4}  {bar}")

    lengths = []
    for r in records:
        try:
            convs = r.get("conversations", [])
            if len(convs) >= 2:
                lengths.append(len(convs[1]["value"].split()))
        except (KeyError, TypeError, AttributeError):
            continue
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"Avg Zephyr response length: {avg_len} words")

    rec_needed = 200 - len(records)
    if rec_needed > 0:
        msg = f"Only {len(records)} pairs — need {rec_needed} more before LoRA training."
        print(f"\n{msg}")
        print("Run run_oracle.py a few more times to build the dataset.")
        print()
        return False, msg
    else:
        msg = f"{len(records)} pairs — ready for LoRA training."
        print(f"\n{msg}")
        print("Run: python blackwell/lora_steer.py")
        print()
        return True, msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackwell LoRA Steering")
    parser.add_argument("--check",   action="store_true", help="Check training data and dependencies")
    parser.add_argument("--steps",   type=int, default=MAX_STEPS, help="Training steps")
    args = parser.parse_args()

    if args.check:
        check_dependencies()
        check_training_data()
    else:
        run_lora_steer(steps=args.steps)

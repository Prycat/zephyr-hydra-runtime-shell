# -*- coding: utf-8 -*-
"""
blackwell/rdsp_heal.py
Post-prune LoRA healing for RDSP.

After a soft prune is validated and committed, the remaining attention heads
need to redistribute the capacity lost by the pruned heads. This module
runs a shortened LoRA fine-tune (max_steps=50 by default) using the existing
lora_steer.py data pipeline.

A short heal is intentional:
  - Too many steps → model adapts to pruned architecture and locks in losses
  - Too few steps  → remaining heads don't compensate
  - 50 steps ≈ 25% of a full training run, enough to redistribute
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class HealConfig:
    """
    Parameters for the post-prune LoRA healing run.

    Attributes
    ----------
    max_steps      : number of training steps (default 50, lighter than full 200)
    lora_rank      : LoRA rank (default 16, lighter than full 32)
    batch_size     : per-device batch size
    grad_accum     : gradient accumulation steps
    learning_rate  : optimizer learning rate
    model_id       : HuggingFace base model ID
    max_seq_length : tokenizer truncation length
    lora_alpha     : computed as lora_rank * 2 (property)
    """
    max_steps:      int   = 50
    lora_rank:      int   = 16
    batch_size:     int   = 2
    grad_accum:     int   = 4
    learning_rate:  float = 1e-4
    model_id:       str   = "NousResearch/Hermes-3-Llama-3.1-8B"
    max_seq_length: int   = 2048

    @property
    def lora_alpha(self) -> int:
        """LoRA alpha is always 2× lora_rank."""
        return self.lora_rank * 2


def build_heal_args(cfg: HealConfig) -> dict:
    """
    Convert HealConfig to a flat dict for logging or SFTTrainer kwargs.

    Parameters
    ----------
    cfg : HealConfig instance

    Returns
    -------
    dict with keys: max_steps, lora_rank, lora_alpha, batch_size,
                    grad_accum, learning_rate
    """
    return {
        "max_steps":     cfg.max_steps,
        "lora_rank":     cfg.lora_rank,
        "lora_alpha":    cfg.lora_alpha,
        "batch_size":    cfg.batch_size,
        "grad_accum":    cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
    }


def run_heal(cfg: HealConfig | None = None) -> bool:
    """
    Run a shortened LoRA fine-tune on the current adapter to heal post-prune.

    Reuses lora_steer.py's data pipeline. Requires CUDA + unsloth.
    Not unit tested (integration only).

    Parameters
    ----------
    cfg : HealConfig — uses defaults if None

    Returns
    -------
    True on success, False on failure.
    """
    if cfg is None:
        cfg = HealConfig()

    print(
        f"[rdsp_heal] Starting heal: {cfg.max_steps} steps, "
        f"LoRA rank {cfg.lora_rank}, lr {cfg.learning_rate:.0e}",
        flush=True,
    )

    try:
        import torch
        if not hasattr(torch, "float8_e8m0fnu"):
            torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig

        adapter_path = os.path.join(_HERE, "adapters", "latest")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = adapter_path,
            max_seq_length = cfg.max_seq_length,
            dtype          = None,
            load_in_4bit   = True,
            device_map     = {"": 0},
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r               = cfg.lora_rank,
            target_modules  = ["q_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            lora_alpha      = cfg.lora_alpha,
            lora_dropout    = 0.0,
            bias            = "none",
            use_gradient_checkpointing = "unsloth",
        )

        sys.path.insert(0, os.path.dirname(_HERE))
        from blackwell.lora_steer import load_training_data
        dataset = load_training_data(target_dims=None)

        trainer = SFTTrainer(
            model         = model,
            tokenizer     = tokenizer,
            train_dataset = dataset,
            args = SFTConfig(
                output_dir                  = adapter_path,
                max_steps                   = cfg.max_steps,
                per_device_train_batch_size = cfg.batch_size,
                gradient_accumulation_steps = cfg.grad_accum,
                learning_rate               = cfg.learning_rate,
                fp16                        = not torch.cuda.is_bf16_supported(),
                bf16                        = torch.cuda.is_bf16_supported(),
                logging_steps               = 10,
                save_steps                  = cfg.max_steps,
                warmup_steps                = 5,
                dataset_text_field          = "text",
                max_seq_length              = cfg.max_seq_length,
                report_to                   = "none",
            ),
        )

        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print("[rdsp_heal] Heal complete. Adapter saved.", flush=True)
        return True

    except Exception as e:
        print(f"[rdsp_heal] Heal failed: {e}", flush=True)
        return False

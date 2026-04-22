"""Tests for rdsp_heal.py — post-prune LoRA recovery."""
import pytest


from blackwell.rdsp_heal import HealConfig, build_heal_args


def test_heal_config_defaults():
    cfg = HealConfig()
    assert cfg.max_steps == 50
    assert cfg.lora_rank == 16
    assert cfg.batch_size == 2
    assert cfg.learning_rate == pytest.approx(1e-4, rel=1e-3)


def test_heal_config_override():
    cfg = HealConfig(max_steps=100, lora_rank=32)
    assert cfg.max_steps == 100
    assert cfg.lora_rank == 32


def test_build_heal_args_keys():
    cfg  = HealConfig()
    args = build_heal_args(cfg)
    for key in ("max_steps", "lora_rank", "lora_alpha", "batch_size",
                "grad_accum", "learning_rate"):
        assert key in args, f"missing key: {key}"


def test_build_heal_args_values():
    cfg  = HealConfig(max_steps=75, lora_rank=24)
    args = build_heal_args(cfg)
    assert args["max_steps"] == 75
    assert args["lora_rank"] == 24
    assert args["lora_alpha"] == 48   # always 2× rank


def test_heal_config_lora_alpha_property():
    """lora_alpha is always 2x lora_rank."""
    cfg = HealConfig(lora_rank=8)
    assert cfg.lora_alpha == 16
    cfg2 = HealConfig(lora_rank=32)
    assert cfg2.lora_alpha == 64

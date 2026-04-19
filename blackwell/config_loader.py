# blackwell/config_loader.py
"""
blackwell/config_loader.py — Central config loader for thinking_config.yaml.

All runtime-tunable parameters live in thinking_config.yaml.
Call load_thinking_config() at the top of each function that needs them
(not at import time) so a GUI SAVE takes effect on the next call.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "thinking_config.yaml")

_DEFAULTS = {
    "inference": {
        "model_temperature": 0.7,
        "oracle_temperature": 0.3,
        "max_tokens": 512,
    },
    "judge": {
        "temperature": 0.0,
        "safety_floor": 0.95,
        "accuracy_floor": 0.95,
    },
    "approachability": {
        "s_bound_low": -0.10,
        "s_bound_high": 0.10,
        "regret_threshold": 0.15,
        "convergence_window": 20,
    },
    "training": {
        "abort_logic_ratio": 0.50,
        "abort_overall_floor": 0.60,
        "min_pairs": 200,
    },
}


@dataclass(frozen=True)
class ThinkingConfig:
    # inference
    model_temperature:  float
    oracle_temperature: float
    max_tokens:         int
    # judge
    judge_temperature:  float
    safety_floor:       float
    accuracy_floor:     float
    # approachability
    s_bound_low:        float
    s_bound_high:       float
    regret_threshold:   float
    convergence_window: int
    # training
    abort_logic_ratio:   float
    abort_overall_floor: float
    min_pairs:           int


def load_thinking_config(path: str | None = None) -> ThinkingConfig:
    """
    Load thinking_config.yaml, merge with hardcoded defaults.
    Missing keys, bad YAML, or type-incompatible values all fall through
    to defaults — never raises.
    """
    raw: dict = {}
    try:
        import yaml
        with open(path or _DEFAULT_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        pass  # missing file, bad YAML — fall through to defaults

    def _get(section: str, key: str):
        return (raw.get(section) or {}).get(key, _DEFAULTS[section][key])

    try:
        return ThinkingConfig(
            model_temperature  = float(_get("inference", "model_temperature")),
            oracle_temperature = float(_get("inference", "oracle_temperature")),
            max_tokens         = int(_get("inference",   "max_tokens")),
            judge_temperature  = float(_get("judge",     "temperature")),
            safety_floor       = float(_get("judge",     "safety_floor")),
            accuracy_floor     = float(_get("judge",     "accuracy_floor")),
            s_bound_low        = float(_get("approachability", "s_bound_low")),
            s_bound_high       = float(_get("approachability", "s_bound_high")),
            regret_threshold   = float(_get("approachability", "regret_threshold")),
            convergence_window = int(_get("approachability",   "convergence_window")),
            abort_logic_ratio   = float(_get("training", "abort_logic_ratio")),
            abort_overall_floor = float(_get("training", "abort_overall_floor")),
            min_pairs           = int(_get("training",   "min_pairs")),
        )
    except Exception:
        # Bad values in YAML (e.g. model_temperature: "high") — return pure defaults
        return ThinkingConfig(
            model_temperature  = 0.7,
            oracle_temperature = 0.3,
            max_tokens         = 512,
            judge_temperature  = 0.0,
            safety_floor       = 0.95,
            accuracy_floor     = 0.95,
            s_bound_low        = -0.10,
            s_bound_high       = 0.10,
            regret_threshold   = 0.15,
            convergence_window = 20,
            abort_logic_ratio   = 0.50,
            abort_overall_floor = 0.60,
            min_pairs           = 200,
        )

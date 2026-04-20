"""
blackwell/export_gguf.py
Manual GGUF export from a saved adapter — run this when /run_lora
completes training but the GGUF step fails.

Usage:
    py -3.11 blackwell/export_gguf.py

What it does:
    1. Loads the base model (NousResearch/Hermes-3-Llama-3.1-8B) in 4-bit
    2. Merges the LoRA adapter at blackwell/adapters/latest
    3. Tries to export as Q4_K_M GGUF
    4. Falls back to f16 GGUF if the quantisation lookup fails
    5. Runs `ollama create prycat` so the model is immediately usable

The adapter must already exist (training must have completed).
Re-running this after a failed /run_lora export is safe — it does NOT retrain.
"""
from __future__ import annotations

import os
import sys

_HERE        = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(_HERE, "adapters", "latest")
GGUF_DIR     = os.path.join(_HERE, "adapters", "gguf")
MODEL_ID     = "NousResearch/Hermes-3-Llama-3.1-8B"
MAX_SEQ_LEN  = 2048


def _check_adapter():
    if not os.path.isdir(ADAPTER_PATH):
        print(f"[export_gguf] ERROR: Adapter not found at {ADAPTER_PATH}")
        print("  Run /run_lora first to complete training.")
        sys.exit(1)
    # adapter_config.json is written by PEFT on save
    cfg = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if not os.path.exists(cfg):
        print(f"[export_gguf] ERROR: adapter_config.json missing in {ADAPTER_PATH}")
        print("  The adapter directory exists but may be incomplete.")
        sys.exit(1)
    print(f"[export_gguf] Adapter found: {ADAPTER_PATH}")


def _load_model_and_adapter():
    print("[export_gguf] Loading base model + adapter (this takes ~2 min)...", flush=True)
    import torch
    torch.cuda.empty_cache()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[export_gguf] ERROR: unsloth not installed.")
        print("  pip install unsloth")
        sys.exit(1)

    # Compat shim for torch 2.6 vs unsloth expecting torch 2.7 dtype
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = ADAPTER_PATH,   # load base + adapter together
        max_seq_length= MAX_SEQ_LEN,
        dtype         = None,
        load_in_4bit  = True,
        device_map    = {"": 0},
    )
    return model, tokenizer


MERGED_DIR = os.path.join(os.path.dirname(ADAPTER_PATH), "merged")
MODELFILE_PATH = os.path.join(_HERE, "Modelfile")


def _export(model, tokenizer) -> str | None:
    os.makedirs(GGUF_DIR, exist_ok=True)

    # Strategy A — unsloth native GGUF (fails on unsloth 2026.x + transformers 5.5
    # with 'dict has no attribute replace'; kept so future unsloth fixes pick it up)
    print("[export_gguf] Attempting unsloth Q4_K_M export...", flush=True)
    try:
        model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
        print(f"[export_gguf] Q4_K_M GGUF saved to {GGUF_DIR}", flush=True)
        return GGUF_DIR
    except Exception as e:
        print(f"[export_gguf] unsloth GGUF failed ({e}) — trying merged HF path...", flush=True)

    # Strategy B — save merged HuggingFace safetensors, let Ollama convert
    # Ollama's FROM directive accepts a local HF directory and converts to GGUF itself.
    print("[export_gguf] Saving merged 16-bit weights for Ollama to convert...", flush=True)
    try:
        os.makedirs(MERGED_DIR, exist_ok=True)
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        print(f"[export_gguf] Merged weights saved to {MERGED_DIR}", flush=True)
        return MERGED_DIR   # signal success; register_with_ollama handles the rest
    except Exception as e:
        print(f"[export_gguf] Merged save failed: {e}", flush=True)

    print("[export_gguf] All export strategies failed.", flush=True)
    return None


def _patch_chat_template(model_dir: str) -> None:
    """
    Ollama's Go parser expects chat_template to be either a plain string or an
    array of {name, template} objects.  Newer transformers saves it as a dict
    ({name: template, ...}), which causes:
        'invalid chat_template: json: cannot unmarshal object into Go value …'

    This patches tokenizer_config.json in-place (backing up the original) so
    ollama create can proceed.  Safe to call multiple times — checks format first.
    """
    import json as _json
    import shutil as _shutil
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tc_path):
        return
    try:
        with open(tc_path, encoding="utf-8") as f:
            tc = _json.load(f)
        ct = tc.get("chat_template")
        if not isinstance(ct, dict):
            return   # already a string or array — nothing to do
        _shutil.copy(tc_path, tc_path + ".bak")
        tc["chat_template"] = [{"name": k, "template": v} for k, v in ct.items()]
        with open(tc_path, "w", encoding="utf-8") as f:
            _json.dump(tc, f, ensure_ascii=False, indent=2)
        print(f"[export_gguf] Patched chat_template dict→array "
              f"({list(ct.keys())}) in tokenizer_config.json", flush=True)
    except Exception as e:
        print(f"[export_gguf] Warning: could not patch chat_template: {e}", flush=True)


def _register(output_dir: str):
    """
    Register with Ollama.  If output_dir contains a .gguf, use it directly.
    If it contains safetensors (merged HF model), write a Modelfile pointing
    at the directory — Ollama converts to GGUF internally on `ollama create`.
    """
    import glob
    import subprocess

    MODEL_NAME = "prycat"

    # Check whether we have a proper GGUF file or just merged weights
    gguf_files = glob.glob(os.path.join(output_dir, "*.gguf"))
    safetensor_files = glob.glob(os.path.join(output_dir, "*.safetensors"))

    if gguf_files:
        from_line = f"FROM {os.path.abspath(gguf_files[0])}"
        print(f"[export_gguf] Using GGUF file: {gguf_files[0]}", flush=True)
    elif safetensor_files:
        # Ollama accepts a directory path — it converts internally
        from_line = f"FROM {os.path.abspath(output_dir)}"
        print(f"[export_gguf] Using merged HF directory (Ollama will convert)...", flush=True)
        print(f"[export_gguf] Note: first run of ollama create may take 15-20 min to quantize.", flush=True)
        # Patch chat_template format before Ollama tries to parse it
        _patch_chat_template(output_dir)
    else:
        print(f"[export_gguf] ERROR: No .gguf or .safetensors found in {output_dir}", flush=True)
        return False

    # Write Modelfile
    try:
        with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
            f.write(from_line + "\n")
        print(f"[export_gguf] Modelfile written: {MODELFILE_PATH}", flush=True)
    except OSError as e:
        print(f"[export_gguf] ERROR writing Modelfile: {e}", flush=True)
        return False

    # Run ollama create
    print(f"[export_gguf] Running: ollama create {MODEL_NAME} ...", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH],
            encoding="utf-8", errors="replace", capture_output=True,
            timeout=1800,   # 30 min — quantising an 8B from safetensors takes ~15-20 min
        )
        if result.stdout:
            for line in result.stdout.splitlines():
                print(f"[ollama] {line}", flush=True)
        if result.returncode == 0:
            print(f"[export_gguf] '{MODEL_NAME}' registered in Ollama.", flush=True)
            try:
                from blackwell.export import _record_registered
                _record_registered(MODEL_NAME)
            except Exception:
                pass
            return True
        else:
            if result.stderr:
                for line in result.stderr.splitlines():
                    print(f"[ollama stderr] {line}", flush=True)
            print(f"[export_gguf] ollama create failed (exit {result.returncode})", flush=True)
            return False
    except FileNotFoundError:
        print("[export_gguf] ERROR: 'ollama' not found in PATH", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[export_gguf] ERROR: ollama create timed out (10 min)", flush=True)
        return False
    except Exception as e:
        print(f"[export_gguf] ERROR: {e}", flush=True)
        return False


def _find_existing_output() -> str | None:
    """
    Return the directory to register if weights already exist on disk,
    so we can skip the expensive model-load + merge step on re-runs.

    Priority:
      1. GGUF_DIR — has .gguf files (unsloth Q4_K_M export succeeded previously)
      2. MERGED_DIR — has .safetensors files (merged HF weights from training run)
    """
    import glob
    if os.path.isdir(GGUF_DIR) and glob.glob(os.path.join(GGUF_DIR, "*.gguf")):
        print(f"[export_gguf] Found existing GGUF weights at {GGUF_DIR} — skipping model load.",
              flush=True)
        return GGUF_DIR
    if os.path.isdir(MERGED_DIR) and glob.glob(os.path.join(MERGED_DIR, "*.safetensors")):
        print(f"[export_gguf] Found existing merged weights at {MERGED_DIR} — skipping model load.",
              flush=True)
        return MERGED_DIR
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("  BLACKWELL GGUF EXPORT")
    print("  Converts saved adapter → GGUF → registers with Ollama")
    print("=" * 60)
    print()

    _check_adapter()

    # Fast-path: if a previous run already produced weights, skip the
    # 10-minute model load + merge and jump straight to Ollama registration.
    output_dir = _find_existing_output()
    if output_dir is None:
        model, tokenizer = _load_model_and_adapter()
        output_dir = _export(model, tokenizer)

    if output_dir:
        print()
        ok = _register(output_dir)
        print()
        if ok:
            print("[export_gguf] Done. Switch to 'prycat' in the model card.")
        else:
            print("[export_gguf] Weights saved but Ollama registration failed.")
            print(f"  Merged model: {output_dir}")
            print(f"  Modelfile:    {MODELFILE_PATH}")
            print("  Run manually:  ollama create prycat -f blackwell\\Modelfile")
    else:
        print()
        print("[export_gguf] Export failed. Your adapter is safe at:")
        print(f"  {ADAPTER_PATH}")
        print("  Rerun this script after checking unsloth version.")
        sys.exit(1)

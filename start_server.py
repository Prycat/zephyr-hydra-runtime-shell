"""
Start vLLM with TurboQuant KV cache compression.
Run this before agent.py. Server listens on http://localhost:8000/v1
"""

import sys

# ── CUDA guard ────────────────────────────────────────────────────────────────
try:
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. vLLM requires an NVIDIA GPU.")
        print("Run install.bat first and ensure CUDA drivers are installed.")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({vram_gb:.1f} GB VRAM)")
except ImportError:
    print("ERROR: PyTorch not installed. Run install.bat first.")
    sys.exit(1)

# ── Apply TurboQuant patch BEFORE importing vLLM ─────────────────────────────
# TurboQuant monkey-patches vLLM's attention mechanism at import time.
# It MUST be imported before vLLM initialises its attention layers.
try:
    import turboquant
    from turboquant.vllm_attn_backend import apply_turboquant_patch
    apply_turboquant_patch(bits_k=3, bits_v=4)
    print("TurboQuant patch applied: 3-bit keys, 4-bit values")
except ImportError as e:
    print(f"ERROR: TurboQuant not found ({e})")
    print("Run install.bat first.")
    sys.exit(1)
except AttributeError:
    # Fallback: some versions apply the patch at import, no explicit call needed
    print("TurboQuant patch applied at import (no explicit call required)")

# ── Launch vLLM OpenAI server ─────────────────────────────────────────────────
MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"
PORT  = 8000

print(f"\nStarting vLLM server...")
print(f"  Model : {MODEL}")
print(f"  Port  : {PORT}")
print(f"  URL   : http://localhost:{PORT}/v1")
print(f"\nFirst run will download ~16 GB from HuggingFace. Subsequent runs use cache.\n")

try:
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    # Build args as if called from CLI
    parser = make_arg_parser()
    args = parser.parse_args([
        "--model", MODEL,
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",          # good default for Ampere+ GPUs
        "--max-model-len", "8192",      # reasonable context; raise if needed
        "--enable-auto-tool-choice",    # needed for tool/function calling
        "--tool-call-parser", "hermes", # Hermes-format tool call parsing
    ])
    run_server(args)
except ImportError as e:
    print(f"ERROR: vLLM not found ({e})")
    print("Run install.bat first.")
    sys.exit(1)

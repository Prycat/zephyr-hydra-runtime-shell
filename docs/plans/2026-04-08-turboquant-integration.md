# TurboQuant Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Ollama with vLLM 0.18.0 as the inference backend and apply TurboQuant KV cache compression (3-bit keys / 4-bit values) for faster, more memory-efficient Hermes 3 inference.

**Architecture:** `agent.py` currently sends requests to Ollama on port 11434. After this change it will send the same OpenAI-format requests to a vLLM server on port 8000. TurboQuant monkey-patches vLLM's attention layer at import time before the server starts, transparently compressing the KV cache during every inference call.

**Tech Stack:** Python 3.9, vLLM 0.18.0, TurboQuant (0xSero/turboquant), Triton, CUDA, HuggingFace (`NousResearch/Hermes-3-Llama-3.1-8B`), openai Python SDK

---

### Task 1: Create install.bat

**Files:**
- Create: `C:/Users/gamer23/Desktop/hermes-agent/install.bat`

This script clones TurboQuant and installs vLLM 0.18.0 + TurboQuant. Run once.

**Step 1: Create install.bat**

```bat
@echo off
echo === TurboQuant + vLLM Setup ===

REM Check Python
python --version || (echo ERROR: Python not found && exit /b 1)

REM Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if errorlevel 1 (
    echo ERROR: CUDA not available. vLLM requires an NVIDIA GPU with CUDA.
    echo Tip: Install PyTorch with CUDA first: https://pytorch.org/get-started/locally/
    exit /b 1
)

REM Install vLLM 0.18.0 (TurboQuant targets this exact version)
echo Installing vLLM 0.18.0...
pip install vllm==0.18.0 || (echo ERROR: vLLM install failed && exit /b 1)

REM Clone TurboQuant if not already present
if not exist turboquant (
    echo Cloning TurboQuant...
    git clone https://github.com/0xSero/turboquant turboquant || (echo ERROR: git clone failed && exit /b 1)
) else (
    echo TurboQuant already cloned, skipping.
)

REM Install TurboQuant
echo Installing TurboQuant...
cd turboquant
pip install -e . || (echo ERROR: TurboQuant install failed && exit /b 1)
cd ..

echo.
echo === Setup complete! ===
echo Next: python start_server.py
```

**Step 2: Verify the file looks right**

Open `install.bat` and confirm the CUDA check, vLLM pin to 0.18.0, and the git clone target match the plan.

**Step 3: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git init  # if not already a git repo
git add install.bat
git commit -m "feat: add TurboQuant+vLLM install script"
```

---

### Task 2: Create start_server.py

**Files:**
- Create: `C:/Users/gamer23/Desktop/hermes-agent/start_server.py`

This script applies TurboQuant's monkey-patch to vLLM's attention layer **before** the server loads any model, then starts vLLM's OpenAI-compatible API on port 8000.

**Step 1: Write the script**

```python
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
```

**Step 2: Verify the two critical ordering points**

- TurboQuant import/patch happens **before** any `from vllm import ...`
- `--enable-auto-tool-choice` and `--tool-call-parser hermes` are present (required for tool calls in `agent.py` to work)

**Step 3: Commit**

```bash
git add start_server.py
git commit -m "feat: add vLLM+TurboQuant server launcher"
```

---

### Task 3: Update agent.py

**Files:**
- Modify: `C:/Users/gamer23/Desktop/hermes-agent/agent.py` (line 12-15)

One change: `base_url` port 11434 → 8000. Everything else stays identical.

**Step 1: Make the change**

Find this block in `agent.py`:
```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the client but not used by Ollama
)
```

Replace with:
```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",  # vLLM does not require an API key by default
)
```

Also update the model name constant on line 17:
```python
# Before:
MODEL = "hermes3"

# After:
MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"
```

And add a startup connectivity check in `main()`, just before the chat loop:
```python
def main():
    print("Hermes 3 Agent (local via vLLM + TurboQuant)")
    print("Type 'exit' or 'quit' to stop, 'clear' to reset history.\n")

    # Verify server is reachable
    import httpx
    try:
        httpx.get("http://localhost:8000/health", timeout=3)
    except Exception:
        print("ERROR: vLLM server not reachable on port 8000.")
        print("Start it first with:  python start_server.py\n")
        return

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    # ... rest of loop unchanged
```

**Step 2: Update SYSTEM_PROMPT to reflect TurboQuant (optional but nice)**

```python
SYSTEM_PROMPT = """You are Hermes, a helpful and capable AI assistant running locally via vLLM with TurboQuant KV cache compression.
You have access to tools: calculate, get_current_time, read_file, write_file.
Use tools whenever they would give a more accurate or useful answer.
Be concise and direct."""
```

**Step 3: Install httpx if not present**

```bash
pip install httpx
```

**Step 4: Commit**

```bash
git add agent.py
git commit -m "feat: point agent at vLLM+TurboQuant backend (port 8000)"
```

---

### Task 4: Smoke Test

**Goal:** Verify the full stack works end-to-end.

**Step 1: Run install.bat (first time only)**

```bat
install.bat
```

Expected output:
```
GPU: NVIDIA ... (N.N GB VRAM)
Installing vLLM 0.18.0...
Cloning TurboQuant...
Installing TurboQuant...
=== Setup complete! ===
```

**Step 2: Start the server**

```bash
python start_server.py
```

Expected (after model download):
```
TurboQuant patch applied: 3-bit keys, 4-bit values
Starting vLLM server...
  Model : NousResearch/Hermes-3-Llama-3.1-8B
  Port  : 8000
INFO:     Application startup complete.
```

**Step 3: Run the non-interactive test**

In a second terminal:
```bash
cd C:/Users/gamer23/Desktop/hermes-agent
python -c "
from agent import run_agent, SYSTEM_PROMPT
history = [{'role': 'system', 'content': SYSTEM_PROMPT}]
reply, _ = run_agent('What is sqrt(144) + 10?', history)
print('Reply:', reply)
"
```

Expected:
```
  [tool] calculate({'expression': 'sqrt(144) + 10'})
Reply: The result of sqrt(144) + 10 is 22.0.
```

**Step 4: Verify TurboQuant is active**

Check GPU VRAM usage is lower than with Ollama for the same model. On an 8B model with long context you should see meaningfully less VRAM consumed by the KV cache.

```bash
python -c "import torch; print(f'VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB')"
```

**Step 5: Run the interactive agent**

```bash
python agent.py
```

Test a multi-tool query: *"What time is it, and what is 2^20?"*

Expected: agent calls `get_current_time` and `calculate` in one turn and returns both results.

**Step 6: Final commit**

```bash
git add .
git commit -m "feat: TurboQuant+vLLM integration complete and smoke tested"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA not available` | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `vLLM install fails` | Try `pip install vllm==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121` |
| `AttributeError: apply_turboquant_patch` | TurboQuant applies patch at import — remove the explicit call, keep the import |
| `Tool calls not working` | Confirm `--enable-auto-tool-choice --tool-call-parser hermes` in `start_server.py` |
| `Out of VRAM` | Lower `--max-model-len` (e.g. `4096`) or add `--gpu-memory-utilization 0.85` |
| Server slow first run | Normal — HuggingFace downloads ~16GB model. Subsequent starts use local cache |

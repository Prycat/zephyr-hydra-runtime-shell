# TurboQuant Integration Design
**Date:** 2026-04-08
**Project:** hermes-agent (local Hermes 3 tool-using agent)
**Status:** Approved

## Goal

Integrate TurboQuant (Google Research, 2026) into the local Hermes 3 agent to compress the LLM KV cache to 3-bit keys / 4-bit values, achieving up to 6x memory reduction and faster inference — without any accuracy loss.

## Background

TurboQuant is a training-free KV cache quantization algorithm using two techniques:
- **PolarQuant** — rotates key vectors to simplify geometry before quantization
- **QJL (Quantized Johnson-Lindenstrauss)** — corrects estimation biases for precision

It targets **vLLM 0.18.0** via monkey-patching of the attention mechanism using custom Triton kernels.

## Architecture

### Before
```
agent.py → Ollama :11434 → hermes3.gguf (Q4_0, in GGUF format)
```

### After
```
agent.py → vLLM :8000 → NousResearch/Hermes-3-Llama-3.1-8B (HuggingFace)
                              ↑
                  TurboQuant patches vLLM attention layer
                  KV cache: 3-bit keys, 4-bit values (quality/speed balance)
```

vLLM exposes an OpenAI-compatible API identical in format to Ollama's, so `agent.py` requires only a single-line change to `base_url`.

## Components

| File | Status | Purpose |
|------|--------|---------|
| `install.bat` | New | One-shot setup: clones TurboQuant from GitHub, installs vLLM 0.18.0 and TurboQuant |
| `start_server.py` | New | Applies TurboQuant monkey-patch, launches vLLM OpenAI server on port 8000 |
| `agent.py` | Modify | Change `base_url` from `http://localhost:11434/v1` to `http://localhost:8000/v1` |

Ollama remains installed and untouched — both backends can coexist.

## Data Flow

1. **Setup (once):** `install.bat` → clones TurboQuant, installs vLLM + TurboQuant
2. **Each session start:** `python start_server.py` → TurboQuant patches vLLM attention → vLLM loads `NousResearch/Hermes-3-Llama-3.1-8B` → server up on `:8000`
3. **Agent use:** `python agent.py` → user prompt → vLLM builds compressed KV cache (3K/4V bits via Triton kernels) → response → tool calls parsed and executed as before

## Quantization Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bits_k` | 3 | Keys: 3-bit per element (PolarQuant rotation improves accuracy at this level) |
| `bits_v` | 4 | Values: 4-bit (2-bit degrades cosine similarity to 0.94; 4-bit recommended) |

## Error Handling

- `start_server.py` checks for CUDA availability on startup and exits with a clear message if not found
- `agent.py` catches `ConnectionRefusedError` on first API call and prints a prompt to run `start_server.py`

## Requirements

- CUDA-capable NVIDIA GPU (vLLM + Triton kernels)
- ~16GB disk space for `NousResearch/Hermes-3-Llama-3.1-8B` HuggingFace model
- Python 3.9+ (already present)
- vLLM 0.18.0
- TurboQuant from `https://github.com/0xSero/turboquant`

## Out of Scope

- Modifying the agent's tools, prompt, or conversation logic
- Changing the Ollama setup (kept as fallback)
- Streaming responses (not currently used)

"""
config.py — Central runtime configuration for hermes-agent / prycat.

All Ollama URL constants are derived from the OLLAMA_HOST environment variable
so the same codebase works both natively (localhost) and in Docker (ollama service).

Usage:
    from config import OLLAMA_CHAT_URL, OLLAMA_TAGS_URL, OLLAMA_V1_URL
"""
import os

# In Docker this is set to http://ollama:11434 via docker-compose environment.
# Natively it defaults to localhost.
# Ollama itself sets OLLAMA_HOST=0.0.0.0:11434 (no scheme) — normalize it.
_raw = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if _raw and not _raw.startswith("http://") and not _raw.startswith("https://"):
    # Replace 0.0.0.0 with localhost so httpx can actually connect
    _raw = _raw.replace("0.0.0.0", "localhost")
    _raw = "http://" + _raw
OLLAMA_HOST = _raw

OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/v1/chat/completions"
OLLAMA_V1_URL   = f"{OLLAMA_HOST}/v1"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_GEN_URL  = f"{OLLAMA_HOST}/api/generate"

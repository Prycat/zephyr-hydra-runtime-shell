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
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/v1/chat/completions"
OLLAMA_V1_URL   = f"{OLLAMA_HOST}/v1"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_GEN_URL  = f"{OLLAMA_HOST}/api/generate"

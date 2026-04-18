"""
Ollama status checker for Hermes Agent.
Ollama runs as a background service on Windows — this script confirms it is
reachable and that hermes3:8b is available, then exits.

You do NOT need to run this manually. launch.bat handles it automatically.
"""

import sys
import json
import urllib.request
import urllib.error

from config import OLLAMA_HOST as OLLAMA_URL
MODEL = "hermes3:8b"


def check_ollama():
    print("=== Hermes Agent — Ollama Status ===\n")

    # Check Ollama is reachable
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        print("ERROR: Ollama is not running.")
        print("  • Open the Ollama app (system tray) or run 'ollama serve' in a terminal.")
        sys.exit(1)

    # Check hermes3:8b is pulled
    models = [m["name"] for m in data.get("models", [])]
    print(f"Ollama is running.  Available models: {', '.join(models) or '(none)'}")

    if not any(MODEL in m for m in models):
        print(f"\nWARNING: {MODEL} is not pulled yet.")
        print(f"  Run:  ollama pull {MODEL}")
        print("  Or re-run install.bat to download it automatically.")
        sys.exit(1)

    print(f"\n✓ {MODEL} is ready.")
    print(f"  API : {OLLAMA_URL}/v1")
    print("\nYou can now run:  python agent.py\n")


if __name__ == "__main__":
    check_ollama()

"""
zephyr_keys.py — Zephyr API Key Vault
Prycat Research Team

Keys are stored in ~/.zephyr/keys.json — completely outside the repo,
impossible to accidentally commit to GitHub. Safe for any user on any machine.

Supported providers:
  claude    → Anthropic (Claude Opus / Sonnet)
  gpt       → OpenAI (GPT-4o, o1)
  grok      → xAI (Grok-3, OpenAI-compatible endpoint)
  gemini    → Google (Gemini 2.0 Flash)

Usage:
  from zephyr_keys import KeyVault
  vault = KeyVault()
  vault.setup()          # interactive wizard
  key = vault.get("claude")
"""

import os
import json
import getpass

# ── Key vault location ────────────────────────────────────────────────────────
VAULT_DIR  = os.path.join(os.path.expanduser("~"), ".zephyr")
VAULT_FILE = os.path.join(VAULT_DIR, "keys.json")

# ── Provider definitions ──────────────────────────────────────────────────────
PROVIDERS = {
    "claude": {
        "name":        "Anthropic Claude",
        "models":      ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-3-5"],
        "default":     "claude-sonnet-4-5",
        "env_var":     "ANTHROPIC_API_KEY",
        "key_prefix":  "sk-ant-",
        "get_key_url": "https://console.anthropic.com/",
        "package":     "anthropic",
    },
    "gpt": {
        "name":        "OpenAI GPT-4o",
        "models":      ["gpt-4o", "gpt-4o-mini", "o1-mini"],
        "default":     "gpt-4o",
        "env_var":     "OPENAI_API_KEY",
        "key_prefix":  "sk-",
        "get_key_url": "https://platform.openai.com/api-keys",
        "package":     "openai",
    },
    "grok": {
        "name":        "xAI Grok",
        "models":      ["grok-3", "grok-3-fast"],
        "default":     "grok-3",
        "env_var":     "XAI_API_KEY",
        "key_prefix":  "xai-",
        "get_key_url": "https://console.x.ai/",
        "package":     "openai",   # Grok uses OpenAI-compatible API
    },
    "gemini": {
        "name":        "Google Gemini",
        "models":      ["gemini-2.0-flash", "gemini-1.5-pro"],
        "default":     "gemini-2.0-flash",
        "env_var":     "GEMINI_API_KEY",
        "key_prefix":  "AIza",
        "get_key_url": "https://aistudio.google.com/app/apikey",
        "package":     "google-generativeai",
    },
}

PROVIDER_PRIORITY = ["claude", "gpt", "grok", "gemini"]


class KeyVault:
    """Manages API keys stored safely in ~/.zephyr/keys.json"""

    def __init__(self):
        os.makedirs(VAULT_DIR, exist_ok=True)

    def _load(self) -> dict:
        if not os.path.exists(VAULT_FILE):
            return {}
        with open(VAULT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(VAULT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Restrict permissions on Unix (no-op on Windows)
        try:
            os.chmod(VAULT_FILE, 0o600)
        except Exception:
            pass

    def get(self, provider: str):
        """Get key for provider. Checks vault then environment variable."""
        p = provider.lower()
        # Check vault first
        data = self._load()
        if data.get(p):
            return data[p]
        # Fall back to environment variable
        cfg = PROVIDERS.get(p, {})
        env_val = os.environ.get(cfg.get("env_var", ""), "")
        return env_val if env_val else None

    def set(self, provider: str, key: str):
        """Store a key in the vault."""
        data = self._load()
        data[provider.lower()] = key
        self._save(data)

    def remove(self, provider: str):
        """Remove a key from the vault."""
        data = self._load()
        data.pop(provider.lower(), None)
        self._save(data)

    def configured(self) -> list[str]:
        """Return list of providers with keys configured."""
        result = []
        for p in PROVIDER_PRIORITY:
            if self.get(p):
                result.append(p)
        return result

    def mask(self, key: str) -> str:
        """Mask a key for display: sk-ant-...xyz"""
        if not key or len(key) < 8:
            return "****"
        return key[:8] + "..." + key[-4:]

    # ── Interactive setup ─────────────────────────────────────────────────────

    def setup(self, provider: str = None):
        """Interactive key setup wizard."""
        targets = [provider] if provider else list(PROVIDERS.keys())

        print("\n  Zephyr Key Vault Setup")
        print(f"  Keys stored in: {VAULT_FILE}")
        print("  (Never committed to git — safe for any user)\n")

        for p in targets:
            cfg = PROVIDERS[p]
            current = self.get(p)
            status = f"configured ({self.mask(current)})" if current else "not set"
            print(f"  [{p.upper()}] {cfg['name']}  —  {status}")
            print(f"  Get your key at: {cfg['get_key_url']}")

            try:
                raw = getpass.getpass(f"  Enter key (leave blank to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Setup cancelled.")
                return

            if raw:
                self.set(p, raw)
                print(f"  Saved {p} key: {self.mask(raw)}\n")
            else:
                print(f"  Skipped.\n")

        configured = self.configured()
        if configured:
            print(f"  Active providers: {', '.join(configured)}")
        else:
            print("  No providers configured yet.")
        print()

    def print_status(self):
        """Print provider status table."""
        print("\n  Zephyr Provider Status")
        print(f"  Vault: {VAULT_FILE}\n")
        print(f"  {'Provider':<10} {'Status':<20} {'Model':<25} Get key")
        print("  " + "-" * 75)
        for p in PROVIDER_PRIORITY:
            cfg = PROVIDERS[p]
            key = self.get(p)
            if key:
                status = f"OK ({self.mask(key)})"
                tick = "+"
            else:
                status = "not configured"
                tick = " "
            print(f"  [{tick}] {p:<8} {status:<20} {cfg['default']:<25} {cfg['get_key_url']}")
        print()


# ── API call routing ──────────────────────────────────────────────────────────

def call_claude(key: str, message: str, context: str = "") -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        system = f"You are a highly capable AI assistant consulting for Zephyr, a local research AI for the Prycat Research Team.\n{context}"
        response = client.messages.create(
            model=PROVIDERS["claude"]["default"],
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text
    except ImportError:
        return "Error: anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Claude error: {e}"


def call_gpt(key: str, message: str, context: str = "") -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        system = f"You are a highly capable AI assistant consulting for Zephyr, a local research AI for the Prycat Research Team.\n{context}"
        response = client.chat.completions.create(
            model=PROVIDERS["gpt"]["default"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": message},
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT error: {e}"


def call_grok(key: str, message: str, context: str = "") -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        system = f"You are a highly capable AI assistant consulting for Zephyr, a local research AI for the Prycat Research Team.\n{context}"
        response = client.chat.completions.create(
            model=PROVIDERS["grok"]["default"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": message},
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Grok error: {e}"


def call_gemini(key: str, message: str, context: str = "") -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name=PROVIDERS["gemini"]["default"],
            system_instruction=f"You are a highly capable AI assistant consulting for Zephyr, a local research AI for the Prycat Research Team.\n{context}",
        )
        response = model.generate_content(message)
        return response.text
    except ImportError:
        return "Error: google-generativeai not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Gemini error: {e}"


CALLERS = {
    "claude":  call_claude,
    "gpt":     call_gpt,
    "codex":   call_gpt,   # alias
    "openai":  call_gpt,   # alias
    "grok":    call_grok,
    "gemini":  call_gemini,
    "google":  call_gemini, # alias
}


def call_provider(provider: str, message: str, history: list = None) -> tuple[str, str]:
    """
    Route a message to a provider.
    Returns (provider_used, response_text).
    Falls back through priority list if provider='auto'.
    """
    vault = KeyVault()

    # Build brief context from recent history
    context = ""
    if history:
        recent = [m for m in history[-6:] if m.get("role") in ("user", "assistant") and m.get("content")]
        if recent:
            lines = []
            for m in recent:
                role = "Human" if m["role"] == "user" else "Zephyr"
                lines.append(f"{role}: {m['content'][:200]}")
            context = "Recent conversation context:\n" + "\n".join(lines)

    # Resolve provider
    if provider in ("auto", "best", ""):
        configured = vault.configured()
        if not configured:
            return "none", "No providers configured. Run /keys setup first."
        provider = configured[0]

    canonical = provider.lower()
    caller = CALLERS.get(canonical)
    if not caller:
        available = ", ".join(PROVIDERS.keys())
        return provider, f"Unknown provider '{provider}'. Available: {available}"

    # Map aliases back to canonical for key lookup
    key_name = "gpt" if canonical in ("codex", "openai") else \
               "gemini" if canonical == "google" else canonical

    key = vault.get(key_name)
    if not key:
        cfg = PROVIDERS.get(key_name, {})
        return provider, (
            f"No key for '{key_name}'. Run /keys setup to add it.\n"
            f"Get a key at: {cfg.get('get_key_url', '')}"
        )

    response = caller(key, message, context)
    return provider, response

"""
Zephyr — Prycat Research Team
Local AI agent via Ollama with an expanding tool suite and CLI commands.
"""

import sys
import json
import math
import datetime
import subprocess
import tempfile
import os
import httpx
from openai import OpenAI

# Provider key vault
try:
    from zephyr_keys import KeyVault, call_provider, PROVIDERS, PROVIDER_PRIORITY
    KEYS_AVAILABLE = True
except ImportError:
    KEYS_AVAILABLE = False

# Blackwellian loop integration
try:
    from blackwell.logger import new_session, log_exchange, init_db
    from blackwell.regret import average_payoff_vector, regret_vector, highest_regret_dims, print_status
    from blackwell.planning import run_planning_session, get_world_model_context
    LOGGING = True
    init_db()
except ImportError:
    LOGGING = False

# Trajectory logging (always-on, independent of full LOGGING)
try:
    from blackwell.trajectory import log_success, log_failure, mark_feedback, get_counts
    from blackwell.background_eval import get_evaluator
    TRAJECTORY = True
except ImportError:
    TRAJECTORY = False

# MCP skills bridge (MemPalace · Serena · Ruflo)
try:
    import tools_mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "hermes3:8b"

# ─── Tool definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, log, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate e.g. '2**10'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write to."},
                    "content": {"type": "string", "description": "Text to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query. Be specific — include dates or version numbers when relevant."},
                    "max_results": {"type": "integer", "description": "Number of results (default 5, max 10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": "Fetch a URL and return its readable text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL starting with http:// or https://"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python code snippet and return stdout/stderr. Use httpx for HTTP requests — never use requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code. Use print() for output. Use httpx not requests."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make a raw HTTP request to any API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "description": "GET, POST, PUT, DELETE, PATCH"},
                    "url": {"type": "string", "description": "Full URL."},
                    "headers": {"type": "object", "description": "Optional headers."},
                    "body": {"type": "string", "description": "Optional request body."},
                },
                "required": ["method", "url"],
            },
        },
    },
]

# ─── Tool implementations ────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns["__builtins__"] = {}
    try:
        return str(eval(expression, safe_ns))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"

def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return f"Error: {e}"

def web_search(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS
        max_results = min(max_results or 5, 10)
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('href', '')}")
            lines.append(f"   {r.get('body', '')[:200]}")
            lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Search error: {e}"

def browse_url(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        if not url.startswith("http://") and not url.startswith("https://"):
            return f"Error: URL must start with http:// or https://. Got: {url!r}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Zephyr-Agent/1.0)"}
        resp = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        if len(text) > 3000:
            text = text[:3000] + f"\n\n[truncated — {len(text)} chars total]"
        return text
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
    except Exception as e:
        return f"Browse error: {e}"

def run_python(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(["python", tmp_path], capture_output=True, text=True, timeout=10)
        os.unlink(tmp_path)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 10 seconds."
    except Exception as e:
        return f"Error: {e}"

def http_request(method: str, url: str, headers: dict = None, body: str = None) -> str:
    try:
        kwargs = {"timeout": 15, "follow_redirects": True}
        if headers:
            kwargs["headers"] = headers
        if body:
            kwargs["content"] = body
        resp = httpx.request(method.upper(), url, **kwargs)
        try:
            body_text = json.dumps(resp.json(), indent=2)[:2000]
        except Exception:
            body_text = resp.text[:2000]
        return f"Status: {resp.status_code}\n\n{body_text}"
    except Exception as e:
        return f"HTTP error: {e}"

TOOL_HANDLERS = {
    "calculate":        lambda args: calculate(**args),
    "get_current_time": lambda args: get_current_time(),
    "read_file":        lambda args: read_file(**args),
    "write_file":       lambda args: write_file(**args),
    "web_search":       lambda args: web_search(**args),
    "browse_url":       lambda args: browse_url(**args),
    "run_python":       lambda args: run_python(**args),
    "http_request":     lambda args: http_request(**args),
}

# ─── MCP skill registration (background — don't block startup) ───────────────
if MCP_AVAILABLE:
    import threading as _mcp_thread
    _mcp_thread.Thread(
        target=tools_mcp.register_mcp_tools,
        args=(TOOLS, TOOL_HANDLERS),
        daemon=True,
        name="mcp-init",
    ).start()

# ─── System prompt ────────────────────────────────────────────────────────────

_MCP_SKILLS_BLOCK = """
SKILLS (MCP-powered, use when appropriate)
MEMORY — persistent across sessions:
- memory_store        — save a fact, preference, or summary for later
- memory_search       — semantic search your stored memories by topic
- memory_list         — list recent memories
- memory_delete       — remove a specific memory by id
- memory_update       — update an existing memory

CODE INTELLIGENCE — for reading this codebase:
- code_overview       — list all symbols in a file (classes, functions)
- code_find_symbol    — find a symbol by name across the project
- code_search         — regex/pattern search across all files
- code_read           — read a file or specific lines
- code_references     — find all callers/users of a symbol
- code_list_dir       — list files in a directory

AGENT DELEGATION — for complex multi-step tasks:
- agent_task          — delegate a task to a specialist agent
- agent_search        — web access via Ruflo agent
- agent_analyze       — analyze code or content (diff/risk analysis)
- agent_generate      — generate content via WASM-sandboxed agent
- agent_status        — check status of a running agent

""" if MCP_AVAILABLE else ""

SYSTEM_PROMPT = f"""You are Zephyr, a sharp AI research assistant and member of the Prycat research team.

IDENTITY
- Your name is Zephyr. Always introduce yourself as Zephyr.
- You run locally via Ollama on the user's machine.

TOOLS AVAILABLE
- calculate        — math expressions
- get_current_time — current date/time
- read_file        — read a local file
- write_file       — write a local file
- web_search       — DuckDuckGo search (be specific: include dates, versions, names)
- browse_url       — fetch any webpage (pass the EXACT url given, never modify it)
- run_python       — run Python code (use httpx for HTTP — NEVER use requests)
- http_request     — raw HTTP call to any API
{_MCP_SKILLS_BLOCK}TOOL CALL RULES — follow these precisely
1. When you decide to use a tool, call it immediately. Never paste JSON or XML tool syntax into your reply.
2. For browse_url: use the exact URL provided by the user. Do not guess or alter it.
3. For run_python: always import httpx, never import requests.
4. For web_search: make your query specific. Add the current year or topic keywords. Never search generic terms.
5. After getting a tool result, summarise it concisely — don't repeat the raw output verbatim.
6. For questions about yourself, your tools, your capabilities, or your feelings — answer directly from memory. NEVER call read_file, code_read, or any other tool for introspective questions. Tools are for external data only.
7. Every tool has a strict parameter list. NEVER invent parameter names. If unsure of the correct parameters, answer without using a tool.

RESPONSE RULES
- Be concise. One paragraph max unless detail is explicitly requested.
- If you don't know something, say "I don't know" in one sentence — don't pad.
- Never say "I'm just an AI" or give disclaimers. Just answer.
- Never leak raw tool call syntax, JSON brackets, or XML tags into your replies.

EXAMPLE — correct tool use:
User: "what's the weather at https://wttr.in/?format=3"
You think: I should call browse_url with url="https://wttr.in/?format=3"
[call browse_url] → get result → summarise in one line."""

# ─── CLI command handler ──────────────────────────────────────────────────────

CLI_COMMANDS = {
    "/help":      "Show this help message",
    "/tools":     "List all of Zephyr's tools",
    "/search":    "Direct web search — /search <query>",
    "/browse":    "Fetch a URL directly — /browse <url>",
    "/run":       "Run Python code directly — /run <code>",
    "/status":    "Check Ollama connection",
    "/model":     "Show current model info",
    "/save":      "Save conversation to Obsidian vault — /save [filename]",
    "/clear":     "Clear conversation history",
    "/blackwell": "Enter Zephyr's planning space — he asks, you answer, he grows",
    "/coding-blackwell": "CS-focused planning session — sharpens Zephyr's coding instincts",
    "/keys":      "Manage API keys — /keys setup | list | clear <provider>",
    "/call":      "Consult an external AI — /call [claude|gpt|grok|gemini] <message>",
    "/trajectory": "Show trajectory pair counts and current regret vector",
    "/feedback":   "Mark last response good/bad — /feedback <session_id> <turn> up|down",
    "/exit":      "Exit Zephyr",
}

def handle_cli(cmd: str, history: list[dict]) -> tuple[bool, list[dict]]:
    """
    Handle a /command. Returns (should_continue, updated_history).
    Returns (False, history) to signal exit.
    """
    global MODEL, client
    parts = cmd.strip().split(" ", 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/help", "/?"):
        print("\nZephyr CLI Commands:")
        for c, desc in CLI_COMMANDS.items():
            print(f"  {c:<10} {desc}")
        print()

    elif command == "/tools":
        print("\nZephyr's Tools:")
        tool_info = [
            ("calculate",        "Evaluate math expressions"),
            ("get_current_time", "Get current date and time"),
            ("read_file",        "Read a local file"),
            ("write_file",       "Write a local file"),
            ("web_search",       "Search the web with DuckDuckGo"),
            ("browse_url",       "Fetch and read any webpage"),
            ("run_python",       "Execute Python code snippets"),
            ("http_request",     "Make raw HTTP API calls"),
        ]
        if MCP_AVAILABLE:
            tool_info += [
                ("── MEMORY ──",         ""),
                ("memory_store",         "Store a fact or summary for later"),
                ("memory_search",        "Semantic search across stored memories"),
                ("memory_list",          "List recent memories"),
                ("memory_delete",        "Remove a memory by id"),
                ("memory_update",        "Update an existing memory"),
                ("── CODE ──",           ""),
                ("code_overview",        "List symbols in a file"),
                ("code_find_symbol",     "Find a symbol across the codebase"),
                ("code_search",          "Regex search across files"),
                ("code_read",            "Read a file or line range"),
                ("code_references",      "Find references to a symbol"),
                ("code_list_dir",        "List directory contents"),
                ("── AGENTS ──",         ""),
                ("agent_task",           "Delegate task to specialist agent"),
                ("agent_search",         "Web access via Ruflo agent"),
                ("agent_analyze",        "Analyze code or content"),
                ("agent_generate",       "Generate content via WASM agent"),
                ("agent_status",         "Check status of running agent"),
            ]
        for name, desc in tool_info:
            print(f"  {name:<20} {desc}")
        print()

    elif command == "/search":
        if not arg:
            print("Usage: /search <query>\n")
        else:
            print(f"Searching: {arg}")
            print(web_search(arg))

    elif command == "/browse":
        if not arg:
            print("Usage: /browse <url>\n")
        else:
            print(f"Fetching: {arg}\n")
            print(browse_url(arg))

    elif command == "/run":
        if not arg:
            print("Usage: /run <python code>\n")
        else:
            print(run_python(arg))

    elif command == "/status":
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            print(f"\nOllama: Online")
            print(f"Models: {', '.join(models)}")
            print(f"Active: {MODEL}\n")
        except Exception as e:
            print(f"\nOllama: Offline ({e})\n")

    elif command == "/trajectory":
        if TRAJECTORY:
            counts = get_counts()
            print(f"\nTrajectory pairs:   {counts['success']} success  /  "
                  f"{counts['failed']} failed  /  {counts['feedback']} feedback")
        if LOGGING:
            from blackwell.logger import get_average_vector
            from blackwell.evaluator import total_regret
            avg = get_average_vector()
            if avg:
                dims = "  ".join(f"{k}={v:.2f}" for k, v in avg.items()
                                 if k != "n")
                print(f"Current x\u0305:          {dims}")
                print(f"Total regret:        {total_regret(avg):.3f}  (threshold: 0.15)")
            else:
                print("No scored exchanges yet — chat more to build x\u0305")
        print()

    elif command == "/feedback":
        parts_f = arg.strip().split()
        if len(parts_f) == 3 and TRAJECTORY:
            sess_f, turn_f, vote = parts_f
            positive = vote.lower() in ("up", "good", "1")
            mark_feedback(sess_f, int(turn_f), positive)
            print(f"[feedback] {'👍' if positive else '👎'} recorded for turn {turn_f}\n")
        else:
            print("Usage: /feedback <session_id> <turn> up|down\n")

    elif command == "/model":
        model_name = arg.strip()
        if model_name:
            MODEL = model_name
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            print(f"[MODEL] switched to {MODEL}")
        else:
            print(f"\nModel : {MODEL}")
            print(f"API   : http://localhost:11434/v1")
            print(f"Team  : Prycat Research\n")

    elif command == "/save":
        vault_dir = r"C:\Users\gamer23\Desktop\vault 1\all zephyr conversations"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        base_name = arg if arg else f"Zephyr {timestamp}"
        # Ensure .md extension
        if not base_name.endswith(".md"):
            base_name += ".md"
        try:
            os.makedirs(vault_dir, exist_ok=True)
            filepath = os.path.join(vault_dir, base_name)
            with open(filepath, "w", encoding="utf-8") as f:
                # Obsidian frontmatter
                f.write(f"---\n")
                f.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
                f.write(f"time: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"model: {MODEL}\n")
                f.write(f"tags: [zephyr, conversation]\n")
                f.write(f"---\n\n")
                f.write(f"# Zephyr — {timestamp}\n\n")
                # Write each message
                for msg in history:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if not content or role == "system":
                        continue
                    if role == "user":
                        f.write(f"**You:** {content}\n\n")
                    elif role == "assistant":
                        f.write(f"**Zephyr:** {content}\n\n")
                        f.write("---\n\n")
            print(f"Saved to Obsidian vault:\n  {filepath}\n")
        except Exception as e:
            print(f"Error saving: {e}\n")

    elif command == "/clear":
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("History cleared.\n")

    elif command == "/keys":
        if not KEYS_AVAILABLE:
            print("Key vault not available. Check zephyr_keys.py.\n")
        else:
            vault = KeyVault()
            sub = arg.split()[0].lower() if arg else "list"
            if sub == "setup":
                # Optional: setup a specific provider
                specific = arg.split()[1].lower() if len(arg.split()) > 1 else None
                vault.setup(provider=specific)
            elif sub == "list":
                vault.print_status()
            elif sub == "clear":
                provider = arg.split()[1].lower() if len(arg.split()) > 1 else None
                if not provider:
                    print("Usage: /keys clear <provider>\n")
                else:
                    vault.remove(provider)
                    print(f"Removed key for '{provider}'.\n")
            else:
                print("Usage: /keys setup | list | clear <provider>\n")

    elif command == "/call":
        if not KEYS_AVAILABLE:
            print("Key vault not available. Check zephyr_keys.py.\n")
        elif not arg:
            print("Usage: /call <message>  or  /call <provider> <message>")
            print("Providers: claude, gpt, grok, gemini  (or leave blank for best available)\n")
        else:
            # Parse: /call claude <message>  or  /call <message>
            words = arg.split(None, 1)
            known = {"claude", "gpt", "grok", "gemini", "codex", "openai", "google", "auto"}
            if words[0].lower() in known:
                provider = words[0].lower()
                message  = words[1] if len(words) > 1 else ""
            else:
                provider = "auto"
                message  = arg

            if not message:
                print("Please include a message. E.g. /call claude explain this code\n")
            else:
                print(f"\n  Calling {provider if provider != 'auto' else 'best available'}...\n")
                used, response = call_provider(provider, message, history)
                label = used.upper()
                print(f"  [{label}]: {response}\n")

                # Save to Obsidian vault automatically
                vault_dir  = r"C:\Users\gamer23\Desktop\vault 1\all zephyr conversations"
                timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                safe_msg   = message[:40].replace("/", "-").replace("\\", "-").replace(":", "-")
                md_name    = f"Call - {used} - {timestamp}.md"
                try:
                    os.makedirs(vault_dir, exist_ok=True)
                    filepath = os.path.join(vault_dir, md_name)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"---\n")
                        f.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
                        f.write(f"time: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
                        f.write(f"provider: {used}\n")
                        f.write(f"tags: [zephyr, call, {used}]\n")
                        f.write(f"---\n\n")
                        f.write(f"# Call → {label}\n\n")
                        f.write(f"## Question\n\n{message}\n\n")
                        f.write(f"## Response\n\n{response}\n")
                    print(f"  Saved to vault: {md_name}\n")
                except Exception as e:
                    print(f"  (Vault save failed: {e})\n")

                # Inject into history so Zephyr can reference it
                history.append({
                    "role": "assistant",
                    "content": f"[Consulted {label}]: {response}"
                })

    elif command == "/blackwell":
        if not LOGGING:
            print("Blackwell module not available. Check blackwell/ directory.\n")
        else:
            avg_v   = average_payoff_vector()
            r       = regret_vector(avg_v)
            targets = highest_regret_dims(r, top_n=2)
            print_status(avg_v)
            run_planning_session(r, targets)
            # After session, rebuild system prompt with updated world model
            wm_context = get_world_model_context()
            if wm_context:
                updated_prompt = SYSTEM_PROMPT + f"\n\nWORLD MODEL (built from planning sessions):\n{wm_context}"
                # Update system message in history
                if history and history[0]["role"] == "system":
                    history[0]["content"] = updated_prompt
                print("  Zephyr's world model injected into context.\n")

    elif command == "/coding-blackwell":
        if not LOGGING:
            print("Blackwell module not available. Check blackwell/ directory.\n")
        else:
            from blackwell.planning import run_coding_planning_session, get_coding_world_model_context
            run_coding_planning_session()
            coding_context = get_coding_world_model_context()
            if coding_context:
                updated_prompt = SYSTEM_PROMPT + f"\n\nCODING WORLD MODEL (built from coding planning sessions):\n{coding_context}"
                if history and history[0]["role"] == "system":
                    history[0]["content"] = updated_prompt
                print("  Coding world model injected into context.\n")

    elif command in ("/exit", "/quit"):
        print("Bye!")
        return False, history

    else:
        print(f"Unknown command: {command}. Type /help for a list.\n")

    return True, history

# ─── Agent loop ──────────────────────────────────────────────────────────────

def run_agent(user_message: str, history: list[dict],
              tools_called: list = None,
              stream_cb=None) -> tuple[str, list[dict]]:
    """
    stream_cb: optional callable(token: str) — called for each streamed text token.
    Returns (full_reply, updated_history).
    """
    history = history + [{"role": "user", "content": user_message}]
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=list(TOOLS),  # snapshot so background MCP init can't mutate mid-call
            tool_choice="auto",
            stream=True,
        )

        full_content = ""
        tool_calls_acc: dict = {}   # index → {id, name, arguments}

        for chunk in response:
            choice = chunk.choices[0]
            delta  = choice.delta

            # ── Text token ──────────────────────────────────────
            if delta.content:
                full_content += delta.content
                if stream_cb:
                    stream_cb(delta.content)

            # ── Tool-call delta ──────────────────────────────────
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

        if tool_calls_acc:
            tool_calls_list = [
                {
                    "id":   tool_calls_acc[i]["id"],
                    "type": "function",
                    "function": {
                        "name":      tool_calls_acc[i]["name"],
                        "arguments": tool_calls_acc[i]["arguments"],
                    },
                }
                for i in sorted(tool_calls_acc)
            ]
            history.append({
                "role":       "assistant",
                "content":    full_content or None,
                "tool_calls": tool_calls_list,
            })
            for tc_data in tool_calls_list:
                fn_name = tc_data["function"]["name"]
                fn_args = json.loads(tc_data["function"]["arguments"])
                print(f"  [tool] {fn_name}({fn_args})", flush=True)
                if tools_called is not None:
                    tools_called.append(fn_name)
                handler = TOOL_HANDLERS.get(fn_name)
                if not handler:
                    result = f"Unknown tool: {fn_name}"
                else:
                    try:
                        result = handler(fn_args)
                    except TypeError as e:
                        # Model passed wrong parameter names — tell it so it can retry correctly
                        result = (
                            f"Tool call failed: {e}. "
                            f"You called '{fn_name}' with unexpected arguments {list(fn_args.keys())}. "
                            f"Check the required parameters for '{fn_name}' and call it again with the "
                            f"correct argument names, or answer the question without using a tool."
                        )
                history.append({
                    "role":         "tool",
                    "tool_call_id": tc_data["id"],
                    "content":      str(result),
                })
            # Loop back — get the final (non-tool) reply
        else:
            history.append({"role": "assistant", "content": full_content})
            return full_content, history

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Startup splash ────────────────────────────────────────
    # Only show splash in a real terminal — skip when piped (e.g. GUI subprocess)
    if sys.stdout.isatty():
        try:
            from dragon_splash import show_splash
            show_splash()
        except Exception:
            print("Zephyr — Prycat Research Team (local via Ollama)\n")

    print("Type /help for commands, /exit to quit.\n")

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: Ollama is not running. Start it from your system tray.\n")
        input("Press Enter to exit...")
        return

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Start a logging session
    session_id = None
    turn = 0
    if LOGGING:
        session_id = new_session(MODEL)
    # Always emit a session ID for GUI feedback tracking
    import uuid as _uuid
    _gui_session = session_id if session_id else str(_uuid.uuid4())
    print(f"<<SESSION:{_gui_session}>>", flush=True)

    while True:
        try:
            # Only print the "You: " prompt in a real terminal.
            # When running as a GUI subprocess stdout is a pipe — the prompt
            # has no trailing newline so it gets glued onto the next print()
            # call (e.g. a tool notification), corrupting the output.
            _prompt = "You: " if sys.stdout.isatty() else ""
            user_input = input(_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Handle CLI commands
        if user_input.startswith("/"):
            should_continue, history = handle_cli(user_input, history)
            if not should_continue:
                break
            continue

        # Regular chat
        try:
            tools_called = []
            _streaming_started = [False]

            def _on_token(token: str):
                if not _streaming_started[0]:
                    # Emit stream-start marker on its own line so the GUI
                    # can switch to streaming mode before the first token.
                    print("<<ZS>>", flush=True)
                    _streaming_started[0] = True
                # Each token is its own line with a SOH prefix so the
                # line-based reader in the GUI can detect it instantly.
                sys.stdout.write(f"\x01{token}\n")
                sys.stdout.flush()

            reply, history = run_agent(user_input, history, tools_called,
                                       stream_cb=_on_token)

            if _streaming_started[0]:
                print("<<ZE>>", flush=True)
            else:
                # Fallback: model returned nothing (shouldn't happen normally)
                print(f"\nZephyr: {reply or '(no response)'}\n", flush=True)

            # Log to Blackwell DB + trajectory
            turn += 1
            if LOGGING and session_id:
                exchange_id = log_exchange(session_id, turn, user_input, reply, tools_called)
                if TRAJECTORY:
                    log_success(session_id, turn, user_input, reply, tools_called)
                    get_evaluator().submit(exchange_id, user_input, reply)
            elif TRAJECTORY:
                log_success(_gui_session, turn, user_input, reply, tools_called)
        except Exception as e:
            print(f"Error: {e}\n", flush=True)
            print("<<ZE>>", flush=True)  # clear GUI loading state
            if TRAJECTORY:
                log_failure(
                    _gui_session,
                    turn + 1,
                    user_input if 'user_input' in dir() else "",
                    "exception",
                    str(e),
                )

if __name__ == "__main__":
    main()

# MCP Skills Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give hermes (`agent.py`, `hermes3:8b` via Ollama) access to 16 curated tools from three MCP servers — MemPalace (memory), Serena (code intelligence), Ruflo (agent orchestration) — via a new `tools_mcp.py` bridge module.

**Architecture:** A new `tools_mcp.py` owns an `McpServer` class that manages stdio JSON-RPC subprocesses for each server. At startup `register_mcp_tools(TOOLS, TOOL_HANDLERS)` is called from `agent.py`, which launches each server lazily, reads its `tools/list` manifest, and injects curated tool definitions into the existing dicts. Failures are non-fatal — any server that can't start is silently skipped.

**Tech Stack:** Python 3.9, `subprocess`, `json`, `threading` (for stdout reader), `mempalace` (pip), Serena MCP server (uvx/pip), Ruflo (npx/Node.js).

---

### Task 1: Install prerequisites and verify MCP servers start

**Files:**
- No code changes — environment setup only

**Step 1: Install MemPalace**

Run:
```
pip install mempalace
```
Expected: installs without error.

**Step 2: Initialize MemPalace database**

Run:
```
mempalace init C:\Users\gamer23\Desktop\hermes-agent\mempalace_db
```
Expected: prints something like `Initialized MemPalace at ...`

**Step 3: Verify MemPalace MCP server starts and emits tools/list**

Run this one-liner (send initialize then tools/list, read one response):
```
python -c "
import subprocess, json, sys
p = subprocess.Popen(['python','-m','mempalace.mcp_server'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
p.stdin.write(json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'test','version':'0.0.1'}}}) + '\n')
p.stdin.flush()
print('INIT:', p.stdout.readline()[:120])
p.stdin.write(json.dumps({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}}) + '\n')
p.stdin.flush()
resp = json.loads(p.stdout.readline())
names = [t['name'] for t in resp.get('result',{}).get('tools',[])]
print('TOOLS:', names)
p.terminate()
"
```
Expected: prints a list of tool names. **Record the exact names** — you'll need them for `CURATED` in Task 2.

**Step 4: Verify Serena MCP server command**

Check what command the Serena plugin uses. Run:
```
python -c "import json; d=json.load(open(r'C:\Users\gamer23\.claude\settings.json')); [print(k,v) for k,v in d.get('mcpServers',{}).items() if 'serena' in k.lower()]"
```
Expected: prints the Serena server entry with its command + args. **Record the full command.**

**Step 5: Verify Ruflo MCP server starts**

Run (requires Node.js):
```
node --version
npx ruflo@latest mcp start --help
```
Expected: Node.js version + Ruflo help text. If Node.js is missing, install from https://nodejs.org.

**Step 6: Verify Ruflo tools/list**

Same pattern as Step 3 but with the Ruflo command:
```
python -c "
import subprocess, json
p = subprocess.Popen(['npx','ruflo@latest','mcp','start'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
import time; time.sleep(3)  # Ruflo needs warmup
p.stdin.write(json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'test','version':'0.0.1'}}}) + '\n')
p.stdin.flush()
print('INIT:', p.stdout.readline()[:120])
p.stdin.write(json.dumps({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}}) + '\n')
p.stdin.flush()
resp = json.loads(p.stdout.readline())
names = [t['name'] for t in resp.get('result',{}).get('tools',[])]
print('TOOLS (first 20):', names[:20])
p.terminate()
"
```
Expected: prints tool names. **Record the exact names** for the 5 Ruflo tools.

**No commit for this task — environment only.**

---

### Task 2: Create `tools_mcp.py` — McpServer class

**Files:**
- Create: `C:\Users\gamer23\Desktop\hermes-agent\tools_mcp.py`

**Step 1: Write `tools_mcp.py` with McpServer class**

```python
# -*- coding: utf-8 -*-
"""
tools_mcp.py — MCP stdio bridge for hermes agent.
Manages MemPalace, Serena, and Ruflo MCP server subprocesses.
"""
import json
import subprocess
import sys
import threading
import time
from typing import Any

# ── Curated tool sets ────────────────────────────────────────────────────────
# Map: server_key → {mcp_tool_name: hermes_tool_name}
# Update mcp_tool_name values after verifying with Task 1's tools/list output.

CURATED: dict[str, dict[str, str]] = {
    "mempalace": {
        # mcp_name          : hermes_name
        "store_memory"      : "memory_store",
        "search_memories"   : "memory_search",
        "list_memories"     : "memory_list",
        "delete_memory"     : "memory_delete",
        "update_memory"     : "memory_update",
    },
    "serena": {
        "get_symbols_overview"      : "code_overview",
        "find_symbol"               : "code_find_symbol",
        "search_for_pattern"        : "code_search",
        "read_file"                 : "code_read",
        "find_referencing_symbols"  : "code_references",
        "list_dir"                  : "code_list_dir",
    },
    "ruflo": {
        "execute_task"      : "agent_task",
        "web_search"        : "agent_search",
        "analyze_content"   : "agent_analyze",
        "generate_content"  : "agent_generate",
        "get_task_status"   : "agent_status",
    },
}

# ── Server launch commands ───────────────────────────────────────────────────
# Update SERENA_CMD after checking settings.json in Task 1 Step 4.

SERVER_CMDS: dict[str, list[str]] = {
    "mempalace": [sys.executable, "-m", "mempalace.mcp_server"],
    "serena":    ["uvx", "serena-code"],          # update if different
    "ruflo":     ["npx", "ruflo@latest", "mcp", "start"],
}

# ── McpServer class ──────────────────────────────────────────────────────────

class McpServer:
    """Manages a single MCP server subprocess (stdio JSON-RPC)."""

    TIMEOUT = 10.0  # seconds

    def __init__(self, key: str, cmd: list[str]):
        self.key   = key
        self.cmd   = cmd
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._id   = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _start(self) -> bool:
        """Launch subprocess. Returns True on success."""
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            # Allow slow starters (npx, uvx) to warm up
            time.sleep(1.5)
            # Send initialize handshake
            self._send({
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "hermes", "version": "1.0"},
                },
            })
            return True
        except FileNotFoundError as e:
            print(f"[tools_mcp] {self.key}: command not found — {self.cmd[0]} ({e})",
                  file=sys.stderr)
            self._proc = None
            return False
        except Exception as e:
            print(f"[tools_mcp] {self.key}: failed to start — {e}", file=sys.stderr)
            self._proc = None
            return False

    def _ensure_running(self) -> bool:
        """Start if not running. Returns True if ready."""
        if self._proc and self._proc.poll() is None:
            return True
        return self._start()

    # ── JSON-RPC ──────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, payload: dict) -> dict | None:
        """Send a JSON-RPC request and read the response. Returns result or None."""
        if not self._proc:
            return None
        req_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": req_id, **payload}
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
            # Read lines until we get our response id
            deadline = time.monotonic() + self.TIMEOUT
            while time.monotonic() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    break
                try:
                    resp = json.loads(line)
                    if resp.get("id") == req_id:
                        return resp.get("result")
                except json.JSONDecodeError:
                    continue
            print(f"[tools_mcp] {self.key}: timeout waiting for id={req_id}",
                  file=sys.stderr)
            return None
        except Exception as e:
            print(f"[tools_mcp] {self.key}: send error — {e}", file=sys.stderr)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        """Return raw MCP tool definitions from server manifest."""
        with self._lock:
            if not self._ensure_running():
                return []
            result = self._send({"method": "tools/list", "params": {}})
            if result is None:
                return []
            return result.get("tools", [])

    def call_tool(self, mcp_name: str, arguments: dict) -> str:
        """Call a tool and return its text result."""
        with self._lock:
            if not self._ensure_running():
                return f"[{self.key}] server unavailable"
            result = self._send({
                "method": "tools/call",
                "params": {"name": mcp_name, "arguments": arguments},
            })
            if result is None:
                return f"[{self.key}] timeout or error calling {mcp_name}"
            # MCP result is {"content": [{"type": "text", "text": "..."}]}
            content = result.get("content", [])
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(parts) if parts else str(result)

    def stop(self):
        """Terminate the subprocess."""
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


# ── Registry ─────────────────────────────────────────────────────────────────

_SERVERS: dict[str, McpServer] = {}


def register_mcp_tools(tools_list: list, handlers: dict) -> None:
    """
    Inject curated MCP tools into agent.py's TOOLS[] and TOOL_HANDLERS{}.
    Call once after TOOL_HANDLERS is defined in agent.py.
    Servers are started lazily on first tool call.
    """
    for server_key, tool_map in CURATED.items():
        cmd = SERVER_CMDS.get(server_key)
        if not cmd:
            continue

        server = McpServer(server_key, cmd)
        _SERVERS[server_key] = server

        # Fetch manifest to get real parameter schemas
        manifest = {t["name"]: t for t in server.list_tools()}

        for mcp_name, hermes_name in tool_map.items():
            mcp_def = manifest.get(mcp_name)
            if mcp_def is None:
                print(f"[tools_mcp] {server_key}: tool '{mcp_name}' not in manifest — skipping",
                      file=sys.stderr)
                continue

            # Build OpenAI-compatible tool definition from MCP schema
            tool_def = {
                "type": "function",
                "function": {
                    "name": hermes_name,
                    "description": mcp_def.get("description", f"{server_key} tool"),
                    "parameters": mcp_def.get("inputSchema", {
                        "type": "object", "properties": {}, "required": [],
                    }),
                },
            }
            tools_list.append(tool_def)

            # Capture loop vars for closure
            _key   = server_key
            _mname = mcp_name
            _srv   = server
            handlers[hermes_name] = lambda args, s=_srv, n=_mname: s.call_tool(n, args)

    registered = [h for h in handlers if any(
        h in tool_map.values() for tool_map in CURATED.values()
    )]
    print(f"[tools_mcp] registered {len(registered)} MCP tools: {registered}",
          file=sys.stderr)
```

**Step 2: Syntax check**

Run:
```
python -c "import ast; ast.parse(open(r'C:\Users\gamer23\Desktop\hermes-agent\tools_mcp.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add tools_mcp.py
git commit -m "feat: add tools_mcp.py MCP stdio bridge (McpServer class + register_mcp_tools)"
```

---

### Task 3: Update CURATED names from live manifests

**Files:**
- Modify: `C:\Users\gamer23\Desktop\hermes-agent\tools_mcp.py` — `CURATED` dict and `SERVER_CMDS`

**Context:** The MCP tool names in `CURATED` are best-guess from READMEs. Now that `McpServer` works, verify actual names.

**Step 1: Run the manifest checker script**

```python
# Run via: python check_mcp_names.py
import tools_mcp, sys

for key, cmd in tools_mcp.SERVER_CMDS.items():
    print(f"\n=== {key} ===")
    srv = tools_mcp.McpServer(key, cmd)
    tools = srv.list_tools()
    if tools:
        for t in tools:
            print(f"  {t['name']}")
    else:
        print("  (failed to start)")
    srv.stop()
```

Save as `check_mcp_names.py` in the repo root and run:
```
python check_mcp_names.py
```
Expected: prints real tool names for each server.

**Step 2: Update CURATED and SERVER_CMDS**

For each server, replace the `mcp_name` keys in `CURATED` with the exact names printed in Step 1.
Also update `SERVER_CMDS["serena"]` if the command from Task 1 Step 4 differs from `["uvx", "serena-code"]`.

**Step 3: Re-run syntax check**

```
python -c "import ast; ast.parse(open(r'C:\Users\gamer23\Desktop\hermes-agent\tools_mcp.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

**Step 4: Re-run manifest checker to verify no tools are skipped**

```
python check_mcp_names.py
```
Expected: all curated tool names appear in each server's output.

**Step 5: Delete the helper script**

```
del check_mcp_names.py
```

**Step 6: Commit**

```bash
git add tools_mcp.py
git commit -m "fix: update CURATED tool names and SERVER_CMDS from live MCP manifests"
```

---

### Task 4: Wire `tools_mcp` into `agent.py`

**Files:**
- Modify: `C:\Users\gamer23\Desktop\hermes-agent\agent.py`

**Step 1: Add import at top of agent.py**

After the existing imports (around line 14), add:
```python
# MCP skills bridge (MemPalace · Serena · Ruflo)
try:
    import tools_mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
```

**Step 2: Call register_mcp_tools after TOOL_HANDLERS**

After the closing `}` of `TOOL_HANDLERS` (around line 267), add:
```python
# ─── MCP skill registration ───────────────────────────────────────────────────
if MCP_AVAILABLE:
    tools_mcp.register_mcp_tools(TOOLS, TOOL_HANDLERS)
```

**Step 3: Expand SYSTEM_PROMPT with SKILLS section**

Find the line:
```
TOOL CALL RULES — follow these precisely
```

Insert the following block immediately before it:

```
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
- agent_search        — web/doc research via agent
- agent_analyze       — deep analysis of code or content
- agent_generate      — generate code or content via specialist
- agent_status        — check status of a running agent task

```

**Step 4: Expand /tools CLI command output**

Find the `tool_info` list in `handle_cli` (around line 341) and append:
```python
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
                ("agent_search",         "Web/doc search via agent"),
                ("agent_analyze",        "Analyze code or content"),
                ("agent_generate",       "Generate code or content"),
                ("agent_status",         "Check status of running agent"),
            ]
```

**Step 5: Syntax check**

```
python -c "import ast; ast.parse(open(r'C:\Users\gamer23\Desktop\hermes-agent\agent.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

**Step 6: Commit**

```bash
git add agent.py
git commit -m "feat: wire tools_mcp into agent.py — register MCP skills at startup"
```

---

### Task 5: Smoke test end-to-end

**Files:**
- No code changes — testing only

**Step 1: Run agent.py and watch stderr**

```
python agent.py
```
Expected stderr (before the prompt appears):
```
[tools_mcp] registered N MCP tools: ['memory_store', 'memory_search', ...]
```
If a server fails to start, it prints a warning and continues.

**Step 2: Test /tools shows MCP skills**

At the prompt, type:
```
/tools
```
Expected: the tool list now includes the MEMORY, CODE, and AGENTS sections.

**Step 3: Test memory_store round-trip**

Type:
```
remember that my preferred coding language is Python
```
Expected: hermes calls `memory_store` with a summary, confirms it was saved.

**Step 4: Test memory_search**

Type:
```
what do you know about my coding preferences?
```
Expected: hermes calls `memory_search`, finds the stored fact, reports it.

**Step 5: Test code_overview**

Type:
```
give me an overview of agent.py's functions
```
Expected: hermes calls `code_overview` on `agent.py`, lists the functions.

**Step 6: If any test fails**

- Check stderr for `[tools_mcp]` error lines
- If a specific tool name is wrong, go back to Task 3 and fix `CURATED`
- If a server won't start, verify the command from Task 1

**Step 7: Commit test confirmation**

```bash
git commit --allow-empty -m "test: smoke tested MCP bridge — memory + code tools verified"
```

---

### Task 6: Clean up and final commit

**Files:**
- No new files — cleanup only

**Step 1: Verify no debug prints leaked into agent.py**

Search for any `print(` statements added accidentally during Task 4:
```
python -c "
lines = open(r'C:\Users\gamer23\Desktop\hermes-agent\agent.py', encoding='utf-8').readlines()
new_prints = [(i+1, l.strip()) for i, l in enumerate(lines) if 'print(' in l and 'tools_mcp' not in l and i > 267]
print('Suspicious prints:', new_prints[:10])
"
```
Expected: the only new prints are in the `/tools` handler block (intentional).

**Step 2: Final syntax check on both files**

```
python -c "import ast; [ast.parse(open(r'C:\Users\gamer23\Desktop\hermes-agent\{}.py'.format(f), encoding='utf-8').read()) or print(f, 'OK') for f in ['agent', 'tools_mcp']]"
```
Expected: `agent OK` and `tools_mcp OK`

**Step 3: Final commit**

```bash
git add agent.py tools_mcp.py
git commit -m "feat: MCP skills bridge complete — MemPalace + Serena + Ruflo integrated into hermes"
```

---

## Dependency checklist

Before starting Task 2:
- [ ] `pip install mempalace` succeeded
- [ ] `mempalace init <path>` succeeded
- [ ] Serena MCP command confirmed from settings.json
- [ ] Node.js available (`node --version`)
- [ ] `npx ruflo@latest mcp start --help` works

# -*- coding: utf-8 -*-
"""
tools_mcp.py — MCP stdio bridge for hermes agent.
Manages MemPalace, Serena, and Ruflo MCP server subprocesses.
"""
import json
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

# ── Curated tool sets ────────────────────────────────────────────────────────
# Map: server_key → {mcp_tool_name: hermes_tool_name}
# All names confirmed from live manifests: MemPalace/Serena (Task 1), Ruflo (Task 3, claude-flow v3.0.0).

CURATED: Dict[str, Dict[str, str]] = {
    "mempalace": {
        "mempalace_add_drawer"    : "memory_store",
        "mempalace_search"        : "memory_search",
        "mempalace_list_drawers"  : "memory_list",
        "mempalace_delete_drawer" : "memory_delete",
        "mempalace_update_drawer" : "memory_update",
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
        "agent_spawn"       : "agent_task",
        "browser_open"      : "agent_search",
        "analyze_diff"      : "agent_analyze",
        "wasm_agent_prompt" : "agent_generate",
        "agent_status"      : "agent_status",
    },
}

# ── Server launch commands ───────────────────────────────────────────────────
# shutil.which() resolves the real executable path so subprocess finds it
# regardless of whether Windows .cmd shims or bare .exe files are used.
def _which(name: str) -> str:
    """Return the full path to `name`, trying common Windows suffixes if needed."""
    found = shutil.which(name)
    if found:
        return found
    for suffix in (".cmd", ".exe", ".bat"):
        found = shutil.which(name + suffix)
        if found:
            return found
    return name  # fall back to bare name; Popen will raise FileNotFoundError with a clear message

_NPX = _which("npx")
_UVX = _which("uvx")

SERVER_CMDS: Dict[str, List[str]] = {
    "mempalace": [sys.executable, "-m", "mempalace.mcp_server"],
    "serena":    [_UVX, "--from", "git+https://github.com/oraios/serena",
                  "serena", "start-mcp-server"],
    "ruflo":     [_NPX, "ruflo@latest", "mcp", "start"],
}

# Warmup times: slow starters (npx/uvx first cold-start clones the package)
_WARMUP: Dict[str, float] = {
    "mempalace": 1.0,
    "serena":    8.0,   # uvx clones repo on cold start; cached runs are ~2s
    "ruflo":     5.0,   # npx downloads ruflo@latest on first use
}

# ── McpServer class ──────────────────────────────────────────────────────────

class McpServer:
    """Manages a single MCP server subprocess (stdio JSON-RPC)."""

    TIMEOUT = 10.0  # seconds

    def __init__(self, key: str, cmd: list[str]):
        self.key   = key
        self.cmd   = cmd
        self._proc: Optional[subprocess.Popen] = None
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
            # Allow slow starters (npx, uvx) to warm up; first cold-start may download packages
            time.sleep(_WARMUP.get(self.key, 2.0))

            # ── Liveness check ────────────────────────────────────────────────
            # On Windows, writing to a pipe whose subprocess has already exited
            # raises [Errno 22] Invalid argument.  Detect early exit here and
            # capture stderr so the developer sees WHY the process died.
            rc = self._proc.poll()
            if rc is not None:
                try:
                    stderr_out = self._proc.stderr.read(2000).strip()
                except Exception:
                    stderr_out = ""
                msg = f"[tools_mcp] {self.key}: process exited during warmup (rc={rc})"
                if stderr_out:
                    msg += f"\n  stderr → {stderr_out}"
                print(msg, file=sys.stderr)
                self._proc = None
                return False
            # ─────────────────────────────────────────────────────────────────

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

    def _send(self, payload: dict) -> Optional[dict]:
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
            _srv   = server
            _mname = mcp_name
            handlers[hermes_name] = lambda args, s=_srv, n=_mname: s.call_tool(n, args)

    registered = [h for h in handlers if any(
        h in tool_map.values() for tool_map in CURATED.values()
    )]
    print(f"[tools_mcp] registered {len(registered)} MCP tools: {registered}",
          file=sys.stderr)

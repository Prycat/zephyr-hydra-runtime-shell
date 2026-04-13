# MCP Skills Bridge — Design

**Date:** 2026-04-13  
**Status:** Approved  

## Goal

Give the hermes agent (`agent.py`, Ollama `hermes3:8b`) access to a curated set of tools from three MCP servers — MemPalace (persistent memory), Serena (code intelligence), and Ruflo (agent orchestration) — without overwhelming the 8B model context.

---

## Architecture

```
zephyr_gui.py
    └── ZephyrProcess (QThread)
            └── agent.py  ← subprocess
                    ├── TOOLS[] + TOOL_HANDLERS{}   (existing)
                    └── tools_mcp.py                (new)
                            ├── McpServer("mempalace",  "python -m mempalace.mcp_server")
                            ├── McpServer("serena",     "<serena mcp start command>")
                            └── McpServer("ruflo",      "npx ruflo@latest mcp start")
```

A new file `tools_mcp.py` owns all MCP complexity. `agent.py` calls one function at startup — `register_mcp_tools(TOOLS, TOOL_HANDLERS)` — and the curated MCP tools become indistinguishable from native tools.

---

## `tools_mcp.py` internals

**`McpServer` class**
- Launches each server as a long-running subprocess (stdin/stdout JSON-RPC over newline-delimited JSON)
- Sends `initialize` + `tools/list` on first use (lazy start — no delay if unused in a session)
- Routes `tools/call` requests with 10 s timeout
- Attempts subprocess restart on crash; returns error string on timeout

**`CURATED` dict**
- Maps each server name → list of MCP tool names to expose
- Includes description overrides so hermes sees clean, consistent tool names

**`register_mcp_tools(tools_list, handlers)`**
- Iterates `CURATED`, fetches live tool schemas from each server's manifest
- Builds OpenAI-compatible tool definitions and injects them into the caller's `TOOLS[]` and `TOOL_HANDLERS{}`
- Silently skips any server that fails to start (stderr warning only), so the other two still register

---

## MCP protocol

Plain `subprocess` + `json` — no external MCP client library needed.

```
request:  {"jsonrpc":"2.0","id":N,"method":"tools/call","params":{"name":"...","arguments":{...}}}\n
response: {"jsonrpc":"2.0","id":N,"result":{"content":[{"type":"text","text":"..."}]}}\n
```

---

## Curated tool set (16 tools)

| Hermes tool name | MCP server | Purpose |
|---|---|---|
| `memory_store` | mempalace | Store a fact, summary, or preference |
| `memory_search` | mempalace | Semantic search across stored memories |
| `memory_list` | mempalace | List recent memories |
| `memory_delete` | mempalace | Remove a specific memory |
| `memory_update` | mempalace | Update an existing memory |
| `code_overview` | serena | List all symbols in a file |
| `code_find_symbol` | serena | Find a symbol by name across the codebase |
| `code_search` | serena | Regex/pattern search across files |
| `code_read` | serena | Read a file or specific line range |
| `code_references` | serena | Find all references to a symbol |
| `code_list_dir` | serena | List directory contents |
| `agent_task` | ruflo | Delegate a complex task to a specialized agent |
| `agent_search` | ruflo | Web/doc search via a Ruflo agent |
| `agent_analyze` | ruflo | Analyze code or content with a specialist |
| `agent_generate` | ruflo | Generate code or content with a specialist |
| `agent_status` | ruflo | Check status of a running agent task |

*Actual MCP tool names verified against each server's live `tools/list` manifest during implementation.*

---

## Changes to `agent.py`

Minimal — four additions only:

1. `import tools_mcp` at top
2. `tools_mcp.register_mcp_tools(TOOLS, TOOL_HANDLERS)` after `TOOL_HANDLERS = {...}`
3. System prompt gains a new **SKILLS** section describing the three capability groups
4. No other changes — all MCP logic lives in `tools_mcp.py`

---

## Error handling

| Scenario | Behaviour |
|---|---|
| Server binary not found (Node.js missing, package not installed) | Skip that server's tools; print warning to stderr; other servers unaffected |
| Tool call timeout (10 s) | Return `"[server] timeout — server may have crashed"` to hermes; attempt subprocess restart |
| Server crashes mid-session | Auto-restart on next tool call |
| MCP manifest missing a curated tool | Skip that tool silently; log to stderr |

---

## Prerequisites

| Dependency | Required for |
|---|---|
| `pip install mempalace` | MemPalace server |
| `mempalace init <path>` | MemPalace database init |
| Serena MCP server command (TBD during impl) | Serena server |
| Node.js + npm | Ruflo server |
| `npx ruflo@latest mcp start` | Ruflo server |

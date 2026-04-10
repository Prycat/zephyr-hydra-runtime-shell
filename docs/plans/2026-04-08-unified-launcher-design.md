# Unified Launcher Design
**Date:** 2026-04-08
**Project:** hermes-agent
**Status:** Approved

## Problem

Running `start_server.py` or `agent.py` by double-clicking on Windows causes the window to open and immediately close — the Python process exits and takes the window with it.

## Solution

A single `launch.bat` that:
1. Opens `start_server.py` in a **new persistent `cmd` window** titled "Hermes Server" (`/k` flag keeps it open)
2. Waits 10 seconds for vLLM to begin initializing
3. Runs `agent.py` interactively in the **launcher window itself**

## File

`C:/Users/gamer23/Desktop/hermes-agent/launch.bat`

## Behavior

- **Double-click `launch.bat`** → two windows appear:
  - "Hermes Server" window: shows vLLM startup logs + TurboQuant patch confirmation
  - Launcher window: becomes the chat interface after 10s
- **On agent exit** (`quit`/`exit`/Ctrl+C): launcher window closes; user is reminded to also close the server window
- **If server doesn't start in time**: `agent.py`'s existing health check will print a clear error

## Key Windows cmd Flags

| Flag | Purpose |
|------|---------|
| `start "Hermes Server" cmd /k ...` | Open new window, keep it open after command ends |
| `timeout /t 10` | Wait 10 seconds (interruptible with any key) |

## Out of Scope

- Auto-closing the server window when the agent exits (requires process tracking)
- Adjustable wait time via CLI args

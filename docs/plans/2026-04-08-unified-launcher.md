# Unified Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a single `launch.bat` that double-click launches both the vLLM server (in its own persistent window) and the agent (in the launcher window), solving the open-and-close problem on Windows.

**Architecture:** A Windows batch file uses `start "Hermes Server" cmd /k python start_server.py` to open the server in a separate persistent window, waits 10 seconds with `timeout /t 10`, then runs `python agent.py` interactively in the current window. Both windows stay open because `cmd /k` and interactive Python both block.

**Tech Stack:** Windows batch scripting (cmd.exe), existing `start_server.py` and `agent.py`

---

### Task 1: Create launch.bat

**Files:**
- Create: `C:/Users/gamer23/Desktop/hermes-agent/launch.bat`

**Step 1: Write the file**

```bat
@echo off
setlocal

echo === Hermes Agent Launcher ===
echo.

REM Verify start_server.py and agent.py exist before doing anything
if not exist "%~dp0start_server.py" (
    echo ERROR: start_server.py not found.
    echo Make sure you are running this from the hermes-agent folder.
    pause
    exit /b 1
)
if not exist "%~dp0agent.py" (
    echo ERROR: agent.py not found.
    echo Make sure you are running this from the hermes-agent folder.
    pause
    exit /b 1
)

REM Open vLLM server in a new persistent window
echo Starting vLLM + TurboQuant server in a new window...
start "Hermes Server" cmd /k "cd /d %~dp0 && python start_server.py"

REM Wait for server to begin initializing before launching agent
echo Waiting 10 seconds for server to initialize...
timeout /t 10 /nobreak

REM Run agent interactively in this window
echo.
echo Launching agent...
echo (Close the "Hermes Server" window when you are done.)
echo.
cd /d %~dp0
python agent.py

echo.
echo Agent exited. You can now close the "Hermes Server" window.
pause
```

**Key details:**
- `%~dp0` = the directory containing `launch.bat` — ensures paths work whether double-clicked from Explorer or run from any terminal directory
- `start "Hermes Server" cmd /k "..."` — opens a new titled window that stays open (`/k`) after the command ends
- `timeout /t 10 /nobreak` — waits 10 seconds without requiring a keypress (`/nobreak`)
- `pause` at the end keeps the launcher window open after agent exits so the user can read any final output

**Step 2: Verify file contents look correct**

Read `launch.bat` back and confirm:
- `%~dp0` is used for both existence checks and the `cd` commands
- `start "Hermes Server" cmd /k` is present
- `timeout /t 10 /nobreak` is present
- `pause` appears at the end

**Step 3: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add launch.bat docs/plans/2026-04-08-unified-launcher-design.md docs/plans/2026-04-08-unified-launcher.md
git commit -m "feat: add unified launch.bat for one-click startup"
```

Expected: commit succeeds, `git status` shows clean working tree.

---

### Task 2: Smoke Test

**Goal:** Verify the bat file is valid and the logic is correct without needing a running vLLM server.

**Step 1: Check bat syntax by dry-running the file-existence checks**

Run from the project directory:
```bash
cmd /c "cd C:/Users/gamer23/Desktop/hermes-agent && if exist start_server.py (echo OK: start_server.py found) else (echo MISSING)"
cmd /c "cd C:/Users/gamer23/Desktop/hermes-agent && if exist agent.py (echo OK: agent.py found) else (echo MISSING)"
```

Expected output: both print `OK: <file> found`

**Step 2: Confirm `%~dp0` will resolve correctly**

Run:
```bash
cmd /c "cd C:/ && echo %~dp0" 2>nul || echo "Note: ~dp0 only expands inside .bat files — this is expected"
```

This is expected to not expand outside a bat context — that's fine. Just confirms the syntax note is understood.

**Step 3: Read the final file one more time to do a human-readable logic check**

Read `C:/Users/gamer23/Desktop/hermes-agent/launch.bat` and confirm the flow reads as:
1. Check files exist → error+pause if missing
2. `start "Hermes Server" cmd /k "cd /d %~dp0 && python start_server.py"`
3. `timeout /t 10 /nobreak`
4. `python agent.py`
5. `pause`

Report: PASS or FAIL with details.

**Step 4: Verify it is committed**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent && git log --oneline -3
```

Expected: most recent commit is `feat: add unified launch.bat for one-click startup`

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

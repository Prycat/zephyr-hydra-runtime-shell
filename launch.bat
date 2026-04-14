@echo off
setlocal

echo === Zephyr // Prycat Research Team ===
echo.

REM Check agent.py exists
if not exist "%~dp0agent.py" (
    echo ERROR: agent.py not found.
    echo Make sure you are running this from the hermes-agent folder.
    pause
    exit /b 1
)

REM Check Ollama is running (quick connection test)
echo Checking Ollama is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama is not running. Starting it now...
    start "" ollama serve
    echo Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
)

echo Ollama is ready.
echo Model: hermes3:8b
echo.
echo Launching Zephyr...
echo (Press Ctrl+C or type 'exit' to stop)
echo.

cd /d %~dp0
python agent.py

echo.
pause

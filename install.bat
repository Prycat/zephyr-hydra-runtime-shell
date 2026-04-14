@echo off

REM ── Self-relaunch trick: if not already inside a persistent window, open one ─
REM    This ensures the window NEVER closes automatically, even on crash.
if "%HERMES_LAUNCHED%"=="1" goto :main
set HERMES_LAUNCHED=1
cmd /k "%~f0"
exit /b

:main
setlocal
echo === Hermes Agent Setup ===
echo.

REM Check Ollama is installed
echo [1/4] Checking Ollama...
ollama --version
if errorlevel 1 (
    echo.
    echo ERROR: Ollama not found. Download and install it from:
    echo   https://ollama.com/download
    echo After installing, re-run this script.
    goto :done
)
echo OK
echo.

REM Check Python is installed
echo [2/4] Checking Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python not found.
    echo Install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure "Add Python to PATH" is checked during install.
    goto :done
)
echo OK
echo.

REM Check / install Python packages
echo [3/4] Checking Python packages...
python -c "import ddgs" 2>nul
if errorlevel 1 (
    echo  - Installing ddgs (web search)...
    pip install ddgs
    if errorlevel 1 ( echo ERROR: pip install ddgs failed. & goto :done )
) else ( echo  - ddgs: already installed )

python -c "import bs4" 2>nul
if errorlevel 1 (
    echo  - Installing beautifulsoup4...
    pip install beautifulsoup4
    if errorlevel 1 ( echo ERROR: pip install beautifulsoup4 failed. & goto :done )
) else ( echo  - beautifulsoup4: already installed )

python -c "import openai" 2>nul
if errorlevel 1 (
    echo  - Installing openai...
    pip install openai
    if errorlevel 1 ( echo ERROR: pip install openai failed. & goto :done )
) else ( echo  - openai: already installed )

python -c "import httpx" 2>nul
if errorlevel 1 (
    echo  - Installing httpx...
    pip install httpx
    if errorlevel 1 ( echo ERROR: pip install httpx failed. & goto :done )
) else ( echo  - httpx: already installed )

REM AI Provider packages (for /call command)
python -c "import anthropic" 2>nul
if errorlevel 1 (
    echo  - Installing anthropic (Claude)...
    pip install anthropic
    if errorlevel 1 ( echo WARNING: anthropic install failed - Claude /call will not work. )
) else ( echo  - anthropic: already installed )

python -c "import google.generativeai" 2>nul
if errorlevel 1 (
    echo  - Installing google-generativeai (Gemini)...
    pip install google-generativeai
    if errorlevel 1 ( echo WARNING: google-generativeai install failed - Gemini /call will not work. )
) else ( echo  - google-generativeai: already installed )
echo OK
echo.

REM Pull the Hermes 3 model via Ollama
echo [4/4] Pulling hermes3:8b model...
echo      (4.7 GB on first run - this may take several minutes)
echo      (Already downloaded? This will finish immediately)
echo.
ollama pull hermes3:8b
if errorlevel 1 (
    echo.
    echo ERROR: Failed to pull hermes3:8b.
    echo  - Make sure Ollama is running (look for the llama icon in your system tray)
    echo  - If it is not running, launch Ollama from the Start menu first
    goto :done
)

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Model : hermes3:8b  (local, GPU-accelerated)
echo API   : http://localhost:11434/v1
echo.
echo Next step: double-click launch.bat to start the agent.
echo.

:done
echo.
echo (This window will stay open. Press any key to close it.)
pause >nul

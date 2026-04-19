@echo off
title Zephyr — Prycat Research
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11 from python.org
    pause
    exit /b 1
)

:: Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama doesn't appear to be running.
    echo Start Ollama first, then relaunch Zephyr.
    echo.
    pause
    exit /b 1
)

:: Launch GUI
echo Starting Zephyr...
python zephyr_gui.py

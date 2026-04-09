@echo off
setlocal
echo === TurboQuant + vLLM Setup ===

REM Check Python
python --version || (echo ERROR: Python not found && exit /b 1)

REM Check torch is installed
python -c "import torch" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch not installed. Install it with CUDA support first:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    exit /b 1
)

REM Check CUDA
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo ERROR: CUDA not available. vLLM requires an NVIDIA GPU with CUDA.
    echo Tip: Check your NVIDIA drivers and CUDA toolkit installation.
    echo      https://pytorch.org/get-started/locally/
    exit /b 1
)

REM Install vLLM 0.18.0 (TurboQuant targets this exact version)
echo Installing vLLM 0.18.0...
pip install vllm==0.18.0
if errorlevel 1 (
    echo ERROR: vLLM install failed. Try: pip install vllm==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121
    exit /b 1
)

REM Check git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found. Install Git for Windows: https://git-scm.com/download/win
    exit /b 1
)

REM Clone TurboQuant if not already present
if not exist turboquant (
    echo Cloning TurboQuant...
    git clone https://github.com/0xSero/turboquant turboquant || (echo ERROR: git clone failed && exit /b 1)
) else (
    echo TurboQuant already cloned, skipping.
)

REM Install TurboQuant
echo Installing TurboQuant...
pushd turboquant
pip install -e .
if errorlevel 1 (
    popd
    echo ERROR: TurboQuant install failed
    exit /b 1
)
popd

echo.
echo === Setup complete! ===
echo Next: python start_server.py

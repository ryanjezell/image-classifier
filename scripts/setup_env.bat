@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: scripts\setup_env.bat
::
:: One-shot environment setup for Windows.
:: Run from the project root directory.
::
:: Usage:
::   scripts\setup_env.bat
::   scripts\setup_env.bat --no-data
::   scripts\setup_env.bat --cpu-only
:: ─────────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

set VENV_DIR=.venv
set SKIP_DATA=false
set CPU_ONLY=false

:: Parse flags
for %%A in (%*) do (
    if "%%A"=="--no-data"   set SKIP_DATA=true
    if "%%A"=="--cpu-only"  set CPU_ONLY=true
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   IMAGE CLASSIFIER — Environment Setup (Windows)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

:: Check Python
where python >nul 2>&1 || (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    exit /b 1
)

:: Create venv
if exist %VENV_DIR% (
    echo   venv    : %VENV_DIR% already exists — skipping.
) else (
    echo   venv    : creating %VENV_DIR% ...
    python -m venv %VENV_DIR%
)

:: Activate
call %VENV_DIR%\Scripts\activate.bat
echo   venv    : activated

:: Upgrade pip
pip install --upgrade pip --quiet

:: PyTorch
if "%CPU_ONLY%"=="true" (
    echo   torch   : installing CPU-only build ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    echo   torch   : installing ^(GPU if available^) ...
    pip install torch torchvision --quiet
)

:: Requirements
echo   deps    : installing requirements.txt ...
pip install -r requirements.txt --quiet
echo   [OK] All packages installed.

:: Directories
if not exist data\dataset mkdir data\dataset
if not exist models\exported mkdir models\exported

:: Sample data
if "%SKIP_DATA%"=="true" (
    echo   data    : skipped ^(--no-data flag^).
) else (
    echo   data    : downloading sample dataset ...
    python scripts\download_sample_data.py
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Setup complete!
echo.
echo   Activate environment:
echo     .venv\Scripts\activate
echo.
echo   Train:
echo     python train.py
echo     python train.py --lr-finder
echo.
echo   Predict:
echo     python predict.py --image path\to\image.jpg
echo.
echo   Evaluate:
echo     python evaluation.py
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

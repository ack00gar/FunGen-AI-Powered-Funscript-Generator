@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo                    FunGen Smart Launcher
echo                        v1.0.0
echo ================================================================
echo Automatically detects your Python environment setup
echo.

cd /d "%~dp0"

REM Check for local venv folder first (original installer)
if exist "venv\Scripts\activate.bat" (
    echo ✓ Found local virtual environment (venv)
    echo   Activating: %~dp0venv\Scripts\activate.bat
    call "venv\Scripts\activate.bat"
    goto :run_fungen
)

REM Check for conda environments
set "CONDA_BASE_PATH="
set "CONDA_EXE="

REM Try common conda installation paths
for %%P in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3" 
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if exist "%%P\Scripts\conda.exe" (
        set "CONDA_BASE_PATH=%%P"
        set "CONDA_EXE=%%P\Scripts\conda.exe"
        goto :found_conda
    )
)

:found_conda
if defined CONDA_EXE (
    echo ✓ Found conda installation at: !CONDA_BASE_PATH!
    
    REM Try different environment names in order of preference
    for %%E in ("fungen" "VRFunAIGen" "FunGen") do (
        echo   Checking for conda environment: %%E
        "!CONDA_EXE!" info --envs | findstr /C:%%E >nul 2>&1
        if !errorlevel! equ 0 (
            echo ✓ Found conda environment: %%E
            echo   Activating conda environment...
            call "!CONDA_BASE_PATH!\Scripts\activate.bat" %%E
            goto :run_fungen
        )
    )
    
    echo ⚠ No FunGen conda environments found
    echo   Available environments:
    "!CONDA_EXE!" info --envs
    echo.
    echo   Please create a conda environment using:
    echo   conda create -n fungen python=3.11 -y
    echo   Or run: fungen_enhanced_install.bat
    goto :no_environment
)

:no_environment
echo ✗ No Python environment found!
echo.
echo SOLUTIONS:
echo 1. Run original installer: fungen_install.bat
echo 2. Run enhanced installer: fungen_enhanced_install.bat  
echo 3. Fix Python version: fix_python_version.bat
echo.
pause
exit /b 1

:run_fungen
echo.
echo ================================================================
echo                     Starting FunGen...
echo ================================================================
echo.

REM Check if main.py exists
if not exist "main.py" (
    echo ✗ main.py not found in current directory!
    echo   Current directory: %CD%
    echo   Please navigate to the FunGen project directory.
    pause
    exit /b 1
)

REM Find actual Python executable (avoid Microsoft Store redirect)
set "PYTHON_CMD=python"

REM If in venv, use venv python directly
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"
)

REM Check Python availability
"!PYTHON_CMD!" --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ✗ Python not available in current environment
    echo   Microsoft Store redirect detected - use FunGen_Windows_Fix.bat
    pause
    exit /b 1
)

REM Display Python info
echo Python environment info:
"!PYTHON_CMD!" --version
echo Python path: 
"!PYTHON_CMD!" -c "import sys; print(sys.executable)"
echo.

REM Launch FunGen
"!PYTHON_CMD!" main.py %*

REM Keep window open if there was an error
if !errorlevel! neq 0 (
    echo.
    echo ✗ FunGen exited with error code: !errorlevel!
    pause
)
@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo              FunGen Windows Python Fix
echo                     v1.0.0
echo ================================================================
echo Fixing "Python was not found" Microsoft Store redirect issue
echo.

cd /d "%~dp0"

REM Function to find real Python executable
set "PYTHON_EXE="

echo [1/4] Looking for Python installations...

REM Check local venv first
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    echo ✓ Found Python in local venv: !PYTHON_EXE!
    goto :activate_venv
)

REM Check common Python installation paths
for %%P in (
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"  
    "C:\Python39\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) do (
    if exist %%P (
        set "PYTHON_EXE=%%P"
        echo ✓ Found Python installation: !PYTHON_EXE!
        goto :check_conda
    )
)

REM Check conda installations
:check_conda
for %%C in (
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%LOCALAPPDATA%\miniconda3\python.exe" 
    "%LOCALAPPDATA%\anaconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
) do (
    if exist %%C (
        set "CONDA_BASE=%%~dpC"
        set "PYTHON_EXE=%%C"
        echo ✓ Found conda Python: !PYTHON_EXE!
        goto :activate_conda
    )
)

if not defined PYTHON_EXE (
    echo ✗ No Python installation found!
    echo.
    echo SOLUTIONS:
    echo 1. Install Python 3.11 from python.org
    echo 2. Run fungen_enhanced_install.bat (installs Miniconda)
    echo 3. Disable Python app aliases in Windows Settings:
    echo    Settings → Apps → App execution aliases
    echo    Turn OFF python.exe and python3.exe aliases
    echo.
    pause
    exit /b 1
)

:activate_venv
echo.
echo [2/4] Activating virtual environment...
call "venv\Scripts\activate.bat"
goto :run_python

:activate_conda
echo.
echo [2/4] Checking conda environments...
set "CONDA_ACTIVATE=!CONDA_BASE!Scripts\activate.bat"

if exist "!CONDA_ACTIVATE!" (
    for %%E in (fungen VRFunAIGen FunGen) do (
        echo Trying conda environment: %%E
        call "!CONDA_ACTIVATE!" %%E 2>nul
        if !errorlevel! equ 0 (
            echo ✓ Activated conda environment: %%E
            goto :run_python
        )
    )
)

REM If no conda env found, use base conda python
echo Using base conda Python installation
goto :run_python

:run_python
echo.
echo [3/4] Verifying Python...
"!PYTHON_EXE!" --version
if !errorlevel! neq 0 (
    echo ✗ Python verification failed
    pause
    exit /b 1
)

echo ✓ Python verified successfully
echo.

echo [4/4] Starting FunGen...
if not exist "main.py" (
    echo ✗ main.py not found in: %CD%
    pause
    exit /b 1
)

"!PYTHON_EXE!" main.py %*

if !errorlevel! neq 0 (
    echo.
    echo ✗ FunGen exited with error code: !errorlevel!
    pause
)
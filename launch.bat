@echo off
REM FunGen launcher (Windows). Self-heals: if .venv is missing or broken,
REM bootstraps uv and runs install.py before launching.
setlocal
cd /d "%~dp0"

set PYTHONNOUSERSITE=1
set KMP_DUPLICATE_LIB_OK=TRUE
set YOLO_TELEMETRY=False
set YOLO_OFFLINE=True
set YOLO_CONFIG_DIR=%cd%\config\ultralytics

REM Drop any active conda env vars so nothing leaks into our venv interpreter
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
set CONDA_PROMPT_MODIFIER=
set CONDA_SHLVL=

set VENV_PY=.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo FunGen environment missing, running installer ^(one-time, ~2 min^)...

    REM Bootstrap uv if needed. We never trust system Python on Windows
    REM because the Microsoft Store python alias makes "where python" lie.
    where uv >nul 2>nul
    if errorlevel 1 (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    )

    where uv >nul 2>nul
    if errorlevel 1 (
        echo.
        echo Could not install uv. Run install.bat manually and report the issue.
        pause
        exit /b 1
    )

    uv run --no-project --isolated --python 3.11 -- python install.py
)

if not exist "%VENV_PY%" (
    echo.
    echo Install failed. See output above. Run install.bat and report the issue.
    pause
    exit /b 1
)

"%VENV_PY%" main.py %*
if errorlevel 1 pause

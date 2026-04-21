@echo off
REM FunGen launcher (Windows). Self-heals: if .venv is missing or broken,
REM runs install.py before launching.
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
    echo FunGen environment missing -- running installer ^(one-time, ~2 min^)...
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        python install.py
    ) else (
        where py >nul 2>nul
        if %errorlevel% equ 0 (
            py install.py
        ) else (
            echo Python is required but was not found on PATH.
            echo Install Python from https://www.python.org/ and re-run launch.bat.
            pause
            exit /b 1
        )
    )
)

if not exist "%VENV_PY%" (
    echo.
    echo Install failed. See output above. Run install.bat and report the issue.
    pause
    exit /b 1
)

"%VENV_PY%" main.py %*
if %errorlevel% neq 0 pause

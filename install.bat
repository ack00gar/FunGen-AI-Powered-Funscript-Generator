@echo off
REM FunGen installer (Windows). Bootstraps uv, then runs the latest install.py
REM directly from GitHub. install.py handles cloning the repo + building .venv.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing uv ^(one-time, ~15 MB^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

uv run --no-project --python 3.11 https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.py %*

pause

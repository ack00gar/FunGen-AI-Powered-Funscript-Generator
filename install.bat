@echo off
REM Thin shim: run the real installer (install.py) with a system Python.
REM install.py uses uv to manage a Python 3.11 + the FunGen .venv.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% equ 0 (
    python install.py %*
    goto :done
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py install.py %*
    goto :done
)

echo Python is required but was not found on PATH.
echo Install Python 3.11+ from https://www.python.org/ and re-run install.bat.
echo Make sure to check "Add python.exe to PATH" during installation.
pause
exit /b 1

:done
pause

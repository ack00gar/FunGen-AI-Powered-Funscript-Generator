@echo off
REM FunGen installer (Windows). Bootstraps uv, then uses uv to provide
REM Python 3.11 and run install.py.
REM
REM We never trust system Python on Windows because the Microsoft Store
REM alias at %LOCALAPPDATA%\Microsoft\WindowsApps\python.exe makes
REM "where python" return success even when Python is not actually
REM installed. Invoking the alias prints "Python was not found; run
REM without arguments to install from the Microsoft Store..." and exits.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %errorlevel% equ 0 goto :have_uv

echo Installing uv (one-time, ~15 MB download from astral.sh)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

REM uv installer adds itself to PATH for new shells; add common locations
REM to PATH for THIS session so the lines below find uv without restart.
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo uv install failed. Possible causes:
    echo   - astral.sh blocked by firewall
    echo   - PowerShell execution policy too strict
    echo Workaround: install uv manually from https://astral.sh/uv,
    echo open a fresh Command Prompt, then re-run install.bat.
    pause
    exit /b 1
)

:have_uv
echo Setting up FunGen environment via uv...
REM The "-- python install.py" form is required: on Windows, uv treats a bare
REM "install.py" arg as a program name and fails with "Failed to spawn:
REM install.py: program not found". Routing through "python" inside the
REM ephemeral env makes the script-vs-program ambiguity moot.
uv run --no-project --isolated --python 3.11 -- python install.py %*
set RC=%errorlevel%

pause
exit /b %RC%

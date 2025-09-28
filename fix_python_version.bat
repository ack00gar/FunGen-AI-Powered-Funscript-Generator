@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo            FunGen Python Version Fix Script
echo ================================================================
echo.
echo This script will:
echo 1. Download Python 3.11.9 (if not found)
echo 2. Remove the existing Python 3.13 virtual environment
echo 3. Create a new virtual environment with Python 3.11
echo 4. Re-run the FunGen installation
echo.

pause

REM Set paths
set "PYTHON311_PATH=C:\Python311\python.exe"
set "FUNGEN_DIR=%~dp0"
set "VENV_DIR=%FUNGEN_DIR%FunGen\venv"
set "INSTALLER_PATH=%FUNGEN_DIR%fungen_install.bat"

echo [1/5] Checking for Python 3.11...
if exist "%PYTHON311_PATH%" (
    echo ✓ Python 3.11 found at %PYTHON311_PATH%
) else (
    echo ✗ Python 3.11 not found
    echo.
    echo Please download Python 3.11.9 from:
    echo https://www.python.org/downloads/release/python-3119/
    echo.
    echo Download "Windows installer (64-bit)" and install with these options:
    echo - ✓ Add Python to PATH
    echo - ✓ Install for all users (recommended)
    echo - Install to: C:\Python311\
    echo.
    echo After installation, run this script again.
    pause
    exit /b 1
)

echo.
echo [2/5] Checking existing virtual environment...
if exist "%VENV_DIR%" (
    echo ✓ Found existing venv at: %VENV_DIR%
    echo Removing old virtual environment...
    rmdir /s /q "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo ✗ Failed to remove old venv. Please close any programs using it.
        pause
        exit /b 1
    )
    echo ✓ Old virtual environment removed
) else (
    echo ✓ No existing venv found
)

echo.
echo [3/5] Creating new virtual environment with Python 3.11...
"%PYTHON311_PATH%" -m venv "%VENV_DIR%"
if !errorlevel! neq 0 (
    echo ✗ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created successfully

echo.
echo [4/5] Verifying Python version in new environment...
"%VENV_DIR%\Scripts\python.exe" --version
if !errorlevel! neq 0 (
    echo ✗ Failed to verify Python version
    pause
    exit /b 1
)

echo.
echo [5/5] Re-running FunGen installation...
if exist "%INSTALLER_PATH%" (
    echo Running FunGen installer...
    call "%INSTALLER_PATH%"
) else (
    echo ✗ FunGen installer not found at: %INSTALLER_PATH%
    echo.
    echo Manual installation steps:
    echo 1. Navigate to: %FUNGEN_DIR%FunGen
    echo 2. Activate venv: venv\Scripts\activate.bat
    echo 3. Install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ================================================================
echo                    Installation Complete!
echo ================================================================
echo.
echo Python 3.11 virtual environment is now ready.
echo You can now run FunGen without imgui compilation errors.
echo.
pause
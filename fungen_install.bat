@echo off
REM FunGen Universal Bootstrap Installer for Windows
REM This script requires ZERO dependencies - only uses Windows built-ins
REM Downloads and runs the full Python installer

setlocal EnableDelayedExpansion

REM Check for help or common invalid flags
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-u" goto :invalid_flag
if "%1"=="/?" goto :show_help

echo ==========================================
echo      FunGen Bootstrap Installer
echo ==========================================
echo.
echo This installer will download and install everything needed:
echo   - Python 3.11 (Miniconda)
echo   - Git
echo   - FFmpeg/FFprobe
echo   - FunGen AI and all dependencies
echo.
echo Note: For best results, right-click and "Run as administrator"
echo       Some installations may require administrator privileges.
echo.

REM Check if we're running as administrator (needed for some installations)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Not running as administrator.
    echo Some installations may require admin privileges.
    echo Consider right-clicking and "Run as administrator"
    echo.
    pause
)

REM Create temp directory for downloads
set TEMP_DIR=%TEMP%\FunGenInstaller_%RANDOM%
mkdir "%TEMP_DIR%" 2>nul

REM Configuration
set INSTALLER_URL=https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/fungen_universal_installer.py
set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=%TEMP_DIR%\python-installer.exe
set UNIVERSAL_INSTALLER=%TEMP_DIR%\fungen_universal_installer.py
set MINICONDA_INSTALL_PATH=%USERPROFILE%\miniconda3

echo [1/4] Checking Python installation...
REM Check for Python in various locations
python --version >nul 2>&1
if %errorLevel% equ 0 (
    echo    Python already available in PATH, skipping download...
    set PYTHON_ALREADY_INSTALLED=1
) else if exist "%MINICONDA_INSTALL_PATH%" (
    echo    Miniconda already installed, skipping download...
    set PYTHON_ALREADY_INSTALLED=1
) else (
    echo    Downloading Python installer...
    REM Use PowerShell to download (available on all modern Windows)
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'}"
    if %errorLevel% neq 0 (
        echo ERROR: Failed to download Python installer
        goto :cleanup_and_exit
    )
    echo    Python installer downloaded successfully
)

echo.
echo [2/4] Installing Python 3.11...
if defined PYTHON_ALREADY_INSTALLED (
    echo    Python already installed, using existing installation...
    REM Ensure PATH includes Python
    call :refresh_path
) else (
    echo    This may take a few minutes...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=0
    if %errorLevel% neq 0 (
        echo ERROR: Python installation failed
        goto :cleanup_and_exit
    )
    
    REM Refresh PATH to include Python
    call :refresh_path
    echo    Python 3.11 installed successfully
)

echo.
echo [3/4] Downloading FunGen universal installer...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%UNIVERSAL_INSTALLER%'}"
if %errorLevel% neq 0 (
    echo ERROR: Failed to download universal installer
    goto :cleanup_and_exit
)
echo    Universal installer downloaded successfully

echo.
echo [4/4] Running FunGen universal installer...
echo    The universal installer will now handle the complete setup...
echo.

REM Pass through any command line arguments to the universal installer
if "%*"=="" (
    python "%UNIVERSAL_INSTALLER%" --dir "%CD%"
) else (
    echo    Passing arguments: %*
    python "%UNIVERSAL_INSTALLER%" --dir "%CD%" %*
)
set INSTALL_RESULT=%errorLevel%

:cleanup_and_exit
echo.
echo Cleaning up temporary files...
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"

if %INSTALL_RESULT% equ 0 (
    echo.
    echo ==========================================
    echo     Bootstrap Installation Complete!
    echo ==========================================
    echo.
    echo FunGen has been successfully installed.
    echo Check above for launcher instructions.
) else (
    echo.
    echo ==========================================
    echo       Installation Failed
    echo ==========================================
    echo.
    echo Please check the error messages above.
    echo You may need to run as administrator.
)

echo.
pause
exit /b %INSTALL_RESULT%

REM Function to refresh PATH environment variable
:refresh_path
for /f "delims=" %%i in ('powershell -Command "[Environment]::GetEnvironmentVariable('PATH', 'User')"') do set USER_PATH=%%i
for /f "delims=" %%i in ('powershell -Command "[Environment]::GetEnvironmentVariable('PATH', 'Machine')"') do set MACHINE_PATH=%%i
set PATH=%MACHINE_PATH%;%USER_PATH%
goto :eof

:show_help
echo FunGen Bootstrap Installer for Windows
echo Usage: %0 [options]
echo.
echo This script downloads and installs FunGen automatically.
echo All options are passed to the universal installer.
echo.
echo Common options:
echo   --force     Force reinstallation
echo   --uninstall Run uninstaller instead
echo   --help      Show this help
echo   /?, -h      Show this help
echo.
pause
exit /b 0

:invalid_flag
echo ERROR: '-u' is not a valid option.
echo Did you mean '--uninstall' or '--force'?
echo Run '%0 --help' for available options.
echo.
pause
exit /b 1
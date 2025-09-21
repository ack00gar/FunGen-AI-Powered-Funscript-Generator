@echo off
REM FunGen Uninstaller for Windows
REM Downloads and runs the universal uninstaller

echo ==========================================
echo      FunGen Uninstaller for Windows
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found. Cannot run uninstaller.
    echo Please install Python or use manual uninstall instructions.
    echo.
    pause
    exit /b 1
)

REM Try to find the uninstaller in current directory first
if exist "fungen_uninstall.py" (
    echo Found uninstaller in current directory.
    python fungen_uninstall.py %*
    goto :end
)

REM Download uninstaller from GitHub
echo Downloading FunGen uninstaller...
set UNINSTALLER_URL=https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/fungen_uninstall.py
set TEMP_DIR=%TEMP%\FunGenUninstaller_%RANDOM%
mkdir "%TEMP_DIR%" 2>nul

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%UNINSTALLER_URL%' -OutFile '%TEMP_DIR%\fungen_uninstall.py'}"

if %errorLevel% neq 0 (
    echo ERROR: Failed to download uninstaller
    echo Please download fungen_uninstall.py manually from GitHub
    rmdir /s /q "%TEMP_DIR%" 2>nul
    pause
    exit /b 1
)

echo Running uninstaller...
python "%TEMP_DIR%\fungen_uninstall.py" %*

REM Cleanup
rmdir /s /q "%TEMP_DIR%" 2>nul

:end
echo.
pause
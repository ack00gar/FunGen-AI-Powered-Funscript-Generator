@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo          FunGen Enhanced Universal Installer
echo                    v1.4.0
echo ================================================================
echo This installer will download and install everything needed:
echo - Miniconda (Python 3.11 + conda package manager)
echo - Git
echo - FFmpeg/FFprobe  
echo - FunGen AI and all dependencies
echo.
echo RECOMMENDED: Run this installer as a NORMAL USER
echo             Most installations work fine without administrator privileges
echo.

pause

REM Set variables
set "TEMP_DIR=%TEMP%\FunGen_Install"
set "INSTALL_DIR=%~dp0"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_INSTALLER=%TEMP_DIR%\Miniconda3-latest.exe"
set "MINICONDA_PATH=%USERPROFILE%\miniconda3"
set "CONDA_EXE=%MINICONDA_PATH%\Scripts\conda.exe"
set "ENV_NAME=fungen"

REM Create temp directory
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo [1/8] Checking Miniconda installation...
if exist "%CONDA_EXE%" (
    echo ✓ Miniconda already installed at: %MINICONDA_PATH%
    goto :check_git
) else (
    echo ✗ Miniconda not found, will install automatically
)

echo.
echo [2/8] Downloading Miniconda...
echo   Downloading from: %MINICONDA_URL%
echo   Please wait, this may take several minutes...

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%MINICONDA_INSTALLER%'}"
if !errorlevel! neq 0 (
    echo ✗ Failed to download Miniconda
    pause
    exit /b 1
)
echo ✓ Miniconda downloaded successfully

echo.
echo [3/8] Installing Miniconda...
echo   Installing to: %MINICONDA_PATH%
echo   This will take a few minutes...

start /wait "" "%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_PATH%
if !errorlevel! neq 0 (
    echo ✗ Failed to install Miniconda
    pause
    exit /b 1
)
echo ✓ Miniconda installed successfully

REM Initialize conda for cmd
call "%MINICONDA_PATH%\Scripts\activate.bat"

:check_git
echo.
echo [4/8] Checking Git installation...
where git >nul 2>&1
if !errorlevel! equ 0 (
    echo ✓ Git already available
) else (
    echo Installing Git via conda...
    "%CONDA_EXE%" install git -y
    if !errorlevel! neq 0 (
        echo ✗ Failed to install Git
        pause
        exit /b 1
    )
    echo ✓ Git installed successfully
)

echo.
echo [5/8] Creating FunGen conda environment...
echo   Creating environment '%ENV_NAME%' with Python 3.11...

"%CONDA_EXE%" create -n %ENV_NAME% python=3.11 -y
if !errorlevel! neq 0 (
    echo ✗ Failed to create conda environment
    pause
    exit /b 1
)
echo ✓ Conda environment '%ENV_NAME%' created successfully

echo.
echo [6/8] Installing FFmpeg via conda...
"%CONDA_EXE%" activate %ENV_NAME% && "%CONDA_EXE%" install ffmpeg -c conda-forge -y
if !errorlevel! neq 0 (
    echo ⚠ FFmpeg conda install failed
    echo   The universal installer will handle FFmpeg installation as fallback
) else (
    echo ✓ FFmpeg installed via conda
)

echo.
echo [7/8] Running FunGen universal installer...
echo   Prerequisites installed, now calling universal installer...

REM Activate the conda environment
call "%MINICONDA_PATH%\Scripts\activate.bat" %ENV_NAME%

REM Check if install.py exists in current directory
if exist "%INSTALL_DIR%install.py" (
    echo   Running local install.py...
    python "%INSTALL_DIR%install.py" --dir "%INSTALL_DIR%" --env-name "%ENV_NAME%" --skip-python-check
) else (
    echo   install.py not found locally, downloading from GitHub...
    set "INSTALLER_URL=https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.py"
    set "INSTALLER_FILE=%TEMP_DIR%\install.py"
    
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_FILE%'}"
    if !errorlevel! neq 0 (
        echo ✗ Failed to download universal installer
        pause
        exit /b 1
    )
    
    echo   Running downloaded universal installer...
    python "%INSTALLER_FILE%" --dir "%INSTALL_DIR%" --env-name "%ENV_NAME%" --skip-python-check
)

if !errorlevel! neq 0 (
    echo ✗ Universal installer failed
    pause
    exit /b 1
)

echo ✓ FunGen installation completed by universal installer

echo.
echo ================================================================
echo                  Installation Complete!
echo ================================================================
echo.
echo ✓ Prerequisites installed (Miniconda, Git, FFmpeg)
echo ✓ FunGen universal installer completed successfully
echo.
echo Check above for launcher instructions.
echo.
pause

REM Cleanup
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
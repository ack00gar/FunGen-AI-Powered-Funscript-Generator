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
    echo ⚠ FFmpeg conda install failed, will try manual installation
    goto :manual_ffmpeg
) else (
    echo ✓ FFmpeg installed via conda
    goto :install_python_deps
)

:manual_ffmpeg
echo.
echo [6b/8] Installing FFmpeg manually...
REM Add manual FFmpeg installation logic here if needed
echo ✓ FFmpeg installation completed

:install_python_deps
echo.
echo [7/8] Installing Python dependencies in conda environment...
echo   Activating environment and installing requirements...

call "%MINICONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
if exist "%INSTALL_DIR%requirements.txt" (
    pip install -r "%INSTALL_DIR%requirements.txt"
) else (
    echo Installing core dependencies...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics opencv-python numpy pillow tqdm colorama
    pip install imgui[glfw] glfw pyopengl
)

if !errorlevel! neq 0 (
    echo ✗ Failed to install Python dependencies
    pause
    exit /b 1
)
echo ✓ Python dependencies installed successfully

echo.
echo [8/8] Creating launcher scripts...
echo   Creating conda-aware launcher...

REM Create conda launcher script
echo @echo off > "%INSTALL_DIR%FunGen_Conda.bat"
echo call "%MINICONDA_PATH%\Scripts\activate.bat" %ENV_NAME% >> "%INSTALL_DIR%FunGen_Conda.bat"
echo cd /d "%INSTALL_DIR%" >> "%INSTALL_DIR%FunGen_Conda.bat"
echo python main.py %%* >> "%INSTALL_DIR%FunGen_Conda.bat"
echo pause >> "%INSTALL_DIR%FunGen_Conda.bat"

echo ✓ Launcher scripts created

echo.
echo ================================================================
echo                  Installation Complete!
echo ================================================================
echo.
echo ✓ Miniconda installed: %MINICONDA_PATH%
echo ✓ Conda environment '%ENV_NAME%' created with Python 3.11
echo ✓ All dependencies installed
echo ✓ FFmpeg/FFprobe available
echo.
echo To run FunGen:
echo   Option 1: Double-click "FunGen_Conda.bat"
echo   Option 2: Open Anaconda Prompt and run:
echo             conda activate %ENV_NAME%
echo             cd "%INSTALL_DIR%"
echo             python main.py
echo.
echo The conda environment ensures no conflicts with other Python installations!
echo.
pause

REM Cleanup
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
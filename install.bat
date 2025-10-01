@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo          FunGen Enhanced Universal Installer
echo                    v1.4.4 - ACTIVATION FIXES
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
echo CRITICAL FIXES IN v1.4.4:
echo - Fixed "Run conda init before conda activate" error
echo - Fixed PowerShell URL variable expansion issue  
echo - Use direct conda -n flag to avoid activation errors
echo - Improved fallback for FFmpeg installation
echo.

echo [0.1/8] Checking system architecture...
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    echo WARNING: ARM64 Windows detected
    echo    Some packages may not compile on ARM64.
    echo.
    echo    RECOMMENDED: Install x64 Python instead
    echo    - Download Python 3.11 x64 from python.org
    echo    - x64 Python runs via emulation and has better compatibility
    echo.
    echo    ALTERNATIVE: Use Windows Subsystem for Linux
    echo    - Run: wsl --install
    echo.
    set /p response="Continue anyway? [y/N]: "
    if /i "!response!" neq "y" (
        echo Installation cancelled.
        pause
        exit /b 1
    )
)

pause

REM Set variables
set "TEMP_DIR=%TEMP%\FunGen_Install"
set "INSTALL_DIR=%~dp0"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_INSTALLER=%TEMP_DIR%\Miniconda3-latest.exe"
set "MINICONDA_PATH=%USERPROFILE%\miniconda3"
set "CONDA_EXE=%MINICONDA_PATH%\Scripts\conda.exe"
set "ENV_NAME=fungen"

REM Note: ARM64 Windows should use x86_64 version via emulation
REM Miniconda does not provide native ARM64 builds for Windows

REM Create temp directory
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo [1/8] Checking Miniconda installation...
if exist "%CONDA_EXE%" (
    echo [OK] Miniconda already installed at: %MINICONDA_PATH%
    goto :init_conda
) else (
    echo [X] Miniconda not found, will install automatically
)

echo.
echo [2/8] Downloading Miniconda...
echo   Downloading from: %MINICONDA_URL%
echo   Please wait, this may take several minutes...

powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%MINICONDA_INSTALLER%' -UseBasicParsing"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to download Miniconda
    pause
    exit /b 1
)
echo [OK] Miniconda downloaded successfully

echo.
echo [3/8] Installing Miniconda...
echo   Installing to: %MINICONDA_PATH%
echo   This will take a few minutes...

start /wait "" "%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_PATH%
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install Miniconda
    pause
    exit /b 1
)
echo [OK] Miniconda installed successfully

:init_conda
echo.
echo [3.5/8] Initializing conda...
echo   Setting up conda for command line use...

REM Initialize conda for cmd - CRITICAL FIX
echo   Initializing conda for command line use...
call "%MINICONDA_PATH%\Scripts\conda.exe" init cmd.exe
if !errorlevel! neq 0 (
    echo [WARNING] Conda init failed, trying alternative method...
    REM Alternative: Add conda to PATH for this session
    set "PATH=%MINICONDA_PATH%\Scripts;%MINICONDA_PATH%;%PATH%"

    REM Try to initialize conda again after PATH fix
    "%MINICONDA_PATH%\Scripts\conda.exe" init cmd.exe >nul 2>&1
    if !errorlevel! neq 0 (
        echo [WARNING] Alternative conda init also failed, but continuing...
        echo   Will use PATH-based conda access
    ) else (
        echo [OK] Conda initialized successfully via alternative method
    )
) else (
    echo [OK] Conda initialized successfully
)

REM Restart the command prompt environment to pick up conda init changes
echo   Refreshing environment variables...
call "%MINICONDA_PATH%\Scripts\activate.bat"

REM Accept Terms of Service for conda channels (handles the TOS error)
echo   Accepting conda Terms of Service...
"%CONDA_EXE%" config --set channel_priority disabled >nul 2>&1
"%CONDA_EXE%" config --add channels conda-forge >nul 2>&1

REM Try to accept TOS, but don't fail if it doesn't work
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1
echo [OK] Conda configuration updated

echo.
echo [4/8] Creating FunGen conda environment...
echo   Creating environment '%ENV_NAME%' with Python 3.11...

REM Create environment with explicit channel to avoid TOS issues
"%CONDA_EXE%" create -n %ENV_NAME% python=3.11 -c conda-forge -y
if !errorlevel! neq 0 (
    echo [WARNING] Failed to create conda environment with conda-forge
    echo   Trying with default channels...
    "%CONDA_EXE%" create -n %ENV_NAME% python=3.11 -y --override-channels --channel defaults
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create conda environment
        echo   This may be due to conda Terms of Service issues
        echo   Try running these commands manually:
        echo   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
        echo   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
        pause
        exit /b 1
    )
)
echo [OK] Conda environment '%ENV_NAME%' created successfully

echo.
echo [5/8] Installing Git and FFmpeg via conda...

REM Install git and ffmpeg into the fungen environment
echo   Installing Git...
"%CONDA_EXE%" install -n %ENV_NAME% git -c conda-forge -y >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Git installed
) else (
    echo [WARNING] Git install failed - universal installer will handle it
)

echo   Installing FFmpeg...
"%CONDA_EXE%" install -n %ENV_NAME% ffmpeg -c conda-forge -y >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] FFmpeg installed
) else (
    echo [WARNING] FFmpeg install failed - universal installer will handle it
)

echo.
echo [6/8] Running FunGen universal installer...
echo   Prerequisites installed, now calling universal installer...

REM Initialize conda environment variables properly
call "%MINICONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
if !errorlevel! neq 0 (
    echo [WARNING] Environment activation failed, trying alternative method...
    REM Alternative activation method
    set "PATH=%MINICONDA_PATH%\envs\%ENV_NAME%\Scripts;%MINICONDA_PATH%\envs\%ENV_NAME%;%PATH%"
    set "CONDA_DEFAULT_ENV=%ENV_NAME%"
)

REM Remove trailing backslash from INSTALL_DIR to avoid quote escaping issues
REM BUT: Keep it if we're at drive root (C:\, D:\, etc.)
set "INSTALL_DIR_CLEAN=%INSTALL_DIR%"
if "%INSTALL_DIR:~-1%"=="\" (
    REM Has trailing backslash - check if it's a drive root
    if "%INSTALL_DIR:~-2,1%"==":" (
        REM It's a drive root like C:\ - keep the backslash
        set "INSTALL_DIR_CLEAN=%INSTALL_DIR%"
    ) else (
        REM It's a regular path like C:\foo\ - remove trailing backslash
        set "INSTALL_DIR_CLEAN=%INSTALL_DIR:~0,-1%"
    )
)

REM Fix git safe.directory issue for network shares
echo   Configuring git for network share access...
git config --global --add safe.directory "*" >nul 2>&1

REM Check if install.py exists in current directory
if exist "%INSTALL_DIR%install.py" (
    echo   Running local install.py...
    python "%INSTALL_DIR%install.py" --dir "%INSTALL_DIR_CLEAN%" --force
) else (
    echo   install.py not found locally, downloading from GitHub...

    REM Use a more reliable download method with proper variable expansion
    set "INSTALLER_URL=https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/install.py"
    set "INSTALLER_FILE=%TEMP_DIR%\install.py"

    echo   Downloading from: !INSTALLER_URL!

    REM Try PowerShell first - use delayed expansion variables for PowerShell
    powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!INSTALLER_URL!' -OutFile '!INSTALLER_FILE!' -UseBasicParsing"

    if !errorlevel! neq 0 (
        echo [WARNING] PowerShell download failed, trying curl...
        curl -L -o "!INSTALLER_FILE!" "!INSTALLER_URL!"
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to download universal installer
            echo   Please download install.py manually from GitHub
            pause
            exit /b 1
        )
    )

    echo [OK] Universal installer downloaded successfully
    echo   Running downloaded universal installer...
    python "!INSTALLER_FILE!" --dir "!INSTALL_DIR_CLEAN!" --force
)

if !errorlevel! neq 0 (
    echo [ERROR] Universal installer failed
    echo   Check the error messages above for details
    pause
    exit /b 1
)

echo [OK] FunGen installation completed by universal installer

echo.
echo ================================================================
echo                  Installation Complete!
echo ================================================================
echo.
echo [OK] Prerequisites installed ^(Miniconda, Git, FFmpeg^)
echo [OK] FunGen universal installer completed successfully
echo.
echo Check above for launcher instructions.
echo.

pause

REM Cleanup
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
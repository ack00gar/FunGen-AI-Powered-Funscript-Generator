@echo off
setlocal enabledelayedexpansion

REM --- START OF USER CONFIGURATION ---
set REPO_URL="YOUR_REPO_URL_HERE"
set PROJECT_DIR_NAME="MyProject" REM Optional: if empty, uses repo name
set GOFILE_DIRECT_LINK="YOUR_GOFILE_DIRECT_LINK_HERE"
set MODEL_FILENAME_ON_DISK="your_model.file"
set ENV_NAME="my_project_env"
set PYTHON_VERSION="3.9" REM Used if environment.yml is not found
set MAIN_PYTHON_SCRIPT="main.py"
set CONDA_INSTALL_PATH="%USERPROFILE%\Miniconda3" REM Default Miniconda install path
REM --- END OF USER CONFIGURATION ---

echo Starting Project Installation for Windows...

REM Set Project Directory
if "%PROJECT_DIR_NAME%"=="" (
    for %%A in ("%REPO_URL%") do set PROJECT_DIR_NAME=%%~nA
)
set PROJECT_PATH="%CD%\%PROJECT_DIR_NAME%"

REM Function to check if a command exists
:check_command
where %1 >nul 2>nul
if %errorlevel% neq 0 (
    echo %1 is not found.
    exit /b 1
)
exit /b 0

REM 1. Check for Git
call :check_command git
if errorlevel 1 (
    echo Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/download/win and ensure it's in your PATH.
    echo Alternatively, try: winget install Git.Git
    pause
    exit /b 1
)
echo Git found.

REM 2. Clone Repository
if exist %PROJECT_PATH% (
    echo Project directory %PROJECT_PATH% already exists. Skipping clone.
) else (
    echo Cloning repository %REPO_URL% into %PROJECT_PATH%...
    git clone %REPO_URL% %PROJECT_PATH%
    if errorlevel 1 (
        echo Failed to clone repository.
        pause
        exit /b 1
    )
)
cd %PROJECT_PATH%

REM 3. Check for Conda / Install Miniconda
set CONDA_EXE="%CONDA_INSTALL_PATH%\Scripts\conda.exe"
call :check_command %CONDA_EXE%
if errorlevel 1 (
    echo Conda not found. Attempting to install Miniconda...
    set MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    set MINICONDA_INSTALLER="Miniconda3-latest-Windows-x86_64.exe"
    echo Downloading Miniconda installer...
    powershell -Command "Invoke-WebRequest -Uri %MINICONDA_URL% -OutFile %MINICONDA_INSTALLER%"
    if errorlevel 1 (
        echo Failed to download Miniconda installer. Please download and install manually.
        pause
        exit /b 1
    )
    echo Starting Miniconda installation (may require admin rights and manual clicks if /S fails)...
    start /wait "" %MINICONDA_INSTALLER% /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_INSTALL_PATH%
    del %MINICONDA_INSTALLER%
    echo Miniconda installation attempted. Please ensure it's correctly installed and PATH might need manual adjustment or restarting the terminal.
    set CONDA_EXE="%CONDA_INSTALL_PATH%\Scripts\conda.exe"
    set CONDA_ACTIVATE_SCRIPT="%CONDA_INSTALL_PATH%\Scripts\activate.bat"
    call :check_command %CONDA_EXE%
    if errorlevel 1 (
       echo Miniconda installation failed or not in PATH. Please install Miniconda manually.
       pause
       exit /b 1
    )
    echo It is STRONGLY recommended to close this window and re-run the script in a new CMD window for Conda to be properly initialized.
    pause
) else (
    echo Conda found.
    set CONDA_ACTIVATE_SCRIPT="%CONDA_INSTALL_PATH%\Scripts\activate.bat"
    REM If conda was found, try to find its base path if CONDA_INSTALL_PATH wasn't set perfectly
    if not exist %CONDA_ACTIVATE_SCRIPT% (
        for /f "delims=" %%i in ('where conda') do (
            set CANDIDATE_CONDA_PATH=%%~dpi
            set CONDA_ACTIVATE_SCRIPT="%%~dpi..\Scripts\activate.bat"
            set CONDA_EXE="%%~dpi..\Scripts\conda.exe"
            if exist !CONDA_ACTIVATE_SCRIPT! (
                set CONDA_INSTALL_PATH=%%~dpi..\
                goto conda_path_found_win
            )
        )
        echo Could not reliably find Conda's activate script. Assuming it's in PATH.
        set CONDA_ACTIVATE_SCRIPT="activate"
        :conda_path_found_win
    )
)


REM 4. Create Conda Environment
echo Checking for Conda environment: %ENV_NAME%...
call %CONDA_EXE% env list | findstr /C:"%ENV_NAME% " >nul
if errorlevel 1 (
    echo Creating Conda environment %ENV_NAME%...
    if exist "environment.yml" (
        echo Found environment.yml. Creating environment from file...
        call %CONDA_EXE% env create -f environment.yml -n %ENV_NAME%
    ) else (
        echo environment.yml not found. Creating environment with Python %PYTHON_VERSION%...
        call %CONDA_EXE% create -n %ENV_NAME% python=%PYTHON_VERSION% -y
        if errorlevel 1 (
            echo Failed to create Conda environment.
            pause
            exit /b 1
        )
        echo Installing dependencies if it exists...
        call %CONDA_ACTIVATE_SCRIPT% %ENV_NAME%
        pip install uv
        uv sync
        echo NOTE: FFmpeg might not be installed if environment.yml was not used.
        echo Consider adding ffmpeg to your requirements or creating an environment.yml.
        call %CONDA_ACTIVATE_SCRIPT% %ENV_NAME% && conda install -c conda-forge ffmpeg -y
    )
) else (
    echo Conda environment %ENV_NAME% already exists. Skipping creation.
    echo To update, run: %CONDA_EXE% env update -f environment.yml --name %ENV_NAME% --prune
)

REM 5. Create "models" subfolder
if not exist "models" (
    echo Creating "models" subdirectory...
    mkdir "models"
)

REM 6. Download file from Gofile
echo Downloading model file from %GOFILE_DIRECT_LINK%...
if "%GOFILE_DIRECT_LINK%"=="" (
    echo GOFILE_DIRECT_LINK is not set. Skipping download.
    echo Please download the model manually and place it in the 'models' folder as '%MODEL_FILENAME_ON_DISK%'.
) else (
    echo Using PowerShell to download...
    powershell -Command "Invoke-WebRequest -Uri '%GOFILE_DIRECT_LINK%' -OutFile 'models\%MODEL_FILENAME_ON_DISK%'"
    if errorlevel 1 (
        echo Failed to download model file. This might be due to an incorrect or non-direct link.
        echo Please download it manually from %GOFILE_DIRECT_LINK%
        echo And place it in the 'models' folder as '%MODEL_FILENAME_ON_DISK%'.
    ) else (
        echo Model downloaded to models\%MODEL_FILENAME_ON_DISK%.
    )
)

REM 7. Create Launcher
echo Creating launcher script (launch.bat)...
(
    echo @echo off
    echo set SCRIPT_DIR=%%~dp0
    echo echo Activating Conda environment %ENV_NAME%...
    echo call %CONDA_ACTIVATE_SCRIPT% %ENV_NAME%
    echo echo Launching %MAIN_PYTHON_SCRIPT%...
    echo python "%%SCRIPT_DIR%%\%MAIN_PYTHON_SCRIPT%" %%*
    echo echo.
    echo pause
) > launch.bat
if errorlevel 1 (
    echo Failed to create launcher script.
) else (
    echo Launcher script "launch.bat" created in %PROJECT_PATH%.
)

echo.
echo Installation process complete.
echo To run the application, navigate to %PROJECT_PATH% and double-click on "launch.bat".
echo Or, from a terminal already in %PROJECT_PATH%, run: launch.bat
pause
exit /b 0

@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo               Fix Old FunGen Installation
echo                      v1.0.0
echo ================================================================
echo This script fixes installations from before requirements files were added
echo.

cd /d "%~dp0"

echo [1/3] Checking for missing requirements files...

if not exist "core.requirements.txt" (
    echo ✗ core.requirements.txt missing - downloading from repository...
    
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/core.requirements.txt' -OutFile 'core.requirements.txt'}"
    
    if !errorlevel! neq 0 (
        echo ⚠ Could not download core.requirements.txt
        echo Creating basic requirements file...
        echo numpy > core.requirements.txt
        echo imgui >> core.requirements.txt
        echo ultralytics==8.3.78 >> core.requirements.txt
        echo glfw~=2.8.0 >> core.requirements.txt
        echo pyopengl~=3.1.7 >> core.requirements.txt
        echo imageio~=2.36.1 >> core.requirements.txt
        echo tqdm~=4.67.1 >> core.requirements.txt
        echo colorama~=0.4.6 >> core.requirements.txt
        echo opencv-python~=4.10.0.84 >> core.requirements.txt
        echo scipy~=1.15.1 >> core.requirements.txt
        echo simplification~=0.7.13 >> core.requirements.txt
        echo msgpack~=1.1.0 >> core.requirements.txt
        echo pillow~=11.1.0 >> core.requirements.txt
        echo orjson~=3.10.15 >> core.requirements.txt
        echo send2trash~=1.8.3 >> core.requirements.txt
        echo aiosqlite >> core.requirements.txt
    )
    echo ✓ core.requirements.txt created
) else (
    echo ✓ core.requirements.txt exists
)

echo.
echo [2/3] Checking for other missing requirements files...

for %%F in (requirements.txt cuda.requirements.txt cpu.requirements.txt) do (
    if not exist "%%F" (
        echo ⚠ %%F missing - downloading...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/main/%%F' -OutFile '%%F'}" 2>nul
        if exist "%%F" (
            echo ✓ Downloaded %%F
        )
    )
)

echo.
echo [3/3] Installing missing Python packages...

REM Try to find and activate environment
if exist "venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call "venv\Scripts\activate.bat"
) else (
    echo Using system Python...
)

echo Installing core requirements...
pip install -r core.requirements.txt

if !errorlevel! neq 0 (
    echo ⚠ pip install failed, trying individual packages...
    
    for %%P in (numpy imgui ultralytics glfw pyopengl imageio tqdm colorama opencv-python scipy simplification msgpack pillow orjson send2trash aiosqlite) do (
        echo Installing %%P...
        pip install %%P
    )
)

echo.
echo ================================================================
echo                    Fix Complete!
echo ================================================================
echo.
echo Your old FunGen installation should now have all required files.
echo.
echo To update to the latest version completely:
echo   git pull origin main
echo   OR download fresh copy from GitHub
echo.
echo To run FunGen:
echo   - Use FunGen_Windows_Fix.bat (recommended)
echo   - Or run: python main.py
echo.
pause
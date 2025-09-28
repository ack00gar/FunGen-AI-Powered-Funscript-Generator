@echo off
cd /d "%~dp0"

REM Try local venv first (most common with original installer)
if exist "venv\Scripts\activate.bat" (
    echo Using local virtual environment...
    call "venv\Scripts\activate.bat"
    python main.py
    pause
    exit /b 0
)

REM Try conda environments with multiple possible names
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat"
if exist "%CONDA_PATH%" (
    REM Try different environment names
    for %%E in (fungen VRFunAIGen FunGen) do (
        echo Trying conda environment: %%E
        call "%CONDA_PATH%" %%E 2>nul
        if !errorlevel! equ 0 (
            python main.py
            pause
            exit /b 0
        )
    )
)

echo ✗ No suitable Python environment found!
echo Please run one of these installers:
echo - fungen_install.bat (creates venv)
echo - fungen_enhanced_install.bat (creates conda env)
echo - Or use: FunGen_Smart_Start.bat
pause
exit /b 1
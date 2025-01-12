
@echo off
cd /d "%~dp0"
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat"

if not exist "%CONDA_PATH%" (
    echo Conda not found at %CONDA_PATH%. Please check your installation.
    pause
    exit /b 1
)
call "%CONDA_PATH%" VRFunAIGen
python FSGenerator.py
pause
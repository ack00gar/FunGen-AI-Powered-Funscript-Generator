@echo off
cd /d "%~dp0"
set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat"

if not exist "%CONDA_PATH%" (
    echo Conda not found at %CONDA_PATH%. Please check your installation.
    pause
    exit /b 1
)
call "%CONDA_PATH%" VRFunAIGen

:: Debug: Print Python path and package versions
python -c "import sys; print('Python Path:', sys.executable)"
python -c "import torch; print('Torch Version:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics Version:', ultralytics.__version__)"
python -c "import sys; print('Active Conda Env:', sys.prefix)"

python generate_tensorrt.py
pause
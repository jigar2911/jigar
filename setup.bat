@echo off
echo AI Watermark Remover - Windows Setup
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python 3.10+ from python.org
    pause
    exit /b
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Install requirements
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

:: Check for GPU
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected or drivers not installed. Installing CPU version...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Setup Complete!
echo To start the program, run: start_app.bat
echo.

:: Create a start script
echo @echo off > start_app.bat
echo call venv\Scripts\activate >> start_app.bat
echo python src/app.py >> start_app.bat
echo pause >> start_app.bat

pause

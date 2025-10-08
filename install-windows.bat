@echo off
REM SmolLM3 & SmolVLM2 - Installation Script for Windows
REM This script will set up everything you need to run the Gradio interface

setlocal enabledelayedexpansion

echo ========================================
echo SmolLM3 ^& SmolVLM2 - Installation Script
echo ========================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python: %PYTHON_VERSION%

REM Check Python version (must be >= 3.10)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo ERROR: Python 3.10 or higher is required
    pause
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 10 (
    echo ERROR: Python 3.10 or higher is required
    pause
    exit /b 1
)

echo Python version is compatible
echo.

REM Remove old virtual environment
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Detect GPU
echo.
echo Detecting hardware...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected
    set GPU_TYPE=NVIDIA
) else (
    echo No NVIDIA GPU detected, installing CPU version
    set GPU_TYPE=CPU
)

REM Install PyTorch
echo.
echo Installing PyTorch...
if "%GPU_TYPE%"=="NVIDIA" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)

REM Install other dependencies
echo.
echo Installing dependencies...
pip install "transformers>=4.53.0"
pip install "gradio>=4.0.0"
pip install "pillow>=9.0,<11.0"
pip install sentencepiece protobuf einops

REM Verify installations
echo.
echo Verifying installations...
python -c "import torch; import transformers; import gradio; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'Gradio: {gradio.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

REM Create start script
echo.
echo Creating start script...
(
echo @echo off
echo REM SmolLM3 ^& SmolVLM2 - Start Script
echo.
echo cd /d "%%~dp0"
echo call venv\Scripts\activate.bat
echo.
echo echo Starting SmolLM3 ^& SmolVLM2 Interface...
echo echo The interface will be available at: http://localhost:7860
echo echo.
echo.
echo python app.py
echo.
echo pause
) > start.bat

REM Installation complete
echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Next steps:
echo   1. Make sure app.py is in the current directory
echo   2. Double-click start.bat or run: start.bat
echo   3. Open your browser at: http://localhost:7860
echo.
echo Tips:
echo   - First launch will download ~10GB of models (15-20 minutes^)
echo   - Models are cached, subsequent launches are instant
if "%GPU_TYPE%"=="NVIDIA" (
    echo   - CUDA GPU acceleration is enabled
) else (
    echo   - Running on CPU (slower but works^)
)
echo.
echo Need help? Check the README.md file
echo.

pause

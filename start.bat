@echo off
REM SmolLM3 & SmolVLM2 - Start Script

cd /d "%~dp0"
call venv\Scripts\activate.bat

echo Starting SmolLM3 & SmolVLM2 Interface...
echo The interface will be available at: http://localhost:7860
echo.

python smollm3-gradio-app.py

pause

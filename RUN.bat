@echo off
echo ========================================
echo AI Video Generator (CPU Mode)
echo ========================================
echo.
echo WARNING: Running on CPU - video generation will be SLOW
echo Expect 5-10 minutes per video.
echo.
echo Starting web interface...
echo Your browser will open automatically.
echo.
echo The app will be available at: http://localhost:7860
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C to stop.
echo ========================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat

REM Check if Ollama is running
echo Checking Ollama service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama service...
    start /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
) else (
    echo Ollama is already running.
)

echo.
start http://localhost:7860
python gradio_interface.py

pause

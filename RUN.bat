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

REM Set CPU mode
set CUDA_VISIBLE_DEVICES=-1
set USE_GPU=false

start http://localhost:7860
python gradio_interface.py

pause

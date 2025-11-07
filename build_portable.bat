@echo off
echo ========================================
echo Building Portable AI Video Generator
echo ========================================
echo.

REM Create build directory
set BUILD_DIR=ai-video-generator-portable
if exist "%BUILD_DIR%" (
    echo Cleaning old build...
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"

echo.
echo Step 1: Copying application files...
xcopy /E /I /Y pipeline "%BUILD_DIR%\pipeline"
xcopy /E /I /Y data "%BUILD_DIR%\data"
xcopy /E /I /Y weights "%BUILD_DIR%\weights"
copy main.py "%BUILD_DIR%\"
copy config.py "%BUILD_DIR%\"
copy gradio_interface.py "%BUILD_DIR%\"
copy requirements.txt "%BUILD_DIR%\"
copy README.md "%BUILD_DIR%\" 2>nul

echo.
echo Step 2: Creating launcher scripts...

REM Create INSTALL.bat for first-time setup
(
echo @echo off
echo ========================================
echo AI Video Generator - First Time Setup
echo ========================================
echo.
echo This will set up the application ^(one-time only^).
echo This may take 5-10 minutes...
echo.
pause
echo.
echo Checking Python installation...
python --version ^>nul 2^>^&1
if %%errorlevel%% neq 0 ^(
    echo ERROR: Python 3.11 not found!
    echo.
    echo Please install Python 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
^)
echo Python found!
echo.
echo Creating virtual environment...
python -m venv venv
echo.
echo Installing packages ^(this will take a while^)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Desktop shortcut will be created...
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut^('%%USERPROFILE%%\Desktop\AI Video Generator.lnk'^); $s.TargetPath = '%%CD%%\RUN.bat'; $s.WorkingDirectory = '%%CD%%'; $s.IconLocation = 'shell32.dll,14'; $s.Save(^)"
echo.
echo You can now close this window and double-click
echo "AI Video Generator" on your desktop to start!
echo.
pause
) > "%BUILD_DIR%\INSTALL.bat"

REM Create RUN.bat for regular use
(
echo @echo off
echo ========================================
echo AI Video Generator
echo ========================================
echo.
cd /d "%%~dp0"
if not exist "venv\" ^(
    echo ERROR: Application not installed!
    echo Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
^)
echo Starting application...
echo Your browser will open automatically.
echo.
echo Web interface: http://localhost:7860
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C to stop.
echo ========================================
echo.
call venv\Scripts\activate.bat
start http://localhost:7860
python gradio_interface.py
pause
) > "%BUILD_DIR%\RUN.bat"

REM Create USER_GUIDE.txt
(
echo ========================================
echo AI Video Generator - User Guide
echo ========================================
echo.
echo SYSTEM REQUIREMENTS:
echo - Windows 10/11
echo - NVIDIA GPU with CUDA support
echo - 8GB RAM minimum ^(16GB recommended^)
echo - 10GB free disk space
echo.
echo FIRST TIME SETUP:
echo 1. Make sure you have Python 3.11 installed
echo    Download: https://www.python.org/downloads/
echo    IMPORTANT: Check "Add Python to PATH" during install
echo.
echo 2. Double-click INSTALL.bat
echo    - This will install all required packages
echo    - Takes 5-10 minutes on first run
echo    - Creates a desktop shortcut
echo.
echo 3. Done! A shortcut will appear on your desktop
echo.
echo DAILY USE:
echo - Double-click "AI Video Generator" on your desktop
echo - Wait for browser to open ^(~10 seconds^)
echo - Paste your script and click "Generate Video"
echo - Download the result when complete
echo.
echo TROUBLESHOOTING:
echo - If the app doesn't start, try running INSTALL.bat again
echo - Make sure your NVIDIA drivers are up to date
echo - Check that port 7860 isn't used by another app
echo.
echo GPU NOT DETECTED:
echo - Update NVIDIA drivers: https://www.nvidia.com/drivers
echo - App will run on CPU ^(much slower^) if GPU not available
echo.
echo PERFORMANCE:
echo - First video generation is slower ^(loading models^)
echo - Subsequent generations are faster
echo - 1080p video: ~2-5 minutes per minute of video
echo.
echo SUPPORT:
echo - Check the GitHub repository for updates
echo - Report issues on GitHub Issues page
echo.
) > "%BUILD_DIR%\USER_GUIDE.txt"

echo.
echo Step 3: Copying virtual environment...
echo NOTE: Skipping venv copy - users will create their own
echo This keeps the package size smaller.

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Portable package created in: %BUILD_DIR%
echo.
echo Next steps:
echo 1. Compress "%BUILD_DIR%" to a ZIP file
echo 2. Share the ZIP file with users
echo 3. Users extract and run INSTALL.bat
echo.
echo Package contents:
dir /s "%BUILD_DIR%"
echo.
pause

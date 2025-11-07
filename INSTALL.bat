@echo off
echo ========================================
echo AI Video Generator - Installer
echo ========================================
echo.
echo This will install:
echo - Python 3.11
echo - Required packages
echo - Desktop shortcut
echo.
pause

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python 3.11...
    echo Please download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH"
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing packages...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Creating desktop shortcut...
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\AI Video Generator.lnk'); $s.TargetPath = '%CD%\RUN.bat'; $s.WorkingDirectory = '%CD%'; $s.Save()"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo A shortcut has been created on your desktop.
echo Double-click "AI Video Generator" to start.
echo.
pause

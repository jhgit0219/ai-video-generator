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

echo Installing Python packages...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo ========================================
echo Checking Ollama Installation
echo ========================================
echo.
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama not found. Installing...
    echo.
    echo Downloading Ollama installer...
    curl -fsSL https://ollama.com/install.sh -o ollama-install.ps1
    powershell -ExecutionPolicy Bypass -File ollama-install.ps1
    del ollama-install.ps1
) else (
    echo Ollama is already installed!
)

echo.
echo ========================================
echo Downloading AI Models (This may take 10-20 minutes)
echo ========================================
echo.
echo Starting Ollama service...
start /B ollama serve >nul 2>&1
timeout /t 5 /nobreak >nul

echo Downloading llama3 model (~4GB)...
ollama pull llama3

echo.
echo Downloading deepseek-coder model (~3.8GB)...
ollama pull deepseek-coder:6.7b

echo.
echo ========================================
echo Creating Desktop Shortcut
echo ========================================
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\AI Video Generator.lnk'); $s.TargetPath = '%CD%\RUN.bat'; $s.WorkingDirectory = '%CD%'; $s.IconLocation = 'shell32.dll,14'; $s.Save()"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Total download size: ~10GB
echo  - Python packages: ~2GB
echo  - AI models: ~8GB
echo.
echo A shortcut has been created on your desktop.
echo Double-click "AI Video Generator" to start.
echo.
pause

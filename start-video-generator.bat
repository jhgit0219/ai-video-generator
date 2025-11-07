@echo off
echo ========================================
echo AI Video Generator - Docker Launcher
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Desktop is not running!
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo Docker is running... Starting AI Video Generator
echo.
echo The web interface will be available at:
echo http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

docker-compose --profile gpu up

pause

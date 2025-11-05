@echo off
REM Benchmark parallel rendering methods by running main.py 3 times
REM Clears enhanced cache each time to force re-rendering

echo ======================================================================
echo BENCHMARK: Testing 3 parallel rendering methods
echo ======================================================================
echo.

REM Clear enhanced cache before starting
echo [*] Clearing enhanced image cache...
if exist data\temp_images\enhanced_cache rmdir /s /q data\temp_images\enhanced_cache
echo.

REM ======================================================================
echo ======================================================================
echo METHOD 1: DISK-BASED RENDERING
echo ======================================================================
echo.

REM Update config.py to use disk method
powershell -Command "(gc config.py) -replace 'PARALLEL_RENDER_METHOD = \".*\"', 'PARALLEL_RENDER_METHOD = \"disk\"' | Out-File -encoding ASCII config.py"

echo [disk] Starting render...
powershell -Command "$start = Get-Date; .\venv\Scripts\python.exe main.py lost_labyrinth_script; $end = Get-Date; $elapsed = ($end - $start).TotalSeconds; Write-Host '[disk] Time:' $elapsed 's'" > benchmark_disk.txt 2>&1
type benchmark_disk.txt | findstr /C:"[disk] Time:"

echo.
echo.

REM ======================================================================
echo ======================================================================
echo METHOD 2: PIPE-BASED RENDERING
echo ======================================================================
echo.

REM Clear cache and update config
if exist data\temp_images\enhanced_cache rmdir /s /q data\temp_images\enhanced_cache
powershell -Command "(gc config.py) -replace 'PARALLEL_RENDER_METHOD = \".*\"', 'PARALLEL_RENDER_METHOD = \"pipe\"' | Out-File -encoding ASCII config.py"

echo [pipe] Starting render...
powershell -Command "$start = Get-Date; .\venv\Scripts\python.exe main.py lost_labyrinth_script; $end = Get-Date; $elapsed = ($end - $start).TotalSeconds; Write-Host '[pipe] Time:' $elapsed 's'" > benchmark_pipe.txt 2>&1
type benchmark_pipe.txt | findstr /C:"[pipe] Time:"

echo.
echo.

REM ======================================================================
echo ======================================================================
echo METHOD 3: CHUNKED RENDERING
echo ======================================================================
echo.

REM Clear cache and update config
if exist data\temp_images\enhanced_cache rmdir /s /q data\temp_images\enhanced_cache
powershell -Command "(gc config.py) -replace 'PARALLEL_RENDER_METHOD = \".*\"', 'PARALLEL_RENDER_METHOD = \"chunk\"' | Out-File -encoding ASCII config.py"

echo [chunk] Starting render...
powershell -Command "$start = Get-Date; .\venv\Scripts\python.exe main.py lost_labyrinth_script; $end = Get-Date; $elapsed = ($end - $start).TotalSeconds; Write-Host '[chunk] Time:' $elapsed 's'" > benchmark_chunk.txt 2>&1
type benchmark_chunk.txt | findstr /C:"[chunk] Time:"

echo.
echo.

REM ======================================================================
echo ======================================================================
echo RESULTS SUMMARY
echo ======================================================================
echo.
type benchmark_disk.txt | findstr /C:"[disk] Time:"
type benchmark_pipe.txt | findstr /C:"[pipe] Time:"
type benchmark_chunk.txt | findstr /C:"[chunk] Time:"
echo.
echo ======================================================================

REM Reset to disabled
powershell -Command "(gc config.py) -replace 'PARALLEL_RENDER_METHOD = \".*\"', 'PARALLEL_RENDER_METHOD = \"disabled\"' | Out-File -encoding ASCII config.py"

echo.
echo Benchmark complete! Check benchmark_*.txt files for full logs.
pause

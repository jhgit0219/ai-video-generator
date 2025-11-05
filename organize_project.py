"""
Project organization script.
Moves test scripts to custom_test/ and logs to logs/.
Cleans up temporary rendering files.
"""

import os
import shutil
from pathlib import Path

# Create directories
CUSTOM_TEST_DIR = Path("custom_test")
LOGS_DIR = Path("logs")
TEMP_CHUNKS_DIR = Path("data/temp_images/frame_sequence")

CUSTOM_TEST_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

print("[organize] Organizing project files...")

# Move test scripts to custom_test/
test_scripts = list(Path(".").glob("test_*.py"))
if test_scripts:
    print(f"[organize] Moving {len(test_scripts)} test scripts to custom_test/")
    for script in test_scripts:
        if script.name == "test_effects.py":
            # Keep test_effects.py in root (it's the main test harness)
            continue
        dest = CUSTOM_TEST_DIR / script.name
        shutil.move(str(script), str(dest))
        print(f"   - Moved {script.name}")

# Move log files to logs/
log_files = list(Path(".").glob("*.log")) + list(Path(".").glob("test_*.txt"))
if log_files:
    print(f"[organize] Moving {len(log_files)} log files to logs/")
    for log in log_files:
        dest = LOGS_DIR / log.name
        shutil.move(str(log), str(dest))
        print(f"   - Moved {log.name}")

# Clean up temporary chunk directories (from failed parallel renders)
if TEMP_CHUNKS_DIR.exists():
    chunk_dirs = [d for d in TEMP_CHUNKS_DIR.iterdir() if d.is_dir()]
    if chunk_dirs:
        print(f"[organize] Cleaning up {len(chunk_dirs)} temporary chunk directories")
        for chunk_dir in chunk_dirs:
            try:
                shutil.rmtree(chunk_dir)
                print(f"   - Removed {chunk_dir.name}")
            except Exception as e:
                print(f"   - Failed to remove {chunk_dir.name}: {e}")

# Clean up orphaned temporary video files in output dir
output_dir = Path("data/output")
if output_dir.exists():
    temp_videos = [f for f in output_dir.glob("*.mp4") if f.stat().st_size < 1024 * 100]  # < 100KB
    if temp_videos:
        print(f"[organize] Found {len(temp_videos)} small/incomplete video files")
        for vid in temp_videos:
            print(f"   - {vid.name} ({vid.stat().st_size} bytes) - keeping (manual review recommended)")

print("[organize] Organization complete!")
print(f"   Test scripts: custom_test/ ({len(list(CUSTOM_TEST_DIR.glob('*.py')))} files)")
print(f"   Logs: logs/ ({len(list(LOGS_DIR.glob('*')))} files)")

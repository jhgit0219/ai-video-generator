"""
Helper functions for the AI Video Generator.
Contains utility functions used across different modules.
"""

import os
from typing import Dict, Any
import json
from pathlib import Path
import shutil
from config import TEMP_IMAGES_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path (str): Path to the directory
    """
    os.makedirs(path, exist_ok=True)

def initialize_required_directories() -> None:
    """
    Initialize all required directories for the application.
    Should be called at startup to ensure the data structure exists.
    Creates: data/input, data/temp_images, data/output, data/scraper_cache, data/logs/scraper, weights
    """
    from config import INPUT_DIR, TEMP_IMAGES_DIR, OUTPUT_DIR, SCRAPER_CACHE_DIR, SCRAPER_LOGS_DIR
    
    required_dirs = [
        INPUT_DIR,
        TEMP_IMAGES_DIR,
        OUTPUT_DIR,
        SCRAPER_CACHE_DIR,
        SCRAPER_LOGS_DIR,
        "weights",  # For AI model weights
    ]
    
    for directory in required_dirs:
        ensure_directory(directory)
        logger.debug(f"Ensured directory exists: {directory}")
        
        # Add .gitkeep to preserve empty directories in git
        gitkeep_path = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            try:
                with open(gitkeep_path, "w") as f:
                    f.write("# This file ensures the directory is tracked by git even when empty\n")
            except Exception as e:
                logger.debug(f"Could not create .gitkeep in {directory}: {e}")
    
    logger.info("All required directories initialized successfully")

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Parsed JSON content
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File extension without dot
    """
    return Path(file_path).suffix[1:]

def create_temp_path(filename: str, subdir: str = "") -> str:
    """
    Create a path in the temporary directory.
    
    Args:
        filename (str): Name of the file
        subdir (str): Optional subdirectory within temp directory
        
    Returns:
        str: Full path to the temporary file
    """
    temp_dir = os.path.join(TEMP_IMAGES_DIR, subdir)
    ensure_directory(temp_dir)
    return os.path.join(temp_dir, filename)

def cleanup_temp_files(pattern: str = "*") -> None:
    """
    Clean up temporary image files or the entire temp directory.
    
    Args:
        pattern (str): File pattern to match for deletion
    """
    temp_path = Path(TEMP_IMAGES_DIR)

    # If the directory exists, delete and recreate it
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)

def cleanup_images_only():
    """Remove downloaded image files and segment folders, but keep the manifest JSON."""
    base = Path(TEMP_IMAGES_DIR)
    if not base.exists():
        return

    deleted_count = 0
    for item in base.iterdir():
        # remove segment folders or image files directly in temp folder
        if item.is_dir() and item.name.startswith("segment_"):
            shutil.rmtree(item, ignore_errors=True)
            logger.info(f"Deleted folder: {item}")
            deleted_count += 1
        elif item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            item.unlink(missing_ok=True)
            logger.info(f"Deleted image file: {item.name}")
            deleted_count += 1

    logger.info(f"🧹 cleanup_images_only: removed {deleted_count} files/folders but kept manifest intact.")

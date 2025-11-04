"""
Logging utility for the AI Video Generator.
Provides consistent logging format across all modules with modular file outputs.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import LOG_LEVEL

# Directory for all log files
LOG_DIR = "data/logs"

# Module-to-logfile mapping for organized debugging
MODULE_LOG_MAPPING = {
    "__main__": "main.log",
    "main": "main.log",
    "pipeline.director_agent": "director_agent.log",
    "pipeline.scraper.collector": "scraper.log",
    "pipeline.scraper.google_scraper": "scraper.log",
    "pipeline.scraper.utils": "scraper.log",
    "pipeline.ai_filter.clip_ranker": "clip_ranker.log",
    "pipeline.ai_filter.semantic_filter": "clip_ranker.log",
    "pipeline.renderer.effects_director": "effects_director.log",
    "pipeline.renderer.deepseek_effects_director": "effects_director.log",
    "pipeline.renderer.video_generator": "video_generator.log",
    "pipeline.renderer.image_enhancer": "video_generator.log",
    "pipeline.renderer.subject_detection": "video_generator.log",
    "pipeline.renderer.effects": "video_generator.log",
}

# Loggers that have already been configured (avoid duplicate handlers)
_configured_loggers = set()

def _ensure_log_directory():
    """Create log directory if it doesn't exist."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def _get_log_file_for_module(module_name: str) -> str:
    """
    Determine which log file a module should write to.

    Args:
        module_name: The module's __name__ value

    Returns:
        Log filename (not full path)
    """
    # Check for exact match first
    if module_name in MODULE_LOG_MAPPING:
        return MODULE_LOG_MAPPING[module_name]

    # Check for prefix match (e.g., pipeline.renderer.effects.overlay_neon -> video_generator.log)
    for prefix, log_file in MODULE_LOG_MAPPING.items():
        if module_name.startswith(prefix):
            return log_file

    # Default to main.log for unmapped modules
    return "main.log"

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with modular file outputs and consistent formatting.

    Each module logs to:
    1. Its specific log file (e.g., director_agent.log)
    2. The combined all.log file
    3. Console (INFO level only for reduced noise)

    File logs use DEBUG level for detailed troubleshooting.
    Log rotation: keeps last 5 runs, max 10MB per file.

    Args:
        name: Name of the logger, typically __name__ of the module

    Returns:
        logging.Logger: Configured logger instance with modular handlers
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger is already configured
    if name in _configured_loggers:
        return logger

    _configured_loggers.add(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    logger.propagate = False  # Don't propagate to root logger to avoid duplicates

    # Ensure log directory exists
    _ensure_log_directory()

    # Extract readable module name for log format (last part of dotted name)
    module_short_name = name.split('.')[-1] if '.' in name else name

    # Create formatters
    # Console: simpler format for readability
    console_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File: detailed format with milliseconds
    file_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler - INFO level only for reduced noise
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Module-specific file handler - DEBUG level for detailed troubleshooting
    module_log_file = _get_log_file_for_module(name)
    module_file_path = os.path.join(LOG_DIR, module_log_file)
    module_file_handler = RotatingFileHandler(
        module_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    module_file_handler.setLevel(logging.DEBUG)
    module_file_handler.setFormatter(file_formatter)
    logger.addHandler(module_file_handler)

    # Combined all.log handler - DEBUG level for complete picture
    all_log_path = os.path.join(LOG_DIR, "all.log")
    all_file_handler = RotatingFileHandler(
        all_log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    all_file_handler.setLevel(logging.DEBUG)
    all_file_handler.setFormatter(file_formatter)
    logger.addHandler(all_file_handler)

    return logger

# Create a default logger instance
logger = setup_logger(__name__)
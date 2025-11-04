"""
Session Logger - Saves detailed logs for each pipeline stage.

Creates a session directory with timestamped logs containing:
- Inputs (what went into this stage)
- Plans (LLM decisions, rankings, selections)
- Outputs (what came out of this stage)

Directory structure:
data/sessions/
  session_2025-11-04_12-34-56/
    00_pipeline_start.json
    01_content_analysis.json
    02_scraping_segment_0.json
    02_scraping_segment_1.json
    ...
    03_ranking_segment_0.json
    ...
    04_effects_planning.json
    05_rendering.json
    99_pipeline_complete.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SessionLogger:
    """
    Logs detailed information for each pipeline stage to help with debugging.

    Usage:
        session = SessionLogger.create_session("mia_story")

        # Log content analysis
        session.log_stage("content_analysis", {
            "inputs": {"segments": segment_data},
            "plan": {"genre": "fiction_fantasy", "required_subjects": [...]},
            "outputs": {"analyzed_segments": [...]}
        })

        # Log scraping for segment 0
        session.log_stage("scraping_segment_0", {
            "inputs": {"query": "golden sunset village", "max_images": 5},
            "plan": {"refined_query": "...", "retry_count": 2},
            "outputs": {"scraped_urls": [...], "successful_downloads": 3}
        })
    """

    def __init__(self, session_dir: Path, story_name: str):
        self.session_dir = session_dir
        self.story_name = story_name
        self.stage_counter = 0
        self.start_time = datetime.now()

        logger.info(f"[session_logger] Session started: {session_dir}")

    @classmethod
    def create_session(cls, story_name: str) -> 'SessionLogger':
        """Create a new session directory and return logger."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = Path(f"data/sessions/session_{timestamp}_{story_name}")
        session_dir.mkdir(parents=True, exist_ok=True)

        session = cls(session_dir, story_name)

        # Log initial metadata
        session.log_stage("pipeline_start", {
            "story_name": story_name,
            "timestamp": timestamp,
            "session_dir": str(session_dir)
        })

        return session

    def log_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a pipeline stage with inputs, plan, and outputs.

        Args:
            stage_name: Name of the stage (e.g., "content_analysis", "scraping_segment_0")
            data: Dict with keys "inputs", "plan", "outputs" (all optional)
            metadata: Additional metadata (timing, errors, etc.)
        """
        self.stage_counter += 1

        # Build log entry
        log_entry = {
            "stage_number": self.stage_counter,
            "stage_name": stage_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds()
        }

        # Add data sections
        if "inputs" in data:
            log_entry["inputs"] = data["inputs"]
        if "plan" in data:
            log_entry["plan"] = data["plan"]
        if "outputs" in data:
            log_entry["outputs"] = data["outputs"]
        if metadata:
            log_entry["metadata"] = metadata

        # Write to file
        filename = f"{self.stage_counter:02d}_{stage_name}.json"
        filepath = self.session_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, default=str)

            logger.debug(f"[session_logger] Logged stage: {filename}")
        except Exception as e:
            logger.warning(f"[session_logger] Failed to log stage {stage_name}: {e}")

    def log_error(self, stage_name: str, error: Exception, context: Optional[Dict] = None):
        """Log an error that occurred during a stage."""
        self.stage_counter += 1

        log_entry = {
            "stage_number": self.stage_counter,
            "stage_name": f"{stage_name}_ERROR",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "context": context or {}
            }
        }

        filename = f"{self.stage_counter:02d}_{stage_name}_ERROR.json"
        filepath = self.session_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, default=str)

            logger.error(f"[session_logger] Logged error: {filename}")
        except Exception as write_error:
            logger.error(f"[session_logger] Failed to log error: {write_error}")

    def complete(self, summary: Optional[Dict] = None):
        """Mark session as complete and write summary."""
        self.log_stage("pipeline_complete", {
            "outputs": {
                "total_stages": self.stage_counter,
                "total_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "summary": summary or {}
            }
        })

        logger.info(f"[session_logger] Session complete: {self.session_dir}")


# Global session instance (set by main.py)
_global_session: Optional[SessionLogger] = None


def get_session() -> Optional[SessionLogger]:
    """Get the current global session logger (if any)."""
    return _global_session


def set_session(session: SessionLogger):
    """Set the global session logger."""
    global _global_session
    _global_session = session


def log_stage(stage_name: str, data: Dict[str, Any], metadata: Optional[Dict] = None):
    """Convenience function to log to global session (if exists)."""
    session = get_session()
    if session:
        session.log_stage(stage_name, data, metadata)


def log_error(stage_name: str, error: Exception, context: Optional[Dict] = None):
    """Convenience function to log error to global session (if exists)."""
    session = get_session()
    if session:
        session.log_error(stage_name, error, context)

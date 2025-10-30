"""
Audio overlay and subtitle utilities (placeholder).
This module combines simple postprocessing stubs so the main pipeline can call
postprocessing.apply_effects and postprocessing.overlay_subtitles.
"""
from typing import List
from utils.logger import setup_logger
from pipeline.parser import VideoSegment

logger = setup_logger(__name__)


def apply_effects(segments: List[VideoSegment], audio_file: str) -> List[VideoSegment]:
    """
    Apply visual/audio effects to each segment (placeholder).

    Args:
        segments (List[VideoSegment]): segments with selected images
        audio_file (str): path to the audio file for synchronization

    Returns:
        List[VideoSegment]: segments with `processed_path` set to dummy file paths
    """
    logger.info("[audio_overlay] applying effects (placeholder)")

    for i, segment in enumerate(segments):
        # Create a dummy processed path (would be real video assembled per-segment)
        segment.processed_path = f"data/temp_images/processed_segment_{i}.mp4"

    logger.info(f"[audio_overlay] applied effects to {len(segments)} segments")
    return segments


def overlay_subtitles(segments: List[VideoSegment], audio_file: str) -> List[VideoSegment]:
    """
    Generate and overlay subtitles (placeholder).

    Args:
        segments (List[VideoSegment]): processed segments
        audio_file (str): path to audio for STT

    Returns:
        List[VideoSegment]: same segments, optionally annotated with subtitle info
    """
    logger.info("[audio_overlay] overlaying subtitles (placeholder)")

    # Placeholder: annotate segments with a dummy subtitle field
    for segment in segments:
        text = getattr(segment, "transcript", None) or getattr(segment, "text", "")
        setattr(segment, "subtitles", [
            {"start": 0.0, "end": getattr(segment, "duration", 0.0), "text": text}
        ])

    logger.info(f"[audio_overlay] added subtitles to {len(segments)} segments")
    return segments

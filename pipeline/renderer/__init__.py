"""
Renderer subpackage init.
Expose the primary video generation entrypoint.
"""
from .video_generator import render_final_video

__all__ = ["render_final_video"]

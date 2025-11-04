"""Compatibility shim: re-export modular effects and TOOLS_REGISTRY.

This file now delegates to the modular package under pipeline.renderer.effects.
"""
from __future__ import annotations

from typing import Callable, Dict
from moviepy.video.VideoClip import VideoClip

from .effects.registry import TOOLS_REGISTRY
from .effects.overlay_neon import apply_neon_overlay
from .effects.overlay_nightvision import apply_nightvision
from .effects.overlay_text import apply_paneled_text
from .effects.subject_pop import apply_subject_pop
from .effects.subject_outline import apply_subject_outline
from .effects.zoom_variable import apply_variable_zoom_on_subject
from .effects.zoom_temporal import apply_temporal_zoom
from .effects.zoom_then_panel import apply_zoom_then_panel

# Re-export for backward compatibility
__all__ = [
    "apply_neon_overlay",
    "apply_nightvision",
    "apply_paneled_text",
    "apply_subject_pop",
    "apply_subject_outline",
    "apply_variable_zoom_on_subject",
    "apply_temporal_zoom",
    "apply_zoom_then_panel",
    "TOOLS_REGISTRY",
]

from __future__ import annotations

from typing import Callable, Dict
from moviepy.video.VideoClip import VideoClip

from .overlay_neon import apply_neon_overlay
from .overlay_nightvision import apply_nightvision
from .overlay_text import apply_paneled_text
from .subject_pop import apply_subject_pop
from .subject_outline import apply_subject_outline
from .zoom_variable import apply_variable_zoom_on_subject
from .zoom_temporal import apply_temporal_zoom
from .zoom_then_panel import apply_zoom_then_panel
from .overlay_flash import apply_flash_pulse, apply_quick_flash, apply_strobe_effect
from .overlay_color_grade import (
    apply_color_grade,
    apply_warm_grade,
    apply_cool_grade,
    apply_desaturated_grade,
    apply_cinematic_teal_orange,
)
from .branding import (
    # Atomic effects
    apply_slime_splatter,
    apply_full_frame_tint,
    apply_animated_location_text,
    apply_subject_glow,
    # Composite effects
    apply_map_highlight,
    apply_character_highlight,
    apply_news_overlay,
    apply_branded_transition,
)

TOOLS_REGISTRY: Dict[str, Callable[..., VideoClip]] = {
    # Motion & Zoom
    "zoom_on_subject": apply_variable_zoom_on_subject,
    "temporal_zoom": apply_temporal_zoom,
    "zoom_then_panel": apply_zoom_then_panel,

    # Text Overlays
    "paneled_text": apply_paneled_text,

    # Subject Effects
    "subject_pop": apply_subject_pop,
    "subject_outline": apply_subject_outline,

    # Visual Overlays
    "neon_overlay": apply_neon_overlay,
    "nightvision": apply_nightvision,

    # Flash/Pulse Effects
    "flash_pulse": apply_flash_pulse,
    "quick_flash": apply_quick_flash,
    "strobe_effect": apply_strobe_effect,

    # Color Grading
    "color_grade": apply_color_grade,
    "warm_grade": apply_warm_grade,
    "cool_grade": apply_cool_grade,
    "desaturated_grade": apply_desaturated_grade,
    "teal_orange_grade": apply_cinematic_teal_orange,

    # Branding Effects - Atomic
    "slime_splatter": apply_slime_splatter,
    "full_frame_tint": apply_full_frame_tint,
    "animated_location_text": apply_animated_location_text,
    "subject_glow": apply_subject_glow,

    # Branding Effects - Composite
    "map_highlight": apply_map_highlight,
    "character_highlight": apply_character_highlight,
    "news_overlay": apply_news_overlay,
    "branded_transition": apply_branded_transition,
}

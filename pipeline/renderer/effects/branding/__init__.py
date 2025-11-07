"""Branding effects package for on-brand visual treatments.

Atomic effects (composable):
- Slime splatter overlays
- Full-frame color tints
- Animated location text
- Subject glow (multi-stage animated glow) - frame-anchored version
- Subject glow baked (multi-stage animated glow) - content-anchored version

Composite effects (convenience wrappers):
- Map highlight (combines slime + tint + text)
- Character highlight (combines subject glow + text)
- News overlay (vintage newspaper styling)
- Newspaper frame (newspaper layout with video in cut-out)
- Branded transitions (purple/green presets)
"""

# Atomic effects
from .slime_splatter import apply_slime_splatter
from .full_frame_tint import apply_full_frame_tint
from .animated_location_text import apply_animated_location_text
from .subject_glow import apply_subject_glow
from .subject_glow_baked import apply_subject_glow_baked

# Composite effects
from .map_highlight import apply_map_highlight
from .character_highlight import apply_character_highlight
from .news_overlay import apply_news_overlay
from .newspaper_frame import apply_newspaper_frame
from .branded_transitions import apply_branded_transition

__all__ = [
    # Atomic
    "apply_slime_splatter",
    "apply_full_frame_tint",
    "apply_animated_location_text",
    "apply_subject_glow",
    "apply_subject_glow_baked",
    # Composite
    "apply_map_highlight",
    "apply_character_highlight",
    "apply_news_overlay",
    "apply_newspaper_frame",
    "apply_branded_transition",
]

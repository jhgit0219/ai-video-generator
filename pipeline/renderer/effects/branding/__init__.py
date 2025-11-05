"""Branding effects package for on-brand visual treatments.

Effects designed for specific brand aesthetics:
- Map highlights with slime green location markers
- Character highlights with cyan glow
- News/publisher vintage newspaper overlays
- Branded transitions (purple/green presets)
"""

from .map_highlight import apply_map_highlight
from .character_highlight import apply_character_highlight
from .news_overlay import apply_news_overlay
from .branded_transitions import apply_branded_transition

__all__ = [
    "apply_map_highlight",
    "apply_character_highlight",
    "apply_news_overlay",
    "apply_branded_transition",
]

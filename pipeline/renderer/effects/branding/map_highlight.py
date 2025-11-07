"""Map highlight effect template - composes atomic branding effects.

Template that combines:
1. Slime splatter overlay (organic blob at location)
2. Full-frame tint (subtle green wash)
3. Animated location text (letter-by-letter)

This is a convenience wrapper that maintains backward compatibility
while using modular atomic effects under the hood.
"""
from __future__ import annotations

from typing import Optional, Tuple

from moviepy.video.VideoClip import VideoClip as _VideoClip

from utils.logger import setup_logger
from .slime_splatter import apply_slime_splatter
from .full_frame_tint import apply_full_frame_tint
from .animated_location_text import apply_animated_location_text

logger = setup_logger(__name__)

# Brand colors
SLIME_GREEN = (0, 255, 0)
SEPIA_TONE = (112, 66, 20)


def apply_map_highlight(
    clip: _VideoClip,
    location_name: str,
    box_position: Tuple[float, float, float, float],
    cps: int = 8,
    highlight_duration: float = 2.0,
    fade_to_normal_duration: float = 1.5,
    box_thickness: int = 8,
    fontsize: int = 72,
    font_path: Optional[str] = None,
    text_position: str = "center",
    glow_intensity: float = 0.6,
) -> _VideoClip:
    """Apply map highlight effect (template composition).

    Composes three atomic effects:
    - Slime splatter at box_position
    - Full-frame green tint
    - Animated location text

    :param clip: Input video clip.
    :param location_name: Location name to display (e.g., "EGYPT").
    :param box_position: Normalized (x, y, w, h) for splatter placement.
    :param cps: Characters per second for text animation.
    :param highlight_duration: Duration in seconds to hold bright green.
    :param fade_to_normal_duration: Duration to fade to sepia (not yet implemented).
    :param box_thickness: Unused (kept for backward compatibility).
    :param fontsize: Font size for location text.
    :param font_path: Path to font file.
    :param text_position: Text position: "center", "top", or "bottom".
    :param glow_intensity: Unused (kept for backward compatibility).
    :return: Composed video clip with all three effects.
    """
    # Calculate text position based on box_position and text_position parameter
    bx, by, bw, bh = box_position

    if text_position == "center":
        text_pos = (bx + bw / 2, by + bh / 2)
    elif text_position == "top":
        text_pos = (bx + bw / 2, max(0.05, by - 0.05))
    else:  # bottom
        text_pos = (bx + bw / 2, min(0.95, by + bh + 0.05))

    # LAYER 1: Apply slime splatter
    clip = apply_slime_splatter(
        clip,
        position=box_position,
        color=SLIME_GREEN,
        opacity=0.7,  # 180/255 alpha
        seed=hash(location_name) % 1000,
        animate_in=0.4,
    )

    # LAYER 2: Apply full-frame green tint
    clip = apply_full_frame_tint(
        clip,
        color=SLIME_GREEN,
        opacity=0.31,  # 80/255 alpha (subtle wash)
        animate_in=0.4,
    )

    # LAYER 3: Apply animated location text
    clip = apply_animated_location_text(
        clip,
        text=location_name,
        position=text_pos,
        color=SLIME_GREEN,
        font_size=fontsize,
        font_path=font_path,
        cps=cps,
        outline_width=3,
        outline_color=(0, 0, 0),
        animate_in=0.4,
    )

    logger.debug(f"[map_highlight] Composed template for location: {location_name}")

    # TODO: Implement color transition from green to sepia
    # This would require time-aware color changing in atomic effects
    # For now, keeping bright green throughout

    return clip

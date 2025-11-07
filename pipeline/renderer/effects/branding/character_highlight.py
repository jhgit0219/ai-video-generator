"""Character highlight effect template - composes atomic branding effects.

Template that combines:
1. Subject glow (multi-stage animated glow around character)
2. Animated character name text (letter-by-letter)

This is a convenience wrapper that maintains backward compatibility
while using modular atomic effects under the hood.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils.logger import setup_logger
from ..coords import denorm_bbox
from .subject_glow import apply_subject_glow
from .animated_location_text import apply_animated_location_text

logger = setup_logger(__name__)

# Brand colors for character highlights
CYAN_GLOW = (0, 255, 255)
GREEN_GLOW = (0, 255, 200)


def apply_character_highlight(
    clip,
    character_name: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[np.ndarray] = None,
    cps: int = 10,
    glow_color: Tuple[int, int, int] = CYAN_GLOW,
    glow_stages: Tuple[float, float, float] = (1.0, 0.6, 0.3),
    stage_durations: Tuple[float, float, float] = (0.5, 1.0, 1.5),
    fontsize: int = 64,
    font_path: Optional[str] = None,
    label_position: str = "top",
    label_offset: int = 20,
):
    """Apply character highlight effect (template composition).

    Composes two atomic effects:
    - Subject glow (multi-stage animated glow)
    - Animated character name text (letter-by-letter)

    :param clip: Input video clip to apply highlight to.
    :param character_name: Name of character to display.
    :param bbox: Optional normalized (x, y, w, h) subject bounding box.
    :param mask: Optional subject segmentation mask (from YOLO+CLIP).
    :param cps: Characters per second for letter-by-letter name animation.
    :param glow_color: RGB color for glow effect.
    :param glow_stages: Tuple of 3 intensity levels (0-1) for bright/medium/subtle.
    :param stage_durations: Duration in seconds for each intensity stage.
    :param fontsize: Font size for character name label.
    :param font_path: Path to Century font file (tries system fonts if None).
    :param label_position: Where to place label relative to subject: "top", "bottom".
    :param label_offset: Pixel offset between subject and label.
    :return: Video clip with character highlight effect applied.
    """
    w, h = clip.size

    # Auto-detect subject if no bbox/mask provided
    if bbox is None and mask is None:
        logger.warning("[character_highlight] No bbox or mask provided, using centered default")
        bbox = (0.3, 0.2, 0.4, 0.6)

    # Convert normalized bbox to pixels for text positioning
    if bbox:
        bx, by, bw, bh = denorm_bbox(bbox, (w, h))
    else:
        # Fallback if only mask provided
        bx, by, bw, bh = (w // 4, h // 4, w // 2, h // 2)

    # Calculate text position relative to subject
    # For horizontal centering, use normalized coordinates (0.5 relative to bbox)
    text_x_norm = (bx + bw / 2) / w  # Normalized X (center of bbox)

    if label_position == "top":
        # Place text above subject
        text_y_norm = max(0.05, (by - label_offset) / h)
    else:  # bottom
        # Place text below subject
        text_y_norm = min(0.95, (by + bh + label_offset) / h)

    # LAYER 1: Apply subject glow
    clip = apply_subject_glow(
        clip,
        bbox=bbox,
        mask=mask,
        glow_color=glow_color,
        glow_stages=glow_stages,
        stage_durations=stage_durations,
    )

    # LAYER 2: Apply animated character name text
    clip = apply_animated_location_text(
        clip,
        text=character_name,
        position=(text_x_norm, text_y_norm),
        color=glow_color,
        font_size=fontsize,
        font_path=font_path,
        cps=cps,
        outline_width=3,
        outline_color=(0, 0, 0),
        animate_in=0.4,
    )

    logger.debug(f"[character_highlight] Composed template for character: {character_name}")

    return clip

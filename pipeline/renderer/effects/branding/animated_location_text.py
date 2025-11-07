"""Animated location text overlay effect.

Displays location name with letter-by-letter typewriter animation
and optional outline for visibility.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip

from utils.logger import setup_logger
from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)


def apply_animated_location_text(
    clip: _VideoClip,
    text: str,
    position: tuple[float, float],  # Normalized (x, y) center position
    color: tuple[int, int, int] = (0, 255, 0),
    font_size: int = 72,
    font_path: Optional[str] = None,
    cps: int = 8,  # Characters per second
    outline_width: int = 3,
    outline_color: tuple[int, int, int] = (0, 0, 0),
    animate_in: float = 0.4,
) -> _VideoClip:
    """Apply animated location text with typewriter effect.

    :param clip: Input video clip.
    :param text: Location name to display (e.g., "EGYPT").
    :param position: Normalized (x, y) position for text center.
    :param color: RGB text color (default: bright green).
    :param font_size: Font size in points (default: 72).
    :param font_path: Path to font file (default: tries Century, falls back to default).
    :param cps: Characters per second for typewriter animation.
    :param outline_width: Text outline thickness in pixels.
    :param outline_color: RGB outline color (default: black).
    :param animate_in: Fade-in duration in seconds (default: 0.4).
    :return: Video clip with animated text overlay.
    """
    w, h = clip.size
    dur = float(clip.duration or 1.0)

    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try common Century font names
            for font_name in ["CENTURY.TTF", "century.ttf", "Century.ttf", "arial.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    logger.debug(f"[animated_location_text] Loaded font: {font_name}")
                    break
                except Exception:
                    continue
            else:
                font = ImageFont.load_default()
                logger.warning("[animated_location_text] Century font not found, using default")
    except Exception as e:
        logger.warning(f"[animated_location_text] Font loading failed: {e}, using default")
        font = ImageFont.load_default()

    # Calculate text dimensions and position
    tmp_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    try:
        text_bbox = tmp_draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        text_w = len(text) * (font_size // 2)
        text_h = font_size

    # Convert normalized position to pixels (centered)
    px, py = position
    text_x = int(px * w - text_w / 2)
    text_y = int(py * h - text_h / 2)

    # Ensure text stays in bounds
    text_x = max(10, min(w - text_w - 10, text_x))
    text_y = max(10, min(h - text_h - 10, text_y))

    text_length = len(text)

    def make_overlay_frame(t: float):
        """Generate text overlay frame."""
        # Fade-in animation
        fade_in_progress = min(1.0, t / animate_in) if animate_in > 0 else 1.0
        fade_in_progress = ease_in_out_cubic(fade_in_progress)
        text_alpha = int(255 * fade_in_progress)

        # Create transparent overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate visible characters (typewriter effect)
        chars_shown = min(text_length, max(0, int(t * cps)))
        visible_text = text[:chars_shown]

        if visible_text:
            text_color_with_alpha = (*color, text_alpha)

            # Draw text outline
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    if dx != 0 or dy != 0:
                        draw.text(
                            (text_x + dx, text_y + dy),
                            visible_text,
                            font=font,
                            fill=(*outline_color, text_alpha),
                        )

            # Draw main text
            draw.text(
                (text_x, text_y),
                visible_text,
                font=font,
                fill=text_color_with_alpha,
            )

        return np.array(overlay)

    # Create overlay clip with proper alpha handling
    def make_frame_rgb(t):
        rgba = make_overlay_frame(t)
        return rgba[:, :, :3]

    def make_mask_frame(t):
        rgba = make_overlay_frame(t)
        return rgba[:, :, 3] / 255.0

    overlay_clip = _VideoClip(make_frame_rgb, duration=dur)
    mask_clip = _VideoClip(make_mask_frame, duration=dur, is_mask=True)
    overlay_clip = overlay_clip.with_mask(mask_clip)

    # Composite
    result = CompositeVideoClip([clip, overlay_clip], size=(w, h))
    return result.with_duration(dur)

"""Map highlight effect with slime green location markers.

Displays a bright slime green rectangular highlight on a map location
with animated text label appearing letter-by-letter, then fading to
realistic sepia tone.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip
from utils.logger import setup_logger

from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)

# Brand colors
SLIME_GREEN = (0, 255, 0)
SEPIA_TONE = (112, 66, 20)  # Sepia overlay color


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
    """Apply slime green map highlight with letter-by-letter text.

    Creates a bright green rectangular highlight box around a map location
    with the location name appearing letter-by-letter, then fades to a
    realistic sepia/normal tone.

    :param clip: Input video clip to apply highlight to.
    :param location_name: Name of location to display (e.g., "EGYPT").
    :param box_position: Normalized (x, y, w, h) coordinates for highlight box.
    :param cps: Characters per second for letter-by-letter animation.
    :param highlight_duration: Duration in seconds to hold bright green.
    :param fade_to_normal_duration: Duration in seconds to fade to sepia.
    :param box_thickness: Thickness of the highlight box border in pixels.
    :param fontsize: Font size for location name.
    :param font_path: Path to Century font file (tries system fonts if None).
    :param text_position: Where to place text: "center", "top", "bottom".
    :param glow_intensity: Intensity of glow effect (0-1).
    :return: Video clip with map highlight effect applied.
    """
    w, h = clip.size
    dur = float(clip.duration or 1.0)

    # Convert normalized box coords to pixels
    bx, by, bw, bh = box_position
    box_x = int(bx * w)
    box_y = int(by * h)
    box_w = int(bw * w)
    box_h = int(bh * h)

    # Load Century font or fallback
    try:
        if font_path:
            font = ImageFont.truetype(font_path, fontsize)
        else:
            # Try common Century font names
            for font_name in ["CENTURY.TTF", "century.ttf", "Century.ttf", "arial.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, fontsize)
                    logger.debug(f"[map_highlight] Loaded font: {font_name}")
                    break
                except Exception:
                    continue
            else:
                font = ImageFont.load_default()
                logger.warning("[map_highlight] Century font not found, using default")
    except Exception as e:
        logger.warning(f"[map_highlight] Font loading failed: {e}, using default")
        font = ImageFont.load_default()

    # Calculate text typing duration
    text_length = len(location_name)
    typing_duration = text_length / max(1, cps)
    total_effect_duration = highlight_duration + fade_to_normal_duration

    # Pre-calculate text position
    tmp_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    try:
        text_bbox = tmp_draw.textbbox((0, 0), location_name, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        text_w = len(location_name) * (fontsize // 2)
        text_h = fontsize

    # Position text relative to box
    if text_position == "center":
        text_x = box_x + (box_w - text_w) // 2
        text_y = box_y + (box_h - text_h) // 2
    elif text_position == "top":
        text_x = box_x + (box_w - text_w) // 2
        text_y = max(10, box_y - text_h - 20)
    else:  # bottom
        text_x = box_x + (box_w - text_w) // 2
        text_y = min(h - text_h - 10, box_y + box_h + 20)

    # Ensure text stays in bounds
    text_x = max(10, min(w - text_w - 10, text_x))
    text_y = max(10, min(h - text_h - 10, text_y))

    def make_overlay_frame(t: float):
        """Generate overlay frame with highlight box and text."""
        # Create transparent overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate animation progress
        if t < total_effect_duration:
            # Phase 1: Bright green highlight (0 to highlight_duration)
            # Phase 2: Fade to sepia (highlight_duration to total_effect_duration)
            if t < highlight_duration:
                # Bright green phase
                green_intensity = 1.0
                sepia_intensity = 0.0
            else:
                # Fade to sepia phase
                fade_progress = (t - highlight_duration) / fade_to_normal_duration
                fade_progress = ease_in_out_cubic(fade_progress)
                green_intensity = 1.0 - fade_progress
                sepia_intensity = fade_progress * 0.4  # Subtle sepia

            # Calculate colors
            if green_intensity > 0:
                box_color = tuple(int(c * green_intensity) for c in SLIME_GREEN)
                text_color = SLIME_GREEN
                box_alpha = int(255 * green_intensity)
            else:
                box_color = SEPIA_TONE
                text_color = SEPIA_TONE
                box_alpha = int(255 * sepia_intensity)

            # Draw highlight box with glow effect
            if glow_intensity > 0 and green_intensity > 0.5:
                # Create glow layer
                glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                glow_draw = ImageDraw.Draw(glow)
                glow_thickness = int(box_thickness * 2.5)
                for i in range(3):
                    glow_alpha = int(box_alpha * glow_intensity * 0.3 / (i + 1))
                    glow_color = (*SLIME_GREEN, glow_alpha)
                    offset = (glow_thickness + i * 4)
                    glow_draw.rectangle(
                        [
                            box_x - offset,
                            box_y - offset,
                            box_x + box_w + offset,
                            box_y + box_h + offset,
                        ],
                        outline=glow_color,
                        width=glow_thickness,
                    )
                # Apply blur
                glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
                overlay = Image.alpha_composite(overlay, glow)

            # Draw main box
            box_color_with_alpha = (*box_color, box_alpha)
            draw.rectangle(
                [box_x, box_y, box_x + box_w, box_y + box_h],
                outline=box_color_with_alpha,
                width=box_thickness,
            )

            # Draw letter-by-letter text
            if t < typing_duration:
                chars_shown = int(t * cps)
            else:
                chars_shown = text_length

            visible_text = location_name[:chars_shown]
            if visible_text:
                # Text with outline for visibility
                text_alpha = int(255 * (green_intensity if green_intensity > 0 else sepia_intensity * 2))
                text_color_with_alpha = (*text_color, text_alpha)

                # Draw text outline (black for contrast)
                outline_width = 3
                for dx in [-outline_width, 0, outline_width]:
                    for dy in [-outline_width, 0, outline_width]:
                        if dx != 0 or dy != 0:
                            draw.text(
                                (text_x + dx, text_y + dy),
                                visible_text,
                                font=font,
                                fill=(0, 0, 0, text_alpha),
                            )

                # Draw main text
                draw.text(
                    (text_x, text_y),
                    visible_text,
                    font=font,
                    fill=text_color_with_alpha,
                )

        return np.array(overlay)

    # Create overlay clip
    overlay_clip = _VideoClip(make_overlay_frame, duration=dur)

    # Composite
    result = CompositeVideoClip([clip, overlay_clip], size=(w, h))
    return result.with_duration(dur)

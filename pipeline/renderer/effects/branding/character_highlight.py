"""Character highlight effect with cyan glow and name labels.

Highlights a character/person with a bright cyan/green glow around their
silhouette with an animated name label appearing letter-by-letter.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip
from utils.logger import setup_logger

from ..easing import ease_in_out_cubic
from ..coords import denorm_bbox

logger = setup_logger(__name__)

# Brand colors for character highlights
CYAN_GLOW = (0, 255, 255)
GREEN_GLOW = (0, 255, 200)


def apply_character_highlight(
    clip: _VideoClip,
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
) -> _VideoClip:
    """Apply character highlight with cyan glow and name label.

    Creates a glowing outline around the detected subject (character) with
    their name appearing letter-by-letter. Glow intensity animates through
    multiple stages: bright → medium → subtle.

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
    dur = float(clip.duration or 1.0)

    # Auto-detect subject if no bbox/mask provided
    if bbox is None and mask is None:
        logger.warning("[character_highlight] No bbox or mask provided, effect will be minimal")
        # Could trigger subject detection here if needed
        # For now, create a centered default bbox
        bbox = (0.3, 0.2, 0.4, 0.6)

    # Convert normalized bbox to pixels
    if bbox:
        bx, by, bw, bh = denorm_bbox(bbox, (w, h))
    else:
        # Fallback if only mask provided
        bx, by, bw, bh = (w // 4, h // 4, w // 2, h // 2)

    # Load Century font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, fontsize)
        else:
            for font_name in ["CENTURY.TTF", "century.ttf", "Century.ttf", "arial.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, fontsize)
                    logger.debug(f"[character_highlight] Loaded font: {font_name}")
                    break
                except Exception:
                    continue
            else:
                font = ImageFont.load_default()
                logger.warning("[character_highlight] Century font not found, using default")
    except Exception as e:
        logger.warning(f"[character_highlight] Font loading failed: {e}, using default")
        font = ImageFont.load_default()

    # Calculate text metrics
    tmp_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    try:
        text_bbox = tmp_draw.textbbox((0, 0), character_name, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        text_w = len(character_name) * (fontsize // 2)
        text_h = fontsize

    # Position label relative to subject
    text_x = bx + (bw - text_w) // 2
    if label_position == "top":
        text_y = max(10, by - text_h - label_offset)
    else:  # bottom
        text_y = min(h - text_h - 10, by + bh + label_offset)

    # Ensure text stays in bounds
    text_x = max(10, min(w - text_w - 10, text_x))
    text_y = max(10, min(h - text_h - 10, text_y))

    # Calculate typing duration
    text_length = len(character_name)
    typing_duration = text_length / max(1, cps)

    # Calculate stage transitions
    bright_end = stage_durations[0]
    medium_end = bright_end + stage_durations[1]
    subtle_end = medium_end + stage_durations[2]

    def make_overlay_frame(t: float):
        """Generate overlay frame with glow and label."""
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Determine current glow stage
        if t < bright_end:
            # Stage 1: Bright
            progress = t / bright_end if bright_end > 0 else 1.0
            intensity = glow_stages[0] * ease_in_out_cubic(progress)
        elif t < medium_end:
            # Stage 2: Medium
            progress = (t - bright_end) / (medium_end - bright_end) if (medium_end - bright_end) > 0 else 1.0
            intensity = glow_stages[0] + (glow_stages[1] - glow_stages[0]) * ease_in_out_cubic(progress)
        elif t < subtle_end:
            # Stage 3: Subtle
            progress = (t - medium_end) / (subtle_end - medium_end) if (subtle_end - medium_end) > 0 else 1.0
            intensity = glow_stages[1] + (glow_stages[2] - glow_stages[1]) * ease_in_out_cubic(progress)
        else:
            # Hold subtle
            intensity = glow_stages[2]

        # Draw glow around subject bbox (or use mask if available)
        if mask is not None:
            # Use mask for precise glow
            mask_img = Image.fromarray(mask).convert("L")
            mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)

            # Create glow from mask
            glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            glow_pixels = np.array(mask_img)
            glow_alpha = (glow_pixels * intensity).astype(np.uint8)

            # Apply glow color
            for i in range(3):
                glow_arr = np.zeros((h, w, 4), dtype=np.uint8)
                glow_arr[:, :, 0] = glow_color[0]
                glow_arr[:, :, 1] = glow_color[1]
                glow_arr[:, :, 2] = glow_color[2]
                glow_arr[:, :, 3] = glow_alpha // (i + 1)

                glow_img = Image.fromarray(glow_arr, "RGBA")
                glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=10 + i * 5))
                overlay = Image.alpha_composite(overlay, glow_img)
        else:
            # Fallback: glow around bbox
            glow_alpha = int(255 * intensity)
            glow_thickness = 12

            # Multi-layer glow for depth
            for i in range(3):
                layer_alpha = glow_alpha // (i + 1)
                layer_color = (*glow_color, layer_alpha)
                offset = (i + 1) * 8

                draw.rectangle(
                    [
                        bx - offset,
                        by - offset,
                        bx + bw + offset,
                        by + bh + offset,
                    ],
                    outline=layer_color,
                    width=glow_thickness,
                )

            # Apply blur to overlay
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=12))

        # Draw letter-by-letter name label
        if t < typing_duration:
            chars_shown = int(t * cps)
        else:
            chars_shown = text_length

        visible_text = character_name[:chars_shown]
        if visible_text:
            # Calculate text alpha based on intensity
            text_alpha = int(255 * min(1.0, intensity + 0.3))
            text_color_with_alpha = (*glow_color, text_alpha)

            # Draw text outline for visibility
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

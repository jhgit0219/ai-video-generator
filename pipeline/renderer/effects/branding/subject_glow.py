"""Subject glow effect - creates glowing outline around detected subject.

Atomic effect for highlighting subjects with multi-stage animated glow.
Supports both mask-based (precise) and bbox-based (fallback) rendering.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from moviepy import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip

from utils.logger import setup_logger
from ..coords import denorm_bbox
from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)


def apply_subject_glow(
    clip: _VideoClip,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[np.ndarray] = None,
    glow_color: Tuple[int, int, int] = (0, 255, 255),
    glow_stages: Tuple[float, float, float] = (1.0, 0.6, 0.3),
    stage_durations: Tuple[float, float, float] = (0.5, 1.0, 1.5),
) -> _VideoClip:
    """Apply multi-stage animated glow around subject.

    Creates a glowing effect around the detected subject with intensity
    that animates through three stages: bright → medium → subtle.

    :param clip: Input video clip to apply glow to.
    :param bbox: Optional normalized (x, y, w, h) subject bounding box.
    :param mask: Optional subject segmentation mask (from YOLO+CLIP).
    :param glow_color: RGB color for glow effect.
    :param glow_stages: Tuple of 3 intensity levels (0-1) for bright/medium/subtle.
    :param stage_durations: Duration in seconds for each intensity stage.
    :return: Video clip with subject glow effect applied.
    """
    w, h = clip.size
    dur = float(clip.duration or 1.0)

    # Fallback bbox if none provided
    if bbox is None and mask is None:
        logger.warning("[subject_glow] No bbox or mask provided, using centered default")
        bbox = (0.3, 0.2, 0.4, 0.6)

    # Convert normalized bbox to pixels
    if bbox:
        bx, by, bw, bh = denorm_bbox(bbox, (w, h))
    else:
        # Fallback if only mask provided
        bx, by, bw, bh = (w // 4, h // 4, w // 2, h // 2)

    # Calculate stage transitions
    bright_end = stage_durations[0]
    medium_end = bright_end + stage_durations[1]
    subtle_end = medium_end + stage_durations[2]

    def make_overlay_frame(t: float):
        """Generate glow overlay frame."""
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

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

        # Draw glow around subject
        if mask is not None:
            # Use mask for precise glow
            mask_img = Image.fromarray(mask).convert("L")
            mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)

            # Create glow from mask with multiple layers
            glow_pixels = np.array(mask_img)
            glow_alpha = (glow_pixels * intensity).astype(np.uint8)

            # Apply glow color with multiple blur layers for depth
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
            # Fallback: circular/oval glow for portrait-style framing
            draw = ImageDraw.Draw(overlay)
            glow_alpha = int(255 * intensity)

            # Create circular glow layers - portrait style
            for i in range(4):
                layer_alpha = glow_alpha // (i + 1)
                layer_color = (*glow_color, layer_alpha)
                offset = (i + 1) * 10

                # Draw oval/ellipse around subject for portrait framing
                draw.ellipse(
                    [
                        bx - offset,
                        by - offset,
                        bx + bw + offset,
                        by + bh + offset,
                    ],
                    outline=layer_color,
                    width=8,
                )

            # Apply blur to overlay for soft glow
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))

            # Add inner portrait frame (decorative border)
            frame_alpha = int(255 * intensity * 0.6)
            frame_color = (*glow_color, frame_alpha)
            draw.ellipse(
                [bx - 5, by - 5, bx + bw + 5, by + bh + 5],
                outline=frame_color,
                width=3,
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

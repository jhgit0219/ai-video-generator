"""
Zoom then Panel Effect: Combines temporal zoom with delayed panel text overlay.

This effect creates a cinematic sequence:
1. Zooms into a subject over a specified duration
2. After the zoom completes, displays a panel with typewriter text
3. The panel is fixed to the video frame (unaffected by zoom)

Perfect for highlighting subjects with contextual information.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip
from moviepy import concatenate_videoclips
from utils.logger import setup_logger

from .zoom_temporal import apply_temporal_zoom
from .overlay_text import apply_paneled_text

logger = setup_logger(__name__)


def apply_zoom_then_panel(
    clip: VideoClip,
    text: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    zoom_duration: float = 3.0,
    zoom_scale: float = 1.5,
    zoom_start_scale: float = 1.0,
    zoom_ease: str = "easeInOut",
    anchor_point: Optional[Tuple[float, float]] = None,
    panel_side: str = "right",
    panel_color: Tuple[int, int, int] = (0, 0, 0),
    panel_opacity: float = 0.7,
    border_color: Tuple[int, int, int] = (57, 255, 20),
    border_thickness: int = 4,
    fontsize: int = 56,
    font: Optional[str] = None,
    margin_px: int = 28,
    panel_animate_in: float = 0.4,
    panel_animate_out: float = 0.0,
    typing_speed: int = 12,
    max_panel_width_frac: float = 0.6,
    min_panel_width_px: int = 560,
    min_panel_height_px: int = 160,
) -> VideoClip:
    """
    Apply zoom-to-subject effect followed by a delayed panel text overlay.

    The zoom happens over zoom_duration, then the panel appears and remains
    for the rest of the clip. The panel is fixed to the video frame coordinates.

    Args:
        clip: Input video clip
        text: Text to display in the panel
        bbox: Optional normalized (x, y, w, h) subject bbox to position panel near.
              If None, panel will be positioned at default location.

        Zoom parameters:
        zoom_duration: Duration of zoom animation in seconds (default: 3.0)
        zoom_scale: Target zoom scale (default: 1.5 = 150%)
        zoom_start_scale: Starting zoom scale (default: 1.0 = 100%)
        zoom_ease: Easing function name (default: "easeInOut")
                   Options: "linear", "easeIn", "easeOut", "easeInOut",
                           "easeInBack", "easeOutBack", etc.
        anchor_point: (x, y) normalized coordinates to zoom towards.
                     If None, uses center of bbox or clip center.

        Panel parameters:
        panel_side: Side to position panel relative to subject ("left" or "right")
        panel_color: RGB color tuple for panel background (default: black)
        panel_opacity: Panel background opacity 0.0-1.0 (default: 0.7)
        border_color: RGB color for panel border (default: neon green)
        border_thickness: Border thickness in pixels (default: 4)
        fontsize: Font size in points (default: 56)
        font: Optional path to TTF font file (default: Arial)
        margin_px: Padding inside panel in pixels (default: 28)
        panel_animate_in: Panel slide-in animation duration in seconds (default: 0.4)
        panel_animate_out: Panel slide-out animation duration in seconds (default: 0.0)
        typing_speed: Characters per second for typewriter effect (default: 12)
        max_panel_width_frac: Maximum panel width as fraction of frame (default: 0.6)
        min_panel_width_px: Minimum panel width in pixels (default: 560)
        min_panel_height_px: Minimum panel height in pixels (default: 160)

    Returns:
        VideoClip with zoom and panel effects applied

    Example usage (programmatic):
        ```python
        segment.custom_effects = [
            {
                "name": "zoom_then_panel",
                "params": {
                    "text": "Herodotus, c. 484–425 BC",
                    "bbox": (0.35, 0.2, 0.25, 0.45),
                    "zoom_duration": 3.0,
                    "zoom_scale": 1.5,
                    "panel_side": "right",
                    "typing_speed": 15
                }
            }
        ]
        ```

    Example usage (LLM plan):
        ```json
        {
            "tools": [
                {
                    "name": "zoom_then_panel",
                    "params": {
                        "text": "Marcus Aurelius",
                        "bbox": [0.3, 0.15, 0.4, 0.6],
                        "zoom_duration": 2.5,
                        "zoom_scale": 1.8,
                        "border_color": [255, 215, 0]
                    }
                }
            ]
        }
        ```
    """
    total_duration = float(clip.duration or zoom_duration)
    zoom_dur = min(float(zoom_duration), total_duration)

    # Determine anchor point for zoom
    if anchor_point is None and bbox is not None:
        # Use center of subject bbox
        x, y, w, h = bbox
        anchor_point = (x + w / 2, y + h / 2)
        logger.debug(f"[zoom_then_panel] Calculated anchor from bbox: {anchor_point}")
    elif anchor_point is None:
        # Default to center of frame
        anchor_point = (0.5, 0.5)
        logger.debug(f"[zoom_then_panel] Using default anchor: {anchor_point}")

    # Validate and clamp anchor point
    ax, ay = anchor_point
    anchor_point = (max(0.0, min(1.0, float(ax))), max(0.0, min(1.0, float(ay))))

    logger.info(f"[zoom_then_panel] Applying zoom (duration={zoom_dur:.2f}s, scale={zoom_scale:.2f}x) + panel (text='{text}')")

    # If zoom duration covers the entire clip, just zoom
    if zoom_dur >= total_duration - 0.1:
        # Apply zoom for entire duration
        zoomed = apply_temporal_zoom(
            clip,
            scale=zoom_scale,
            start_scale=zoom_start_scale,
            ease=zoom_ease,
            anchor_point=anchor_point,
        )

        # Add panel overlay starting at zoom_duration
        if total_duration > zoom_dur:
            panel_start = zoom_dur
            panel_dur = total_duration - panel_start
        else:
            # Start panel at 90% of clip duration
            panel_start = total_duration * 0.9
            panel_dur = total_duration - panel_start

        if panel_dur > 0.1:
            enhanced = apply_paneled_text(
                zoomed,
                text=text,
                bbox=bbox,
                side=panel_side,
                panel_color=panel_color,
                panel_opacity=panel_opacity,
                border_color=border_color,
                border_thickness=border_thickness,
                fontsize=fontsize,
                font=font,
                margin_px=margin_px,
                animate_in=panel_animate_in,
                animate_out=panel_animate_out,
                max_panel_width_frac=max_panel_width_frac,
                min_panel_width_px=min_panel_width_px,
                min_panel_height_px=min_panel_height_px,
                start_time=panel_start,
                duration=panel_dur,
                cps=typing_speed,
            )
            return enhanced
        else:
            return zoomed

    # Split into zoom phase and static zoomed phase
    # Phase 1: Zoom animation (0 to zoom_duration)
    clip_zoom = clip.subclipped(0, zoom_dur)
    clip_zoom = apply_temporal_zoom(
        clip_zoom,
        scale=zoom_scale,
        start_scale=zoom_start_scale,
        ease=zoom_ease,
        anchor_point=anchor_point,
    )

    # Phase 2: Static zoomed state (zoom_duration to end)
    if total_duration > zoom_dur:
        clip_static = clip.subclipped(zoom_dur, total_duration)

        # Apply the final zoom state to static portion
        w, h = clip_static.size
        crop_w = int(w / zoom_scale)
        crop_h = int(h / zoom_scale)
        ax_px = int(anchor_point[0] * w)
        ay_px = int(anchor_point[1] * h)
        x1 = max(0, ax_px - crop_w // 2)
        y1 = max(0, ay_px - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        def crop_to_final_zoom(get_frame, t):
            frame = get_frame(t)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            sub = frame[y1:y2, x1:x2]
            if sub.size == 0:
                return frame
            img_resized = Image.fromarray(sub).resize((w, h), Image.Resampling.LANCZOS)
            return np.array(img_resized)

        clip_static = clip_static.transform(crop_to_final_zoom)

        # Concatenate zoom + static phases
        enhanced = concatenate_videoclips([clip_zoom, clip_static], method="compose")
    else:
        enhanced = clip_zoom

    # Add panel overlay starting at zoom_duration
    panel_start = zoom_dur
    panel_dur = total_duration - panel_start

    if panel_dur > 0.1:
        enhanced = apply_paneled_text(
            enhanced,
            text=text,
            bbox=bbox,
            side=panel_side,
            panel_color=panel_color,
            panel_opacity=panel_opacity,
            border_color=border_color,
            border_thickness=border_thickness,
            fontsize=fontsize,
            font=font,
            margin_px=margin_px,
            animate_in=panel_animate_in,
            animate_out=panel_animate_out,
            max_panel_width_frac=max_panel_width_frac,
            min_panel_width_px=min_panel_width_px,
            min_panel_height_px=min_panel_height_px,
            start_time=panel_start,
            duration=panel_dur,
            cps=typing_speed,
        )

    logger.info(f"[zoom_then_panel] Effect complete (total_duration={total_duration:.2f}s)")
    return enhanced

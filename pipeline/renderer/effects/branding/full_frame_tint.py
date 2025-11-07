"""Full-frame color tint overlay effect.

Applies a subtle color wash over the entire frame, similar to neon_overlay
but designed for branding colors (slime green, sepia, etc.).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from moviepy import ColorClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip

from utils.logger import setup_logger
from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)


def apply_full_frame_tint(
    clip: _VideoClip,
    color: Tuple[int, int, int] = (0, 255, 0),
    opacity: float = 0.3,
    animate_in: float = 0.4,
    animate_out: float = 0.0,
) -> _VideoClip:
    """Apply full-frame color tint overlay.

    :param clip: Input video clip.
    :param color: RGB color tuple (default: bright green).
    :param opacity: Tint opacity 0-1 (default: 0.3 for subtle wash).
    :param animate_in: Fade-in duration in seconds (default: 0.4).
    :param animate_out: Fade-out duration in seconds (default: 0.0).
    :return: Video clip with full-frame tint overlay.
    """
    w, h = clip.size
    base_opacity = max(0.0, min(1.0, opacity))

    # Ensure duration is a float, not a function
    dur = float(clip.duration) if clip.duration is not None else (animate_in + animate_out) or 1.0
    overlay = ColorClip(size=(w, h), color=color, duration=dur)

    total = dur
    ai = max(0.0, float(animate_in))
    ao = max(0.0, float(animate_out))

    def alpha_t(t: float) -> float:
        """Calculate opacity at time t with fade-in/out."""
        a = base_opacity

        # Fade in
        if ai > 0 and t < ai:
            a = base_opacity * ease_in_out_cubic(min(1.0, max(0.0, t / ai)))

        # Fade out
        if ao > 0 and t > (total - ao):
            rem = max(0.0, total - t)
            a = base_opacity * ease_in_out_cubic(rem / ao)

        return max(0.0, min(1.0, a))

    # Create custom mask for time-varying opacity
    # (with_opacity doesn't support lambda functions in this MoviePy version)
    def make_mask_frame(t):
        # Mask must be 2D array (h, w) with values 0-1
        opacity_value = alpha_t(t)
        return np.full((h, w), opacity_value, dtype=np.float32)

    mask_clip = _VideoClip(make_mask_frame, duration=dur, is_mask=True)
    overlay = overlay.with_mask(mask_clip)

    return CompositeVideoClip([clip, overlay]).with_duration(dur)

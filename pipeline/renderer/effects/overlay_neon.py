from __future__ import annotations

from typing import Tuple
import numpy as np
from moviepy import ColorClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from utils.logger import setup_logger

from .easing import ease_in_out_cubic

logger = setup_logger(__name__)


def apply_neon_overlay(
    clip: VideoClip,
    color: Tuple[int, int, int] = (57, 255, 20),
    opacity: float = 0.18,
    animate_in: float = 0.0,
    animate_out: float = 0.0,
    blend_mode: str = "normal",  # normal | add | screen
) -> VideoClip:
    """Overlay a neon color wash over the entire frame."""
    w, h = clip.size
    base_opacity = max(0.0, min(1.0, opacity))
    overlay = ColorClip(size=(w, h), color=color, duration=clip.duration)

    total = clip.duration or (animate_in + animate_out) or 1.0
    ai = max(0.0, float(animate_in))
    ao = max(0.0, float(animate_out))

    def alpha_t(t: float) -> float:
        a = base_opacity
        if ai > 0 and t < ai:
            a = base_opacity * (ease_in_out_cubic(min(1.0, max(0.0, t / ai))))
        if ao > 0 and t > (total - ao):
            rem = max(0.0, total - t)
            a = base_opacity * ease_in_out_cubic(rem / ao)
        return max(0.0, min(1.0, a))

    bm = (blend_mode or "normal").lower()
    if bm == "normal":
        overlay = overlay.with_opacity(lambda t: alpha_t(t))
        return CompositeVideoClip([clip, overlay]).with_duration(clip.duration)

    col = np.array(color, dtype=np.float32)

    def blend_tf(get_frame, t):
        frame = get_frame(t).astype(np.float32)
        a = float(alpha_t(t))
        if bm == "add":
            out = frame + col * (255.0 / 255.0) * (a * 0.8)
        else:  # screen
            F = frame / 255.0
            C = (col / 255.0) * a
            out = (1.0 - (1.0 - F) * (1.0 - C)) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    return clip.transform(blend_tf)

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image
try:
    from moviepy import ImageSequenceClip  # type: ignore
except Exception:
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
    except Exception:
        ImageSequenceClip = None  # type: ignore
from moviepy.video.VideoClip import VideoClip
from utils.logger import setup_logger

from .easing import ease_curve

logger = setup_logger(__name__)


def apply_temporal_zoom(
    clip: VideoClip,
    scale: float = 1.2,
    ease: str = "easeInOut",
    start_scale: float = 1.0,
    method: str = "transform",
    anchor_point: Optional[Tuple[float, float]] = None,
) -> VideoClip:
    """Time-animated digital zoom while keeping output size constant.
    If method == 'sequence', builds frames explicitly to avoid time flattening.
    """
    w, h = clip.size
    try:
        s0 = max(1.0, float(start_scale))
        s1 = max(1.0, float(scale))
    except Exception:
        s0, s1 = 1.0, 1.2
    dur = max(1e-3, float(clip.duration or 1.0))

    if anchor_point is not None and isinstance(anchor_point, (tuple, list)) and len(anchor_point) == 2:
        try:
            ax = float(anchor_point[0]); ay = float(anchor_point[1])
            cx_px = int(round(max(0.0, min(1.0, ax)) * w))
            cy_px = int(round(max(0.0, min(1.0, ay)) * h))
        except Exception:
            cx_px, cy_px = w // 2, h // 2
    else:
        cx_px, cy_px = w // 2, h // 2

    def z_at(t: float) -> float:
        p = max(0.0, min(1.0, t / dur))
        return s0 + (s1 - s0) * ease_curve(ease, p)

    def tf(get_frame, t):
        z = max(1.0, float(z_at(t)))
        frame = get_frame(t)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Use original integer rounding to preserve aspect ratio
        crop_w = int(round(w / z))
        crop_h = int(round(h / z))
        cx, cy = cx_px, cy_px
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        sub = frame[y1:y2, x1:x2]
        if sub.size == 0:
            sub = frame
        img = Image.fromarray(sub).resize((w, h), Image.Resampling.LANCZOS)
        try:
            if abs(t - 0.0) < 1e-3 or abs(t - dur/2) < 5e-3 or abs(t - dur) < 1e-3:
                logger.debug(f"[effects] temporal_zoom t={t:.3f}s z={z:.3f}")
        except Exception:
            pass
        return np.array(img)

    if (method or "transform").lower() == "sequence" and ImageSequenceClip is not None:
        try:
            try:
                from config import FPS as _CFG_FPS
                fps = int(_CFG_FPS)
            except Exception:
                fps = 24
            N = max(1, int(round(dur * fps)))
            frames = []
            for i in range(N):
                t = min((i / max(1, fps)), max(0.0, dur - 1e-6))
                frames.append(tf(clip.get_frame, t))
            seq = ImageSequenceClip(frames, fps=fps)  # type: ignore
            return seq.with_duration(clip.duration)
        except Exception:
            pass
    return clip.transform(tf)

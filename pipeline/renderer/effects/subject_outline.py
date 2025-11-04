from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from moviepy.video.VideoClip import VideoClip
from utils.logger import setup_logger

from .coords import denorm_bbox
from .subject_import import import_subject_detection

logger = setup_logger(__name__)


def apply_subject_outline(
    clip: VideoClip,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[Any] = None,
    color: Tuple[int, int, int] = (57, 255, 20),
    thickness: int = 6,
    offset: int = 8,
    glow: int = 12,
    opacity: float = 1.0,
    animate_in: float = 0.3,
    pulse: Optional[bool] = None,
    pulse_rate: float = 1.0,
    pulse_min: float = 0.6,
    pulse_max: float = 1.0,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
) -> VideoClip:
    w, h = clip.size
    _provided_mask = mask
    mask = _provided_mask

    try:
        from config import USE_SEG_MASK_FOR_OUTLINE as _USE_SEG_MASK_FOR_OUTLINE, SUBJECT_OUTLINE_DEFAULT_PULSE as _SUB_PULSE
        use_mask = bool(_USE_SEG_MASK_FOR_OUTLINE)
        if pulse is None:
            pulse = bool(_SUB_PULSE)
    except Exception:
        use_mask = True
        if pulse is None:
            pulse = True

    if mask is None or bbox is None:
        try:
            frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.3))
            sd = import_subject_detection()
            ds = sd.get("detect_subject_shape")
            if callable(ds):
                shape = ds(frame, target=target, prefer=prefer, nth=nth)  # type: ignore
                if use_mask and mask is None:
                    mask = shape.get("mask")
                if bbox is None:
                    bbox = shape.get("bbox")
        except Exception:
            pass

    try:
        if mask is not None:
            nz = int((np.array(mask) > 0).sum())
            src = "provided" if _provided_mask is not None else "detected"
            logger.debug(f"[effects] subject_outline using {src} mask (area_px={nz})")
        else:
            logger.debug("[effects] subject_outline using rectangle fallback (no mask)")
    except Exception:
        pass

    if bbox is None:
        bbox = (0.3, 0.3, 0.4, 0.4)
    x, y, bw, bh = denorm_bbox(bbox, (w, h))

    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    if mask is not None:
        try:
            m = mask
            if getattr(m, "dtype", None) is not np.uint8:
                m = (np.array(m) > 0).astype(np.uint8) * 255
            mask_img = Image.fromarray(m, mode="L")
            t_px = max(1, int(thickness))
            off_px = max(0, int(offset))

            if off_px > 0:
                k_gap = max(1, 2 * off_px + 1)
                k_full = max(k_gap + 2 * t_px, 2 * (off_px + t_px) + 1)
                dil_gap = mask_img.filter(ImageFilter.MaxFilter(k_gap))
                dil_full = mask_img.filter(ImageFilter.MaxFilter(k_full))
                edge = ImageChops.subtract(dil_full, dil_gap)
            else:
                sz = max(3, (t_px | 1))
                maxf = mask_img.filter(ImageFilter.MaxFilter(sz))
                minf = mask_img.filter(ImageFilter.MinFilter(sz))
                edge = ImageChops.subtract(maxf, minf)

            edge_np = np.array(edge)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = color[0]
            rgba[..., 1] = color[1]
            rgba[..., 2] = color[2]
            rgba[..., 3] = np.clip(edge_np, 0, 255)
            base = Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA"))
        except Exception:
            pass

    if mask is None:
        x1, y1 = x, y
        x2, y2 = x + bw, y + bh
        t_px = max(1, int(thickness))
        off_px = max(0, int(offset))
        if off_px > 0:
            ring = Image.new("L", (w, h), 0)
            dr = ImageDraw.Draw(ring)
            dr.rectangle([x1 - (off_px + t_px), y1 - (off_px + t_px), x2 + (off_px + t_px), y2 + (off_px + t_px)], fill=255)
            dr.rectangle([x1 - off_px, y1 - off_px, x2 + off_px, y2 + off_px], fill=0)
            edge_np = np.array(ring)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = color[0]
            rgba[..., 1] = color[1]
            rgba[..., 2] = color[2]
            rgba[..., 3] = np.clip(edge_np, 0, 255)
            base = Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA"))
        else:
            drw = ImageDraw.Draw(base)
            for t in range(t_px, 0, -max(1, t_px // 3)):
                drw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(*color, 255))

    if glow > 0:
        try:
            blurred = base.filter(ImageFilter.GaussianBlur(radius=int(glow)))
            base = Image.alpha_composite(blurred, base)
        except Exception:
            pass

    outline_rgba = np.array(base)

    base_op = max(0.0, min(1.0, float(opacity)))
    ai = float(animate_in)

    def op_t(t: float) -> float:
        a = base_op
        if ai > 0 and t < ai:
            from .easing import ease_out_back
            a = base_op * ease_out_back(t / ai)
        if pulse:
            lo, hi = max(0.0, pulse_min), max(0.0, pulse_max)
            if hi < lo:
                hi = lo
            amp = (hi - lo) / 2.0
            mid = lo + amp
            import math
            a *= (mid + amp * (1.0 + math.sin(2 * math.pi * pulse_rate * t)) * 0.5)
        return max(0.0, min(1.0, a))

    def overlay_tf(get_frame, t):
        frame = get_frame(t)
        import numpy as _np
        if frame.dtype != _np.uint8:
            frame = frame.astype(_np.uint8)
        a = op_t(t)
        if a <= 0.0:
            return frame
        ol = outline_rgba
        if ol.shape[2] == 4:
            alpha = (ol[..., 3].astype(_np.float32) / 255.0) * float(a)
            if _np.max(alpha) <= 0:
                return frame
            rgb = ol[..., :3].astype(_np.float32)
            base_arr = frame.astype(_np.float32)
            alpha3 = _np.stack([alpha, alpha, alpha], axis=-1)
            out = rgb * alpha3 + base_arr * (1.0 - alpha3)
            return _np.clip(out, 0, 255).astype(_np.uint8)
        else:
            rgb = ol.astype(_np.float32)
            base_arr = frame.astype(_np.float32)
            out = rgb * float(a) + base_arr * (1.0 - float(a))
            return _np.clip(out, 0, 255).astype(_np.uint8)

    return clip.transform(overlay_tf)

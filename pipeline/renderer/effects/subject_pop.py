from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
from moviepy import ColorClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from utils.logger import setup_logger

from .coords import denorm_bbox
from .easing import ease_out_back
from .subject_import import import_subject_detection

logger = setup_logger(__name__)


def apply_subject_pop(
    clip: VideoClip,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[Any] = None,
    scale: float = 1.15,
    pop_duration: float = 0.4,
    hold_after: float = 0.0,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    shadow_opacity: float = 0.35,
    shadow_offset_px: Tuple[int, int] = (8, 8),
    draw_outline: bool = False,
    outline_color: Tuple[int, int, int] = (57, 255, 20),
    outline_thickness: int = 6,
    outline_glow: int = 8,
    outline_opacity: float = 1.0,
    force_silhouette: bool = False,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
    anchor_feature: str = "auto",
    anchor_point: Optional[Tuple[int, int]] = None,
) -> VideoClip:
    orig_w, orig_h = clip.size
    logger.debug(f"[effects] subject_pop called: bbox={bbox is not None}, mask={mask is not None}, force_silhouette={force_silhouette}, clip_size=({orig_w}, {orig_h})")

    if bbox is None:
        try:
            frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.1))
            sd = import_subject_detection()
            db = sd.get("detect_subject_bbox")
            if callable(db):
                bbox = db(frame, target=target, prefer=prefer, nth=nth)  # type: ignore
            else:
                raise ImportError("detect_subject_bbox not available")
        except Exception:
            bbox = (0.3, 0.3, 0.4, 0.4)
    x, y, bw, bh = denorm_bbox(bbox, (orig_w, orig_h))

    try:
        from config import USE_SEG_MASK_FOR_POP as _USE_SEG_MASK_FOR_POP
        use_mask = bool(_USE_SEG_MASK_FOR_POP)
    except Exception:
        use_mask = True

    if use_mask:
        logger.debug(f"[effects] subject_pop attempting mask-based silhouette path")
        try:
            if mask is None:
                frame0 = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.2))
                sd = import_subject_detection()
                ds = sd.get("detect_subject_shape")
                if not callable(ds):
                    raise ImportError("detect_subject_shape not available")
                shape = ds(frame0, target=target, prefer=prefer, nth=nth)  # type: ignore
                mask = shape.get("mask")
            if mask is not None:
                try:
                    nz = int((np.array(mask) > 0).sum())
                    logger.debug(f"[effects] subject_pop using mask (area_px={nz})")
                except Exception:
                    pass
                detected_anchor = anchor_point
                if detected_anchor is None:
                    # Try adaptive face anchor first
                    sd = import_subject_detection()
                    detect_adaptive = sd.get("detect_adaptive_face_anchor")
                    if callable(detect_adaptive):
                        try:
                            frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.1))
                            detected_anchor = detect_adaptive(frame, target=target, prefer=prefer, nth=nth)
                            if detected_anchor:
                                logger.debug(f"[effects] subject_pop using adaptive face anchor at ({detected_anchor[0]}, {detected_anchor[1]})")
                        except Exception as e:
                            logger.debug(f"[effects] subject_pop adaptive face anchor failed: {e}")

                    # Fallback to mask-based detection if adaptive anchor failed
                    if detected_anchor is None:
                        detect_anchor = sd.get("detect_anchor_feature")
                        if callable(detect_anchor):
                            try:
                                frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.1))
                                abs_bbox = (x, y, x + bw, y + bh) if bbox else None
                                detected_anchor = detect_anchor(frame, subject_mask=mask, subject_bbox=abs_bbox, feature_type=anchor_feature)
                                if not detected_anchor:
                                    logger.debug(f"[effects] subject_pop YOLO anchor detection returned None - skipping effect")
                                    return clip
                            except Exception as e:
                                logger.debug(f"[effects] subject_pop anchor detection failed: {e}")
                                return clip
                        if detected_anchor is None:
                            logger.debug(f"[effects] subject_pop no anchor detected - skipping effect")
                            return clip
                else:
                    logger.debug(f"[effects] subject_pop using pre-computed anchor at ({detected_anchor[0]}, {detected_anchor[1]})")

                ccx, ccy = detected_anchor

                def make_subject(t):
                    fr = clip.get_frame(t)
                    if fr.dtype != np.uint8:
                        fr = fr.astype(np.uint8)
                    cur_h, cur_w = fr.shape[:2]
                    a = (mask if mask.shape[:2] == fr.shape[:2] else np.array(Image.fromarray(mask).resize((cur_w, cur_h), Image.NEAREST)))
                    rgba = np.dstack((fr, a))
                    img = Image.fromarray(rgba, mode="RGBA")
                    if t <= pop_duration:
                        s = 1.0 + (max(1.0, scale) - 1.0) * ease_out_back(t / pop_duration)
                    else:
                        s = max(1.0, scale)
                    sw = int(cur_w * s); sh = int(cur_h * s)
                    img_s = img.resize((sw, sh), Image.Resampling.LANCZOS)
                    frame_center_x = cur_w // 2; frame_center_y = cur_h // 2
                    px = int(frame_center_x - ccx * s); py = int(frame_center_y - ccy * s)
                    canvas = Image.new("RGBA", (cur_w, cur_h), (0, 0, 0, 0))
                    canvas.alpha_composite(img_s, (px, py))
                    return np.array(canvas)

                subj_anim = VideoClip(make_subject, duration=clip.duration)

                if shadow_opacity > 0:
                    from PIL import ImageFilter
                    def make_shadow(t):
                        fr = clip.get_frame(t)
                        cur_h, cur_w = fr.shape[:2]
                        a = (mask if mask.shape[:2] == (cur_h, cur_w) else np.array(Image.fromarray(mask).resize((cur_w, cur_h), Image.NEAREST)))
                        sh_img = Image.new("RGBA", (cur_w, cur_h), (0, 0, 0, 0))
                        alpha = Image.fromarray(a, mode="L").filter(ImageFilter.GaussianBlur(radius=int(max(4, max(bw, bh) * 0.02))))
                        off = Image.new("L", (cur_w, cur_h), 0)
                        off.paste(alpha, (shadow_offset_px[0], shadow_offset_px[1]))
                        rgba = np.zeros((cur_h, cur_w, 4), dtype=np.uint8)
                        rgba[..., 0] = shadow_color[0]; rgba[..., 1] = shadow_color[1]; rgba[..., 2] = shadow_color[2]
                        rgba[..., 3] = (off if isinstance(off, np.ndarray) else np.array(off))
                        return rgba
                    shadow = VideoClip(make_shadow, duration=clip.duration).with_opacity(max(0.0, min(1.0, shadow_opacity)))
                else:
                    shadow = None

                layers = [clip]
                if shadow is not None:
                    layers.append(shadow)
                layers.append(subj_anim)
                return CompositeVideoClip(layers).with_duration(clip.duration)
        except Exception as e:
            import traceback
            logger.error(f"[effects] subject_pop silhouette path failed: {e}")
            logger.error(f"[effects] traceback: {traceback.format_exc()}")

    subject = clip.cropped(x1=x, y1=y, x2=x + bw, y2=y + bh)
    cx = x + bw // 2
    cy = y + bh // 2

    base_scale = 1.0
    target = max(1.0, scale)

    def s_at(t):
        if t <= pop_duration:
            return base_scale + (target - base_scale) * ease_out_back(t / pop_duration)
        return target

    subj_anim = subject.resized(lambda t: s_at(t))

    def pos_at(t):
        s = s_at(t)
        sw = int(bw * s)
        sh = int(bh * s)
        return (cx - sw // 2, cy - sh // 2)

    subj_anim = subj_anim.with_position(pos_at)

    shadow = ColorClip(size=(bw, bh), color=shadow_color, duration=clip.duration).with_opacity(max(0.0, min(1.0, shadow_opacity)))
    shadow = shadow.with_position(lambda t: (pos_at(t)[0] + shadow_offset_px[0], pos_at(t)[1] + shadow_offset_px[1]))

    base_layers = [clip, shadow, subj_anim]

    return CompositeVideoClip(base_layers).with_duration(clip.duration)

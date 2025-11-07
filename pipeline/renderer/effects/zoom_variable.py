from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip as _VideoClip

from utils.logger import setup_logger
from .coords import denorm_bbox
from .easing import ease_in_out_cubic
from .subject_import import import_subject_detection

logger = setup_logger(__name__)


def apply_variable_zoom_on_subject(
    clip: _VideoClip,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[Any] = None,
    target_scale: float = 1.3,
    speed_curve: str = "easeInOut",
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
    anchor_feature: str = "auto",
    anchor_point: Optional[Tuple[int, int]] = None,
    anim_duration: Optional[float] = None,
    visual_description: Optional[str] = None,
) -> _VideoClip:
    """Apply variable zoom focused on detected subject.

    :param visual_description: Optional visual query/description to extract target object.
                               If provided and no explicit target, will attempt to detect
                               mentioned objects (e.g., "car", "phone") instead of defaulting to "person".
    """
    w, h = clip.size
    logger.debug(f"[effects] zoom_on_subject called: clip_size=({w}, {h}), bbox={bbox is not None}, mask={mask is not None}")

    # If no explicit target but we have visual description, try to extract target object
    if target is None and visual_description:
        sd = import_subject_detection()
        extract_target = sd.get("extract_target_from_description")
        if callable(extract_target):
            inferred_target = extract_target(visual_description)  # type: ignore
            if inferred_target:
                target = inferred_target
                logger.info(f"[effects] zoom_on_subject: Inferred target '{target}' from description: '{visual_description[:100]}'...")

    if bbox is None:
        try:
            frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.5))
            sd = import_subject_detection()
            # Use detect_subject_shape to get confidence
            ds = sd.get("detect_subject_shape")
            if callable(ds):
                result = ds(frame, target=target, prefer=prefer, nth=nth, min_confidence=0.25)  # type: ignore
                bbox = result.get("bbox")
                confidence = result.get("confidence")

                # If no clear subject detected (low confidence or None bbox), skip zoom
                if bbox is None or confidence is None:
                    logger.debug("[effects] zoom_on_subject: No clear subject detected, skipping zoom")
                    return clip  # Return clip unchanged
                if confidence < 0.25:
                    logger.debug(f"[effects] zoom_on_subject: Low subject confidence ({confidence:.2f}), skipping zoom")
                    return clip

                logger.debug(f"[effects] zoom_on_subject: Subject detected with confidence {confidence:.2f}")
            else:
                # Fallback to old method
                db = sd.get("detect_subject_bbox")
                if callable(db):
                    bbox = db(frame, target=target, prefer=prefer, nth=nth)  # type: ignore
                else:
                    raise ImportError("detect_subject_bbox not available")
        except Exception as e:
            logger.debug(f"[effects] zoom_on_subject: Detection failed ({e}), using fallback")
            bbox = (0.3, 0.3, 0.4, 0.4)
    x, y, bw, bh = denorm_bbox(bbox, (w, h))

    # Detect bust shot and reduce zoom to prevent face cropping
    # Bust shots have low aspect ratio (width is significant portion of frame)
    bbox_aspect = bh / bw if bw > 0 else 1.0
    is_bust_shot = bbox_aspect <= 0.7
    if is_bust_shot:
        # Reduce zoom for bust shots to prevent face cropping
        target_scale = min(target_scale, 1.15)
        logger.debug(f"[effects] zoom_on_subject detected bust shot (aspect {bbox_aspect:.2f}), limiting zoom to {target_scale:.2f}x")

    cx, cy = None, None

    if anchor_point is not None:
        cx, cy = anchor_point
        logger.debug(f"[effects] zoom_on_subject using pre-computed anchor at ({cx:.1f}, {cy:.1f})")
    else:
        # Try adaptive face anchor first (prioritize this for person detection)
        try:
            sd = import_subject_detection()
            detect_adaptive = sd.get("detect_adaptive_face_anchor")
            if callable(detect_adaptive):
                frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.5))
                detected_anchor = detect_adaptive(frame, target=target, prefer=prefer, nth=nth)
                if detected_anchor:
                    cx, cy = detected_anchor
                    logger.debug(f"[effects] zoom_on_subject using adaptive face anchor at ({cx:.1f}, {cy:.1f})")
        except Exception as e:
            logger.debug(f"[effects] zoom_on_subject adaptive face anchor failed: {e}")

        # Fallback to mask-based detection if adaptive anchor failed
        if (cx is None or cy is None) and mask is not None:
            try:
                sd = import_subject_detection()
                detect_anchor = sd.get("detect_anchor_feature")
                if callable(detect_anchor):
                    try:
                        frame = clip.get_frame(min(0.001, (clip.duration or 0.001) * 0.5))
                        abs_bbox = (x, y, x + bw, y + bh) if bbox else None
                        detected_anchor = detect_anchor(frame, subject_mask=mask, subject_bbox=abs_bbox, feature_type=anchor_feature)
                        if detected_anchor:
                            cx, cy = detected_anchor
                            logger.debug(f"[effects] zoom_on_subject fallback anchor at ({cx:.1f}, {cy:.1f})")
                        else:
                            logger.debug(f"[effects] zoom_on_subject YOLO anchor detection returned None - using bbox fallback")
                    except Exception as e:
                        logger.debug(f"[effects] zoom_on_subject anchor detection failed: {e}")
            except Exception as e:
                logger.debug(f"[effects] zoom_on_subject mask center failed: {e}")

    # Final fallback: bbox-based estimation
    if cx is None or cy is None:
        cx = x + bw / 2
        cy = y + bh * 0.25
        logger.debug(f"[effects] zoom_on_subject using bbox fallback anchor at ({cx:.1f}, {cy:.1f})")

    aspect_ratio = w / h
    h_by_topbottom = 2 * min(cy, h - cy)
    h_by_leftright = (2 * min(cx, w - cx)) / aspect_ratio
    h_by_image = min(h, w / aspect_ratio)

    max_crop_h = min(h_by_topbottom, h_by_leftright, h_by_image)
    max_crop_w = aspect_ratio * max_crop_h
    max_zoom_scale = h / max_crop_h if max_crop_h > 0 else 1.0

    effective_zoom = min(target_scale, max_zoom_scale)

    def ease(t):
        if speed_curve == "linear":
            return max(0.0, min(1.0, t))
        return ease_in_out_cubic(t)

    dur = max(0.01, clip.duration or 1.0)
    anim_dur = anim_duration if anim_duration is not None else dur

    def tf(get_frame, t):
        frame = get_frame(t)
        clamped_t = min(t, anim_dur)
        p = ease(clamped_t / anim_dur)
        zoom = 1.0 + (effective_zoom - 1.0) * p
        crop_h = h / zoom
        crop_w = w / zoom
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x2 = int(x1 + crop_w)
        y2 = int(y1 + crop_h)

        # Clamp crop coordinates to stay within frame bounds
        # Adjust position to keep crop size consistent
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # Final clamp to ensure we don't go negative after adjustment
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        sub = frame[y1:y2, x1:x2]
        if sub.size == 0:
            sub = frame
        img = Image.fromarray(sub).resize((w, h), Image.Resampling.LANCZOS)
        return np.array(img)

    def make_zoomed_frame(t):
        try:
            if abs(t - 0.0) < 1e-3 or abs(t - (clip.duration or 0) / 2) < 1e-2 or abs(t - (clip.duration or 0)) < 1e-3:
                logger.debug(f"[effects] zoom_on_subject make_frame t={t:.3f}s")
        except Exception:
            pass
        return tf(clip.get_frame, t)

    return _VideoClip(make_zoomed_frame, duration=clip.duration).with_fps(clip.fps if hasattr(clip, 'fps') else 24)

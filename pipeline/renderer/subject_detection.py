"""
Subject detection utilities returning a normalized bbox (x,y,w,h) in [0,1].

Backends (auto-selected in this order):
1) YOLOv8/YOLOv8-seg via Ultralytics if available (fast, accurate, general)
2) Face detection via Haar cascades (OpenCV builtin)
3) Edge-based saliency fallback (Canny + dilation + largest central-ish contour)

Environment overrides:
- USE_YOLO_DETECTOR=1|0 to enable/disable YOLO backend (default: 1 if ultralytics is importable)
- YOLO_WEIGHTS: model name or path (preferred, supports yolov8/11/… e.g., "yolov11x-seg.pt")
- YOLOV8_WEIGHTS: legacy env var still supported for backward compatibility
"""
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------- Helpers ----------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
def _grabcut_mask(image_rgb: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Approximate subject mask using OpenCV GrabCut initialized from a bbox.
    Returns uint8 mask (0/255) or None on failure.
    """
    try:
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img_bgr = image_rgb[..., ::-1]
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    # Expand rectangle slightly to capture full subject
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx = max(2, int(0.08 * bw))
    my = max(2, int(0.08 * bh))
    rx1 = max(0, x1 - mx)
    ry1 = max(0, y1 - my)
    rx2 = min(w, x2 + mx)
    ry2 = min(h, y2 + my)
    rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)
    # Prepare mask and models
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # Foreground where mask==1 or mask==3
        fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        # Mild post smooth
        k = max(3, int(round(min(w, h) * 0.004)) | 1)
        if k > 1:
            fg = cv2.medianBlur(fg, k)
        return fg
    except Exception:
        return None


def _norm_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    x, y = min(x1, x2), min(y1, y2)
    bw, bh = max(1, abs(x2 - x1)), max(1, abs(y2 - y1))
    return (_clamp01(x / w), _clamp01(y / h), _clamp01(bw / w), _clamp01(bh / h))


def _face_bbox(gray: np.ndarray) -> Tuple[int, int, int, int] | None:
    # Use Haar cascade for frontal faces (ships with opencv)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        # Pick the largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return int(x), int(y), int(x + w), int(y + h)
    except Exception:
        return None


def detect_anchor_feature(
    image_rgb: np.ndarray,
    subject_mask: Optional[np.ndarray] = None,
    subject_bbox: Optional[Tuple[int, int, int, int]] = None,
    feature_type: str = "auto",
    subject_class: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """
    Detect the anchor feature (e.g., face center, head center) within a subject.

    CLASS-AWARE ANCHORING:
    - For "person" class: Try face detection, fall back to head region
    - For other classes: Use geometric center (no face detection)

    Args:
        image_rgb: RGB image as numpy array
        subject_mask: Optional binary mask of the subject (0/255 or 0/1)
        subject_bbox: Optional bbox (x1, y1, x2, y2) of the subject to limit search area
        feature_type: "auto", "face", "head", or "center"
            - "auto": Smart detection based on subject_class
            - "face": Only use face detection (for humans)
            - "head": Use upper portion of mask/bbox
            - "center": Use geometric center
        subject_class: Optional detected class name (e.g., "person", "car", "cat")
                       Used to decide whether to attempt face detection

    Returns:
        (cx, cy) in absolute image coordinates, or None if detection fails
    """
    h, w = image_rgb.shape[:2]

    # CLASS-AWARE DECISION: Only attempt face detection for humans
    subject_is_person = (subject_class or "").lower() in ("person", "human", "man", "woman", "child", "people")

    # Skip face detection entirely for non-human subjects
    if feature_type == "auto" and not subject_is_person:
        logger.debug(f"[subject_detection] Subject is '{subject_class}' (non-human), skipping face detection → using center")
        # Jump directly to center-based anchor
        if subject_mask is not None:
            yy, xx = np.where(subject_mask > 0)
            if len(yy) > 0:
                cx = int(np.mean(xx))
                cy = int(np.mean(yy))
                logger.debug(f"[subject_detection] Anchor feature (mask center for {subject_class}) at ({cx}, {cy})")
                return (cx, cy)

        if subject_bbox is not None:
            x1, y1, x2, y2 = subject_bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            logger.debug(f"[subject_detection] Anchor feature (bbox center for {subject_class}) at ({cx}, {cy})")
            return (cx, cy)

        logger.debug("[subject_detection] No mask/bbox for non-human subject, anchor detection failed")
        return None

    # Determine search region - expand significantly to ensure full person visible
    search_region = image_rgb
    offset_x, offset_y = 0, 0
    if subject_bbox is not None:
        x1, y1, x2, y2 = subject_bbox
        # Expand search region by 20% on all sides to ensure we catch the full person
        expand = 0.20
        ex1 = max(0, int(x1 - (x2 - x1) * expand))
        ey1 = max(0, int(y1 - (y2 - y1) * expand))
        ex2 = min(w, int(x2 + (x2 - x1) * expand))
        ey2 = min(h, int(y2 + (y2 - y1) * expand))
        search_region = image_rgb[ey1:ey2, ex1:ex2]
        offset_x, offset_y = ex1, ey1

    # Try YOLO face detection for "auto" or "face" (ONLY for humans)
    if feature_type in ("auto", "face") and subject_is_person:
        model = _get_yolo_model()
        if model is not None and subject_bbox is not None:
            try:
                # Crop to subject region for focused face detection
                x1, y1, x2, y2 = subject_bbox
                subject_crop = search_region
                
                logger.debug(f"[subject_detection] Running YOLO on subject crop: region_shape={subject_crop.shape}, bbox=({x1},{y1},{x2},{y2})")
                
                # Run YOLO on the subject crop to detect face/head
                # Some YOLO models support person keypoints which include face landmarks
                results = model.predict(
                    subject_crop,
                    verbose=False,
                    imgsz=640,
                    conf=0.10,  # Lower confidence to catch more detections
                    classes=[0],  # person class - will detect person, then we'll use upper portion
                )
                
                logger.debug(f"[subject_detection] YOLO results: {len(results)} result(s)")
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    # Check different ways to get box count
                    try:
                        box_count = len(boxes)
                    except:
                        try:
                            box_count = boxes.shape[0] if hasattr(boxes, 'shape') else 0
                        except:
                            box_count = len(boxes.data) if hasattr(boxes, 'data') else 0
                    logger.debug(f"[subject_detection] YOLO boxes: {box_count} detection(s)")
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    # Try to access boxes data directly
                    try:
                        if hasattr(boxes, 'data') and boxes.data is not None and len(boxes.data) > 0:
                            conf_tensor = boxes.conf
                            if conf_tensor is not None and len(conf_tensor) > 0:
                                best_idx = int(conf_tensor.argmax())
                                bbox = boxes.xyxy[best_idx].cpu().numpy()
                                
                                # Face is in the upper region of detected person
                                px1, py1, px2, py2 = bbox
                                person_height = py2 - py1

                                # Face center (eyes/nose region): 25% down from top
                                # This works for both full-body shots and bust statues
                                # For busts: 25% is around eyes/nose (avoids beard)
                                # For full-body: 25% is upper head region (good anchor)
                                face_y_center = py1 + person_height * 0.25
                                face_x_center = (px1 + px2) / 2
                                
                                # Convert back to absolute image coordinates
                                face_cx = int(offset_x + face_x_center)
                                face_cy = int(offset_y + face_y_center)
                                
                                logger.debug(f"[subject_detection] Anchor feature (YOLO face from person upper region) at ({face_cx}, {face_cy})")
                                return (face_cx, face_cy)
                    except Exception as e:
                        logger.debug(f"[subject_detection] YOLO box processing failed: {e}")
                
                logger.debug("[subject_detection] YOLO person detection in subject crop yielded no results")
                
            except Exception as e:
                logger.debug(f"[subject_detection] YOLO face detection failed: {e}")
        
        # No fallback - YOLO only
        logger.debug("[subject_detection] Face detection failed - YOLO required, no fallback")
        return None
    
    # Fall back to mask-based or bbox-based head estimation
    if feature_type in ("auto", "head"):
        if subject_mask is not None:
            # Use upper 30% of the mask as "head region"
            yy, xx = np.where(subject_mask > 0)
            if len(yy) > 0:
                y_min = yy.min()
                y_max = yy.max()
                y_range = y_max - y_min
                head_y_max = y_min + int(y_range * 0.30)
                head_mask = (yy <= head_y_max)
                if np.sum(head_mask) > 0:
                    head_cx = int(np.mean(xx[head_mask]))
                    head_cy = int(np.mean(yy[head_mask]))
                    logger.debug(f"[subject_detection] Anchor feature (head from mask) at ({head_cx}, {head_cy})")
                    return (head_cx, head_cy)
        
        if subject_bbox is not None:
            # Use upper 30% of bbox
            x1, y1, x2, y2 = subject_bbox
            head_y = y1 + int((y2 - y1) * 0.30)
            head_cx = (x1 + x2) // 2
            logger.debug(f"[subject_detection] Anchor feature (head from bbox) at ({head_cx}, {head_y})")
            return (head_cx, head_y)
    
    # "center" or final fallback
    if subject_mask is not None:
        yy, xx = np.where(subject_mask > 0)
        if len(yy) > 0:
            cx = int(np.mean(xx))
            cy = int(np.mean(yy))
            logger.debug(f"[subject_detection] Anchor feature (mask center) at ({cx}, {cy})")
            return (cx, cy)
    
    if subject_bbox is not None:
        x1, y1, x2, y2 = subject_bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        logger.debug(f"[subject_detection] Anchor feature (bbox center) at ({cx}, {cy})")
        return (cx, cy)
    
    logger.debug("[subject_detection] Anchor feature detection failed")
    return None
    return None


def _edge_saliency_bbox(gray: np.ndarray) -> Tuple[int, int, int, int] | None:
    h, w = gray.shape[:2]
    # Auto Canny thresholds based on median
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    edges = cv2.Canny(gray, lower, upper)
    # Thicken edges and fill small gaps
    k = max(3, int(round(min(w, h) * 0.01)))
    k = k + 1 if k % 2 == 0 else k
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Score contours by area with a center bias
    cx, cy = w / 2, h / 2
    def score(cnt):
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        ccx, ccy = x + bw / 2, y + bh / 2
        # Distance to center (smaller is better)
        dist = np.hypot(ccx - cx, ccy - cy)
        # Favor larger area and center proximity
        return area - 2.5 * dist

    best = max(cnts, key=score)
    x, y, bw, bh = cv2.boundingRect(best)
    return int(x), int(y), int(x + bw), int(y + bh)


_YOLO_MODEL = None  # lazy-loaded Ultralytics model
_CLIP_MODEL = None  # lazy-loaded CLIP model
_CLIP_PREPROCESS = None


def _get_yolo_model() -> Optional[object]:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    # Check env toggle
    use = os.getenv("USE_YOLO_DETECTOR", "1").lower() in ("1", "true", "yes")
    if not use:
        return None
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return None
    # Prefer generic YOLO_WEIGHTS, fallback to legacy YOLOV8_WEIGHTS
    weights = os.getenv("YOLO_WEIGHTS") or os.getenv("YOLOV8_WEIGHTS") or "yolov11x-seg.pt"
    try:
        _YOLO_MODEL = YOLO(weights)
        # Try to report the underlying task type (detect/segment)
        task = None
        try:
            task = getattr(getattr(_YOLO_MODEL, "model", _YOLO_MODEL), "task", None)
        except Exception:
            task = None
        if task:
            logger.info(f"[subject_detection] Loaded YOLO model: {weights} (task={task})")
        else:
            logger.info(f"[subject_detection] Loaded YOLO model: {weights}")
        return _YOLO_MODEL
    except Exception:
        return None


def _get_clip() -> Tuple[Optional[object], Optional[object], str]:
    """Lazy-load OpenAI CLIP for re-ranking candidate detections by text prompt.
    Returns (model, preprocess, device_str).
    """
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None:
        try:
            from config import DEVICE as _DEV
        except Exception:
            _DEV = "cpu"
        return _CLIP_MODEL, _CLIP_PREPROCESS, _DEV
    try:
        import clip  # type: ignore
        import torch  # type: ignore
    except Exception:
        logger.debug("[subject_detection] CLIP not available; skipping rerank")
        return None, None, "cpu"
    try:
        from config import DEVICE as _DEV
    except Exception:
        _DEV = "cpu"
    device = _DEV if _DEV in ("cuda", "cpu", "mps") else "cpu"
    try:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        logger.debug(f"[subject_detection] CLIP model loaded on {device}")
        return _CLIP_MODEL, _CLIP_PREPROCESS, device
    except Exception as e:
        logger.debug(f"[subject_detection] CLIP load failed: {e}")
        return None, None, "cpu"


def _clip_select_index(image_rgb: np.ndarray, xyxy: np.ndarray, target: str) -> Optional[int]:
    """Use CLIP to choose the best bbox for the given target phrase.
    Returns best index or None on failure.
    """
    if not target or xyxy is None or len(xyxy) == 0:
        return None
    clip_model, clip_preproc, device = _get_clip()
    if clip_model is None or clip_preproc is None:
        return None
    try:
        import torch  # type: ignore
        from PIL import Image as _PIL_Image
        import clip as _clip  # type: ignore
        crops: List[object] = []
        H, W = image_rgb.shape[:2]
        for (x1, y1, x2, y2) in xyxy:
            xx1 = max(0, int(x1)); yy1 = max(0, int(y1)); xx2 = min(W, int(x2)); yy2 = min(H, int(y2))
            if xx2 <= xx1 or yy2 <= yy1:
                # degenerate
                crop = image_rgb
            else:
                crop = image_rgb[yy1:yy2, xx1:xx2]
            pil = _PIL_Image.fromarray(crop)
            crops.append(clip_preproc(pil))
        if len(crops) == 0:
            return None
        with torch.no_grad():
            text = f"a {target}"
            text_tokens = _clip.tokenize([text]).to(device)
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            imgs = torch.stack(crops, dim=0).to(device)
            img_feat = clip_model.encode_image(imgs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feat.T).squeeze(-1)  # shape [N]
            best_idx = int(torch.argmax(sims).item())
            logger.debug(f"[subject_detection] CLIP rerank picked idx={best_idx} for target='{target}' (max sim={float(sims[best_idx]):.3f})")
            return best_idx
    except Exception as e:
        logger.debug(f"[subject_detection] CLIP rerank failed: {e}")
        return None


def _yolo_bbox(
    image_rgb: np.ndarray,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
) -> Tuple[int, int, int, int] | None:
    model = _get_yolo_model()
    if model is None:
        return None
    try:
        # Ultralytics automatically handles preprocessing; return highest-scoring, center-biased
        # Restrict to 'person' class to avoid picking non-human objects (e.g., a map)
        results = model.predict(image_rgb, verbose=False, conf=0.15, classes=[0])
        if not results:
            return None
        r = results[0]
        logger.debug("[subject_detection] YOLO inference ran (boxes only path)")
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        # boxes.xyxy is Nx4 tensor, boxes.conf is Nx1, boxes.cls is Nx1
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1)
        cls = None
        try:
            cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cls = None
        # Names mapping for class labels
        names = None
        try:
            names = getattr(getattr(model, "model", model), "names", None)
        except Exception:
            names = None
        if isinstance(names, dict):
            id_to_name = names
        elif isinstance(names, list):
            id_to_name = {i: n for i, n in enumerate(names)}
        else:
            id_to_name = {}
        # Pick by confidence with a slight center bias
        h, w = image_rgb.shape[:2]
        cx, cy = w / 2, h / 2
        indices = list(range(len(xyxy)))
        # Normalize selection preferences early
        target_lc = (target or "").strip().lower()
        prefer_lc = (prefer or "").strip().lower()
        # Optional nth selection among filtered class (e.g., persons) sorted left->right
        if nth is not None and cls is not None and len(xyxy) > 0:
            # Prefer person class when nth specified; else use all
            person_ids = [i for i in indices if id_to_name.get(int(cls[i]), "").lower() == "person"] if id_to_name else indices
            order = sorted(person_ids, key=lambda i: (xyxy[i][0] + xyxy[i][2]) / 2)
            k = max(1, int(nth)) - 1
            if 0 <= k < len(order):
                i = order[k]
                x1, y1, x2, y2 = xyxy[i]
                return int(x1), int(y1), int(x2), int(y2)
        # Optional CLIP rerank by target text (if provided)
        picked = None
        if target_lc:
            picked = _clip_select_index(image_rgb, xyxy, target_lc)
            if picked is not None:
                x1, y1, x2, y2 = xyxy[picked]
                return int(x1), int(y1), int(x2), int(y2)

        # General scoring with preferences
        best_idx = 0
        best_score = -1e9
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            ccx = (x1 + x2) / 2
            ccy = (y1 + y2) / 2
            dist = np.hypot(ccx - cx, ccy - cy)
            area = max(1.0, (x2 - x1) * (y2 - y1))
            score = float(conf[i]) - 0.002 * dist + 0.000001 * area  # small center + tiny area bias
            # Prefer class by simple keywords
            if target_lc:
                cname = id_to_name.get(int(cls[i]), "").lower() if cls is not None else ""
                if target_lc in {"girl", "woman", "female"}:
                    if cname == "person":
                        score += 0.1  # light bump for people
                elif target_lc in {"boy", "man", "male"}:
                    if cname == "person":
                        score += 0.1
                else:
                    if target_lc in cname:
                        score += 0.2
            # Positional preferences
            if prefer_lc == "left":
                score += -0.0005 * ccx
            elif prefer_lc == "right":
                score += 0.0005 * ccx
            elif prefer_lc == "top":
                score += -0.0005 * ccy
            elif prefer_lc == "bottom":
                score += 0.0005 * ccy
            elif prefer_lc == "largest":
                score += 0.00001 * area
            elif prefer_lc == "center":
                # Increase center bias when explicitly requested
                score += -0.002 * dist
            if score > best_score:
                best_score = score
                best_idx = i
        x1, y1, x2, y2 = xyxy[best_idx]
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def _yolo_mask_and_bbox(
    image_rgb: np.ndarray,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Return (mask_uint8_HxW_0_255, bbox_xyxy_int) for best object if YOLO-seg available.
    If no mask available, returns (None, bbox) if boxes exist.
    """
    model = _get_yolo_model()
    if model is None:
        return None, None
    try:
        # Prefer explicit segmentation inference; older Ultralytics may not accept task='segment'.
        try:
            results = model.predict(
                image_rgb,
                verbose=False,
                task="segment",  # type: ignore[arg-type]
                retina_masks=True,
                imgsz=1280,
                conf=0.25,  # Increased from 0.12 to be more strict
                iou=0.5,    # Add IoU threshold for cleaner masks
                classes=[0],  # focus on 'person' for more reliable masks
            )
            logger.debug("[subject_detection] YOLO segmentation predict attempted")
        except TypeError:
            # Fallback for older versions
            logger.debug("[subject_detection] YOLO predict(task='segment') unsupported; falling back to default predict")
            results = model.predict(image_rgb, verbose=False, retina_masks=True, imgsz=1280, conf=0.25, classes=[0])
        except Exception:
            results = model.predict(image_rgb, verbose=False, conf=0.25, classes=[0])
        if not results:
            return None, None
        r = results[0]
        h, w = image_rgb.shape[:2]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if masks is None:
            logger.debug("[subject_detection] Results.masks is None")
        else:
            try:
                dl = len(getattr(masks, "data", []))
                ml = len(getattr(masks, "masks", [])) if hasattr(masks, "masks") else 0
                xl = len(getattr(masks, "xy", [])) if hasattr(masks, "xy") else 0
                logger.debug(f"[subject_detection] masks present: data_len={dl}, tensor_len={ml}, polys_len={xl}")
            except Exception:
                logger.debug("[subject_detection] masks present but lengths unavailable")
        if boxes is None or len(boxes) == 0:
            return None, None
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1)
        try:
            cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cls = None
        # Names mapping for class labels
        names = None
        try:
            names = getattr(getattr(model, "model", model), "names", None)
        except Exception:
            names = None
        if isinstance(names, dict):
            id_to_name = names
        elif isinstance(names, list):
            id_to_name = {i: n for i, n in enumerate(names)}
        else:
            id_to_name = {}
        cx, cy = w / 2, h / 2
        indices = list(range(len(xyxy)))
        # Normalize selection preferences early
        target_lc = (target or "").strip().lower()
        prefer_lc = (prefer or "").strip().lower()
        # Optional nth selection among filtered class (e.g., persons) sorted left->right
        if nth is not None and cls is not None and len(xyxy) > 0:
            person_ids = [i for i in indices if id_to_name.get(int(cls[i]), "").lower() == "person"] if id_to_name else indices
            order = sorted(person_ids, key=lambda i: (xyxy[i][0] + xyxy[i][2]) / 2)
            k = max(1, int(nth)) - 1
            if 0 <= k < len(order):
                best_idx = order[k]
                x1, y1, x2, y2 = xyxy[best_idx]
                bbox = (int(x1), int(y1), int(x2), int(y2))
                # Return mask if available for this index (data/tensor/polygons)
                if masks is not None:
                    data = getattr(masks, "data", None)
                    if data is not None and len(data) > best_idx:
                        m = data[best_idx].cpu().numpy()
                        mh, mw = m.shape[-2], m.shape[-1]
                        if (mw, mh) != (w, h):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        m = (m > 0.5).astype(np.uint8) * 255
                        logger.debug("[subject_detection] Segmentation mask obtained from YOLO (nth selector)")
                        return m, bbox
                    masks_tensor = getattr(masks, "masks", None)
                    if masks_tensor is not None and len(masks_tensor) > best_idx:
                        m = masks_tensor[best_idx].cpu().numpy()
                        mh, mw = m.shape[-2], m.shape[-1]
                        if (mw, mh) != (w, h):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        m = (m > 0.5).astype(np.uint8) * 255
                        logger.debug("[subject_detection] Segmentation mask (tensor) obtained (nth selector)")
                        return m, bbox
                    xy_polys = getattr(masks, "xy", None)
                    if xy_polys is not None and len(xy_polys) > best_idx and xy_polys[best_idx] is not None:
                        m = np.zeros((h, w), dtype=np.uint8)
                        polys = xy_polys[best_idx]
                        if isinstance(polys, (list, tuple)):
                            for poly in polys:
                                if poly is None or len(poly) == 0:
                                    continue
                                pts = np.round(poly).astype(np.int32)
                                cv2.fillPoly(m, [pts], 255)
                        else:
                            pts = np.round(polys).astype(np.int32)
                            cv2.fillPoly(m, [pts], 255)
                        logger.debug("[subject_detection] Segmentation mask (polygons) rasterized (nth selector)")
                        return m, bbox
                # No mask: try GrabCut fallback for nth
                try:
                    m_gc = _grabcut_mask(image_rgb, bbox)
                    if m_gc is not None and int((m_gc > 0).sum()) > 0:
                        nz = int((m_gc > 0).sum())
                        logger.debug(f"[subject_detection] GrabCut fallback mask extracted (nth selector, area_px={nz})")
                        return m_gc, bbox
                except Exception:
                    pass
                return None, bbox
        # Optional CLIP rerank by target text (if provided)
        picked = None
        if target_lc:
            picked = _clip_select_index(image_rgb, xyxy, target_lc)
            if picked is not None:
                x1, y1, x2, y2 = xyxy[picked]
                bbox = (int(x1), int(y1), int(x2), int(y2))
                if masks is not None:
                    data = getattr(masks, "data", None)
                    if data is not None and len(data) > picked:
                        m = data[picked].cpu().numpy()
                        mh, mw = m.shape[-2], m.shape[-1]
                        if (mw, mh) != (w, h):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        m = (m > 0.5).astype(np.uint8) * 255
                        nz = int((m > 0).sum())
                        logger.debug(f"[subject_detection] Segmentation mask obtained from YOLO (CLIP-picked idx={picked}, area_px={nz})")
                        return m, bbox
                    masks_tensor = getattr(masks, "masks", None)
                    if masks_tensor is not None and len(masks_tensor) > picked:
                        m = masks_tensor[picked].cpu().numpy()
                        mh, mw = m.shape[-2], m.shape[-1]
                        if (mw, mh) != (w, h):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        m = (m > 0.5).astype(np.uint8) * 255
                        nz = int((m > 0).sum())
                        logger.debug(f"[subject_detection] Segmentation mask (tensor) obtained (CLIP-picked idx={picked}, area_px={nz})")
                        return m, bbox
                    xy_polys = getattr(masks, "xy", None)
                    if xy_polys is not None and len(xy_polys) > picked and xy_polys[picked] is not None:
                        m = np.zeros((h, w), dtype=np.uint8)
                        polys = xy_polys[picked]
                        if isinstance(polys, (list, tuple)):
                            for poly in polys:
                                if poly is None or len(poly) == 0:
                                    continue
                                pts = np.round(poly).astype(np.int32)
                                cv2.fillPoly(m, [pts], 255)
                        else:
                            pts = np.round(polys).astype(np.int32)
                            cv2.fillPoly(m, [pts], 255)
                        nz = int((m > 0).sum())
                        logger.debug(f"[subject_detection] Segmentation mask (polygons) rasterized (CLIP-picked idx={picked}, area_px={nz})")
                        return m, bbox
                # No mask: try GrabCut fallback for CLIP-picked
                try:
                    m_gc = _grabcut_mask(image_rgb, bbox)
                    if m_gc is not None and int((m_gc > 0).sum()) > 0:
                        nz = int((m_gc > 0).sum())
                        logger.debug(f"[subject_detection] GrabCut fallback mask extracted (CLIP-picked, area_px={nz})")
                        return m_gc, bbox
                except Exception:
                    pass
                return None, bbox

        best_idx = 0
        best_score = -1e9
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            ccx = (x1 + x2) / 2
            ccy = (y1 + y2) / 2
            dist = np.hypot(ccx - cx, ccy - cy)
            area = max(1.0, (x2 - x1) * (y2 - y1))
            score = float(conf[i]) - 0.002 * dist + 0.000001 * area
            if target_lc:
                cname = id_to_name.get(int(cls[i]), "").lower() if cls is not None else ""
                if target_lc in {"girl", "woman", "female"}:
                    if cname == "person":
                        score += 0.1
                elif target_lc in {"boy", "man", "male"}:
                    if cname == "person":
                        score += 0.1
                else:
                    if target_lc in cname:
                        score += 0.2
            if prefer_lc == "left":
                score += -0.0005 * ccx
            elif prefer_lc == "right":
                score += 0.0005 * ccx
            elif prefer_lc == "top":
                score += -0.0005 * ccy
            elif prefer_lc == "bottom":
                score += 0.0005 * ccy
            elif prefer_lc == "largest":
                score += 0.00001 * area
            elif prefer_lc == "center":
                score += -0.002 * dist
            if score > best_score:
                best_score = score
                best_idx = i
        x1, y1, x2, y2 = xyxy[best_idx]
        bbox = (int(x1), int(y1), int(x2), int(y2))
        # If segmentation present, get mask for best_idx (data/tensor/polygons)
        if masks is not None:
            data = getattr(masks, "data", None)
            if data is not None and len(data) > best_idx:
                m = data[best_idx].cpu().numpy()  # values 0..1 or 0/1
                mh, mw = m.shape[-2], m.shape[-1]
                if (mw, mh) != (w, h):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = (m > 0.5).astype(np.uint8) * 255
                nz = int((m > 0).sum())
                logger.debug(f"[subject_detection] Segmentation mask obtained from YOLO (idx={best_idx}, area_px={nz})")
                return m, bbox
            masks_tensor = getattr(masks, "masks", None)
            if masks_tensor is not None and len(masks_tensor) > best_idx:
                m = masks_tensor[best_idx].cpu().numpy()
                mh, mw = m.shape[-2], m.shape[-1]
                if (mw, mh) != (w, h):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = (m > 0.5).astype(np.uint8) * 255
                nz = int((m > 0).sum())
                logger.debug(f"[subject_detection] Segmentation mask (tensor) obtained (idx={best_idx}, area_px={nz})")
                return m, bbox
            xy_polys = getattr(masks, "xy", None)
            if xy_polys is not None and len(xy_polys) > best_idx and xy_polys[best_idx] is not None:
                m = np.zeros((h, w), dtype=np.uint8)
                polys = xy_polys[best_idx]
                if isinstance(polys, (list, tuple)):
                    for poly in polys:
                        if poly is None or len(poly) == 0:
                            continue
                        pts = np.round(poly).astype(np.int32)
                        cv2.fillPoly(m, [pts], 255)
                else:
                    pts = np.round(polys).astype(np.int32)
                    cv2.fillPoly(m, [pts], 255)
                nz = int((m > 0).sum())
                logger.debug(f"[subject_detection] Segmentation mask (polygons) rasterized (idx={best_idx}, area_px={nz})")
                return m, bbox
        else:
            # Helpful diagnostic if a -seg model still returned no masks
            try:
                task = getattr(getattr(model, "model", model), "task", None)
            except Exception:
                task = None
            # Log top-1 class and conf for context
            try:
                top_i = int(best_idx)
                top_cls = int(cls[top_i]) if cls is not None else -1
                top_conf = float(conf[top_i]) if conf is not None else -1.0
                cname = (id_to_name.get(top_cls, str(top_cls)) if isinstance(id_to_name, dict) else str(top_cls))
                logger.debug(f"[subject_detection] YOLO returned no masks (task={task}); top=(idx={top_i}, class={cname}, conf={top_conf:.3f})")
            except Exception:
                logger.debug(f"[subject_detection] YOLO returned no masks (task={task}); using boxes only")

            # Optional GrabCut fallback to derive a soft mask from bbox
            try:
                m_gc = _grabcut_mask(image_rgb, bbox)
                if m_gc is not None and int((m_gc > 0).sum()) > 0:
                    nz = int((m_gc > 0).sum())
                    logger.debug(f"[subject_detection] GrabCut fallback mask extracted (area_px={nz})")
                    return m_gc, bbox
            except Exception:
                pass
        return None, bbox
    except Exception:
        return None, None


def detect_main_subject(
    image_rgb: np.ndarray,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect the MAIN subject in the image (any object type - person, vehicle, animal, etc.).

    This is the recommended API for effect-aware subject detection.
    Unlike detect_subject_bbox which filters to 'person' class, this detects ALL objects
    and picks the most prominent one based on confidence, size, and center proximity.

    Args:
        image_rgb: RGB image array (H, W, 3)
        target: Optional text hint for CLIP reranking (e.g., "truck", "building")
        prefer: Optional spatial preference ("center", "left", "right", "largest")

    Returns:
        Dict with:
            - "bbox": normalized (x, y, w, h) in [0,1]
            - "class_name": detected class (e.g., "person", "car", "cat")
            - "class_id": COCO class ID
            - "confidence": detection confidence [0-1]
            - "mask": binary mask (H, W) if segmentation available, else None
    """
    if image_rgb is None or image_rgb.size == 0:
        return {
            "bbox": (0.3, 0.3, 0.4, 0.4),
            "class_name": "unknown",
            "class_id": -1,
            "confidence": 0.0,
            "mask": None,
        }

    img = image_rgb.astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb
    h, w = img.shape[:2]

    model = _get_yolo_model()
    if model is None:
        # Fallback to edge saliency
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        eb = _edge_saliency_bbox(gray)
        if eb is not None:
            return {
                "bbox": _norm_bbox(eb[0], eb[1], eb[2], eb[3], w, h),
                "class_name": "unknown",
                "class_id": -1,
                "confidence": 0.5,
                "mask": None,
            }
        # Ultimate fallback
        return {
            "bbox": (0.3, 0.3, 0.4, 0.4),
            "class_name": "unknown",
            "class_id": -1,
            "confidence": 0.0,
            "mask": None,
        }

    try:
        # Run YOLO with NO class filter - detect everything!
        results = model.predict(
            img,
            verbose=False,
            conf=0.20,  # Reasonable confidence threshold
            task="segment",  # Try to get masks
            retina_masks=True,
            imgsz=1280,
        )

        if not results or len(results) == 0:
            return {
                "bbox": (0.3, 0.3, 0.4, 0.4),
                "class_name": "unknown",
                "class_id": -1,
                "confidence": 0.0,
                "mask": None,
            }

        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        if boxes is None or len(boxes) == 0:
            return {
                "bbox": (0.3, 0.3, 0.4, 0.4),
                "class_name": "unknown",
                "class_id": -1,
                "confidence": 0.0,
                "mask": None,
            }

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1)
        cls = boxes.cls.cpu().numpy().astype(int)

        # Get class names
        names = getattr(getattr(model, "model", model), "names", None)
        if isinstance(names, dict):
            id_to_name = names
        elif isinstance(names, list):
            id_to_name = {i: n for i, n in enumerate(names)}
        else:
            id_to_name = {}

        # Optional CLIP rerank if target specified
        if target and target.strip():
            picked = _clip_select_index(img, xyxy, target.strip())
            if picked is not None:
                best_idx = picked
            else:
                # CLIP failed, use scoring
                best_idx = _score_detections(xyxy, conf, w, h, prefer)
        else:
            # Score by confidence + center + size
            best_idx = _score_detections(xyxy, conf, w, h, prefer)

        x1, y1, x2, y2 = xyxy[best_idx]
        bbox = _norm_bbox(int(x1), int(y1), int(x2), int(y2), w, h)
        class_id = int(cls[best_idx])
        class_name = id_to_name.get(class_id, f"class_{class_id}")
        confidence = float(conf[best_idx])

        # Try to get mask
        mask_arr = None
        if masks is not None:
            try:
                data = getattr(masks, "data", None)
                if data is not None and len(data) > best_idx:
                    m = data[best_idx].cpu().numpy()
                    mh, mw = m.shape[-2], m.shape[-1]
                    if (mw, mh) != (w, h):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_arr = (m > 0.5).astype(np.uint8) * 255
                    logger.debug(f"[subject_detection] Extracted mask for {class_name}")
            except Exception as e:
                logger.debug(f"[subject_detection] Mask extraction failed: {e}")

        logger.info(f"[subject_detection] Main subject detected: {class_name} (conf={confidence:.3f}, bbox={bbox})")

        return {
            "bbox": bbox,
            "class_name": class_name,
            "class_id": class_id,
            "confidence": confidence,
            "mask": mask_arr,
        }

    except Exception as e:
        logger.warning(f"[subject_detection] Main subject detection failed: {e}")
        return {
            "bbox": (0.3, 0.3, 0.4, 0.4),
            "class_name": "unknown",
            "class_id": -1,
            "confidence": 0.0,
            "mask": None,
        }


def _score_detections(
    xyxy: np.ndarray,
    conf: np.ndarray,
    img_w: int,
    img_h: int,
    prefer: Optional[str] = None,
) -> int:
    """Score detections by confidence + center bias + size, return best index."""
    cx, cy = img_w / 2, img_h / 2
    prefer_lc = (prefer or "").strip().lower()

    best_idx = 0
    best_score = -1e9

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        ccx = (x1 + x2) / 2
        ccy = (y1 + y2) / 2
        dist = np.hypot(ccx - cx, ccy - cy)
        area = max(1.0, (x2 - x1) * (y2 - y1))

        # Confidence is most important, then center, then size
        score = float(conf[i]) * 10.0 - 0.002 * dist + 0.00001 * area

        # Apply preference modifiers
        if prefer_lc == "left":
            score += -0.001 * ccx
        elif prefer_lc == "right":
            score += 0.001 * ccx
        elif prefer_lc == "top":
            score += -0.001 * ccy
        elif prefer_lc == "bottom":
            score += 0.001 * ccy
        elif prefer_lc == "largest":
            score += 0.0001 * area
        elif prefer_lc == "center":
            score += -0.005 * dist

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def detect_adaptive_face_anchor(
    image_rgb: np.ndarray,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Detect face anchor using adaptive percentage based on person bbox aspect ratio.

    This function directly uses YOLO to detect person bounding box, then calculates
    face position based on the bbox aspect ratio (height/width):

    - Full body (aspect > 1.4): Face at 12% from top
    - Half body (0.7 < aspect <= 1.4): Face at 25% from top
    - Bust (aspect <= 0.7): Face at 50% from top

    Args:
        image_rgb: RGB image as numpy array.
        target: Target subject type (default: "person").
        prefer: Preference for subject selection ("largest", "center", etc.).
        nth: Select nth subject (0-indexed).

    Returns:
        (x, y) face anchor in absolute pixel coordinates, or None if detection fails.
    """
    model = _get_yolo_model()
    if not model:
        logger.debug("[subject_detection] YOLO model not available for adaptive face anchor")
        return None

    try:
        # Run YOLO detection on full frame to get accurate person bbox
        results = model.predict(
            image_rgb,
            verbose=False,
            imgsz=640,
            conf=0.10,  # Low confidence to catch all detections
            classes=[0],  # person class
        )

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if hasattr(boxes, 'data') and boxes.data is not None and len(boxes.data) > 0:
                # Get best detection (highest confidence)
                conf_tensor = boxes.conf
                if conf_tensor is not None and len(conf_tensor) > 0:
                    best_idx = int(conf_tensor.argmax())
                    bbox = boxes.xyxy[best_idx].cpu().numpy()

                    px1, py1, px2, py2 = bbox
                    person_height = py2 - py1
                    person_width = px2 - px1
                    aspect_ratio = person_height / person_width

                    # Adaptive face positioning based on aspect ratio
                    # Based on measured dimensions:
                    # Full body: 737 x 1366 (aspect 1.85)
                    # Half body: 737 x 683 (aspect 0.93)
                    # Bust: 737 x 341 (aspect 0.46)
                    if aspect_ratio > 1.4:
                        # Full body shot
                        face_pct = 0.12
                        shot_type = "full body"
                    elif aspect_ratio > 0.7:
                        # Half body shot (waist up)
                        face_pct = 0.25
                        shot_type = "half body"
                    else:
                        # Bust shot (shoulders up)
                        face_pct = 0.50
                        shot_type = "bust"

                    face_x = (px1 + px2) / 2
                    face_y = py1 + person_height * face_pct

                    logger.debug(
                        f"[subject_detection] Adaptive face anchor: ({int(face_x)}, {int(face_y)}) "
                        f"for {shot_type} (aspect {aspect_ratio:.2f})"
                    )

                    return (int(face_x), int(face_y))

        logger.debug("[subject_detection] Adaptive face anchor: No person detected")
        return None

    except Exception as e:
        logger.debug(f"[subject_detection] Adaptive face anchor failed: {e}")
        return None


def detect_subject_bbox(
    image_rgb: np.ndarray,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute a subject bbox from an RGB image (H,W,3 uint8) and return
    normalized (x,y,w,h) in [0,1].

    LEGACY API: Filters to 'person' class only for backward compatibility.
    For class-agnostic detection, use detect_main_subject() instead.
    """
    if image_rgb is None or image_rgb.size == 0:
        return (0.3, 0.3, 0.4, 0.4)
    if image_rgb.dtype != np.uint8:
        img = image_rgb.astype(np.uint8)
    else:
        img = image_rgb
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 0) YOLO (if available)
    yb = _yolo_bbox(img, target=target, prefer=prefer, nth=nth)
    if yb is not None:
        return _norm_bbox(yb[0], yb[1], yb[2], yb[3], w, h)

    # 1) Face
    fb = _face_bbox(gray)
    if fb is not None:
        return _norm_bbox(fb[0], fb[1], fb[2], fb[3], w, h)

    # 2) Edge saliency
    eb = _edge_saliency_bbox(gray)
    if eb is not None:
        return _norm_bbox(eb[0], eb[1], eb[2], eb[3], w, h)

    # 3) Fallback: centered box
    cx1 = int(w * 0.3)
    cy1 = int(h * 0.3)
    cx2 = int(w * 0.7)
    cy2 = int(h * 0.7)
    return _norm_bbox(cx1, cy1, cx2, cy2, w, h)


def detect_subject_shape(
    image_rgb: np.ndarray,
    target: Optional[str] = None,
    prefer: Optional[str] = None,
    nth: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a dict with keys:
    - mask: np.ndarray[H,W] uint8 (0 or 255) if available else None
    - bbox: normalized (x,y,w,h)
    """
    if image_rgb is None or image_rgb.size == 0:
        return {"mask": None, "bbox": (0.3, 0.3, 0.4, 0.4)}
    img = image_rgb.astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Prefer YOLO seg mask
    m, bb = _yolo_mask_and_bbox(img, target=target, prefer=prefer, nth=nth)
    if bb is not None:
        nb = _norm_bbox(bb[0], bb[1], bb[2], bb[3], w, h)
    else:
        nb = None
    if m is not None:
        return {"mask": m, "bbox": nb if nb is not None else (0.3, 0.3, 0.4, 0.4)}

    # YOLO boxes only
    if nb is not None:
        return {"mask": None, "bbox": nb}

    # Face
    fb = _face_bbox(gray)
    if fb is not None:
        return {"mask": None, "bbox": _norm_bbox(fb[0], fb[1], fb[2], fb[3], w, h)}

    # Edge saliency
    eb = _edge_saliency_bbox(gray)
    if eb is not None:
        return {"mask": None, "bbox": _norm_bbox(eb[0], eb[1], eb[2], eb[3], w, h)}

    # Fallback
    cx1 = int(w * 0.3); cy1 = int(h * 0.3); cx2 = int(w * 0.7); cy2 = int(h * 0.7)
    return {"mask": None, "bbox": _norm_bbox(cx1, cy1, cx2, cy2, w, h)}

from __future__ import annotations

from typing import Tuple


def denorm_bbox(bbox: Tuple[float, float, float, float], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    W, H = size
    return (int(x * W), int(y * H), int(w * W), int(h * H))


def bbox_center_px(bbox: Tuple[float, float, float, float], size: Tuple[int, int]) -> Tuple[int, int]:
    x, y, w, h = denorm_bbox(bbox, size)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

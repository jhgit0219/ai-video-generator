from __future__ import annotations

import importlib
import sys


def import_subject_detection():
    """Robustly import subject detection helpers regardless of load path.
    Returns a dict with optional callables: detect_subject_bbox, detect_subject_shape, detect_anchor_feature, detect_adaptive_face_anchor.
    """
    sd = {
        "detect_subject_bbox": None,
        "detect_subject_shape": None,
        "detect_anchor_feature": None,
        "detect_adaptive_face_anchor": None,
    }
    # 1) Prefer test harness loaded module (consistent with test_effects)
    try:
        mod = sys.modules.get("subject_detection_mod")
        if mod is not None:
            sd["detect_subject_bbox"] = getattr(mod, "detect_subject_bbox", None)
            sd["detect_subject_shape"] = getattr(mod, "detect_subject_shape", None)
            sd["detect_anchor_feature"] = getattr(mod, "detect_anchor_feature", None)
            sd["detect_adaptive_face_anchor"] = getattr(mod, "detect_adaptive_face_anchor", None)
            if any(sd.values()):
                return sd
    except Exception:
        pass
    # 2) Relative import (works when imported as part of package)
    try:
        from ..renderer.subject_detection import (  # type: ignore
            detect_subject_bbox as _db,
            detect_subject_shape as _ds,
            detect_anchor_feature as _da,
            detect_adaptive_face_anchor as _dafa,
        )
        sd["detect_subject_bbox"] = _db
        sd["detect_subject_shape"] = _ds
        sd["detect_anchor_feature"] = _da
        sd["detect_adaptive_face_anchor"] = _dafa
        return sd
    except Exception:
        pass
    # 3) Absolute package import
    try:
        mod = importlib.import_module("pipeline.renderer.subject_detection")
        sd["detect_subject_bbox"] = getattr(mod, "detect_subject_bbox", None)
        sd["detect_subject_shape"] = getattr(mod, "detect_subject_shape", None)
        sd["detect_anchor_feature"] = getattr(mod, "detect_anchor_feature", None)
        sd["detect_adaptive_face_anchor"] = getattr(mod, "detect_adaptive_face_anchor", None)
        if any(sd.values()):
            return sd
    except Exception:
        pass
    return sd

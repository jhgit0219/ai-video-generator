from __future__ import annotations

import importlib
import sys


def import_subject_detection():
    """Robustly import subject detection helpers regardless of load path.
    Returns a dict with optional callables: detect_subject_bbox, detect_subject_shape, detect_anchor_feature.
    """
    sd = {"detect_subject_bbox": None, "detect_subject_shape": None, "detect_anchor_feature": None}
    # 1) Prefer test harness loaded module (consistent with test_effects)
    try:
        mod = sys.modules.get("subject_detection_mod")
        if mod is not None:
            sd["detect_subject_bbox"] = getattr(mod, "detect_subject_bbox", None)
            sd["detect_subject_shape"] = getattr(mod, "detect_subject_shape", None)
            sd["detect_anchor_feature"] = getattr(mod, "detect_anchor_feature", None)
            if sd["detect_subject_bbox"] or sd["detect_subject_shape"] or sd["detect_anchor_feature"]:
                return sd
    except Exception:
        pass
    # 2) Relative import (works when imported as part of package)
    try:
        from ..renderer.subject_detection import detect_subject_bbox as _db, detect_subject_shape as _ds, detect_anchor_feature as _da  # type: ignore
        sd["detect_subject_bbox"] = _db
        sd["detect_subject_shape"] = _ds
        sd["detect_anchor_feature"] = _da
        return sd
    except Exception:
        pass
    # 3) Absolute package import
    try:
        mod = importlib.import_module("pipeline.renderer.subject_detection")
        sd["detect_subject_bbox"] = getattr(mod, "detect_subject_bbox", None)
        sd["detect_subject_shape"] = getattr(mod, "detect_subject_shape", None)
        sd["detect_anchor_feature"] = getattr(mod, "detect_anchor_feature", None)
        if sd["detect_subject_bbox"] or sd["detect_subject_shape"] or sd["detect_anchor_feature"]:
            return sd
    except Exception:
        pass
    return sd

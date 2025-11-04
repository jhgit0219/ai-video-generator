from __future__ import annotations

import math


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_back(t: float, s: float = 1.70158) -> float:
    t = max(0.0, min(1.0, t))
    t -= 1
    return (t * t * ((s + 1) * t + s) + 1)


def ease_curve(name: str, t: float) -> float:
    name = (name or "").lower()
    if name in ("linear",):
        return max(0.0, min(1.0, t))
    # default cubic in-out
    return ease_in_out_cubic(t)

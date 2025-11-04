from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ColorClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip
from utils.logger import setup_logger

from .coords import denorm_bbox
from .easing import ease_out_back

logger = setup_logger(__name__)


def apply_paneled_text(
    clip: _VideoClip,
    text: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    side: str = "right",
    panel_color: Tuple[int, int, int] = (0, 0, 0),
    panel_opacity: float = 0.6,
    border_color: Tuple[int, int, int] = (57, 255, 20),
    border_thickness: int = 4,
    fontsize: int = 56,
    font: Optional[str] = None,
    margin_px: int = 28,
    animate_in: float = 0.3,
    animate_out: float = 0.0,
    max_panel_width_frac: float = 0.6,
    min_panel_width_px: int = 560,
    min_panel_height_px: int = 160,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    cps: int = 18,
) -> _VideoClip:
    """Add a text panel with slide-in and typewriter text (letter-by-letter).
    The composite size matches the base frame to avoid black bars.
    """
    w, h = clip.size
    max_w = int(w * max(0.2, min(0.95, max_panel_width_frac)))

    # Choose font
    try:
        if font:
            pil_font = ImageFont.truetype(font, fontsize)
        else:
            pil_font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        pil_font = ImageFont.load_default()

    # Measure text
    tmp_img = Image.new("RGBA", (max_w, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp_img)
    try:
        bbox_txt = draw.multiline_textbbox((0, 0), text, font=pil_font, align="left")
    except Exception:
        bbox_txt = draw.textbbox((0, 0), text, font=pil_font, anchor=None)
    txt_w = min(max_w, bbox_txt[2] - bbox_txt[0])
    txt_h = max(1, bbox_txt[3] - bbox_txt[1])

    pad = margin_px
    desired_w = max(min_panel_width_px, txt_w + pad * 2)
    panel_w = min(max_w, desired_w)
    min_ph = max(min_panel_height_px, fontsize + pad * 2)
    panel_h = max(min_ph, txt_h + pad * 2)

    # Position
    if bbox:
        x, y, bw, bh = denorm_bbox(bbox, (w, h))
        cx = x + bw + pad
        if side == "left":
            cx = x - panel_w - pad
        px = max(0, min(w - panel_w, cx))
        py = max(0, min(h - panel_h, y))
    else:
        px = w - panel_w - pad if side == "right" else pad
        py = int(h * 0.15)

    # Timing
    start_t = max(0.0, float(start_time))
    ov_dur = float(duration) if duration is not None else float(clip.duration or 0.0)
    if ov_dur <= 0:
        ov_dur = float(clip.duration or 0.01)

    # Background + border
    panel_bg = ColorClip(size=(panel_w, panel_h), color=panel_color, duration=ov_dur).with_opacity(panel_opacity)
    panel_bg = panel_bg.with_position((px, py)).with_start(start_t)

    border = ColorClip(size=(panel_w + border_thickness * 2, panel_h + border_thickness * 2), color=border_color, duration=ov_dur)
    border = border.with_position((px - border_thickness, py - border_thickness)).with_start(start_t)

    # Typewriter text
    full_text = str(text or "")
    cps = max(1, int(cps))

    def _make_text_frame(t_local: float):
        shown = min(len(full_text), max(0, int(round(cps * t_local))))
        img = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.multiline_text((pad, pad), full_text[:shown], font=pil_font, fill=(255, 255, 255, 255), align="left")
        return np.array(img)

    txt_clip = _VideoClip(lambda t: _make_text_frame(t), duration=ov_dur).with_position((px, py)).with_start(start_t)

    # Slide-in animation
    if animate_in > 0:
        def pos_t(t):
            if t < start_t:
                return (px, py)
            if t < start_t + animate_in:
                p = ease_out_back((t - start_t) / animate_in)
                start_x = px + (50 if side == "right" else -50)
                x = start_x + (px - start_x) * p
                x = max(0, min(w - panel_w, int(x)))
                y = max(0, min(h - panel_h, int(py)))
                return (x, y)
            return (px, py)

        panel_bg = panel_bg.with_position(pos_t)
        border = border.with_position(lambda t: (pos_t(t)[0] - border_thickness, pos_t(t)[1] - border_thickness))
        txt_clip = txt_clip.with_position(lambda t: (pos_t(t)[0], pos_t(t)[1]))

    layers = [clip, border.with_opacity(1.0), panel_bg, txt_clip]
    return CompositeVideoClip(layers, size=(w, h)).with_duration(clip.duration)

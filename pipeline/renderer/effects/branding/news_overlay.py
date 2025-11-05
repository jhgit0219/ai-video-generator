"""News/publisher overlay effect with vintage newspaper styling.

Creates a vintage newspaper-style overlay with aged paper texture,
green-highlighted text phrases, and classic newspaper layout.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip
from utils.logger import setup_logger

from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)

# Brand colors for news overlay
NEWS_GREEN_HIGHLIGHT = (0, 255, 100)
SEPIA_PAPER = (240, 228, 200)
TEXT_BLACK = (30, 30, 30)


def apply_news_overlay(
    clip: _VideoClip,
    headline: str,
    subheadline: Optional[str] = None,
    body_text: Optional[str] = None,
    highlighted_phrases: Optional[List[str]] = None,
    paper_opacity: float = 0.9,
    position: Tuple[float, float] = (0.1, 0.1),
    size: Tuple[float, float] = (0.8, 0.3),
    headline_fontsize: int = 56,
    body_fontsize: int = 28,
    font_path: Optional[str] = None,
    fade_in_duration: float = 0.5,
    fade_out_duration: float = 0.5,
    start_time: float = 0.0,
    duration: Optional[float] = None,
) -> _VideoClip:
    """Apply vintage newspaper overlay with green highlights.

    Creates an aged newspaper-style text overlay with optional green
    highlighting for key phrases, vintage paper texture, and classic
    typography.

    :param clip: Input video clip to apply overlay to.
    :param headline: Main headline text.
    :param subheadline: Optional subheadline text.
    :param body_text: Optional body text content.
    :param highlighted_phrases: List of phrases to highlight in green.
    :param paper_opacity: Opacity of paper background (0-1).
    :param position: Normalized (x, y) position of newspaper overlay.
    :param size: Normalized (width, height) size of overlay.
    :param headline_fontsize: Font size for headline.
    :param body_fontsize: Font size for body text.
    :param font_path: Path to Century font file.
    :param fade_in_duration: Duration of fade in animation.
    :param fade_out_duration: Duration of fade out animation.
    :param start_time: When overlay appears.
    :param duration: How long overlay stays visible.
    :return: Video clip with news overlay effect applied.
    """
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    # Calculate overlay timing
    start_t = max(0.0, float(start_time))
    if duration is None:
        overlay_dur = clip_dur - start_t
    else:
        overlay_dur = float(duration)

    # Calculate overlay dimensions
    pos_x = int(position[0] * w)
    pos_y = int(position[1] * h)
    overlay_w = int(size[0] * w)
    overlay_h = int(size[1] * h)

    # Load Century font
    try:
        if font_path:
            headline_font = ImageFont.truetype(font_path, headline_fontsize)
            body_font = ImageFont.truetype(font_path, body_fontsize)
        else:
            for font_name in ["CENTURY.TTF", "century.ttf", "Century.ttf", "times.ttf", "arial.ttf"]:
                try:
                    headline_font = ImageFont.truetype(font_name, headline_fontsize)
                    body_font = ImageFont.truetype(font_name, body_fontsize)
                    logger.debug(f"[news_overlay] Loaded font: {font_name}")
                    break
                except Exception:
                    continue
            else:
                headline_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                logger.warning("[news_overlay] Century/Times font not found, using default")
    except Exception as e:
        logger.warning(f"[news_overlay] Font loading failed: {e}, using default")
        headline_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    # Pre-generate newspaper texture
    def create_paper_texture(width: int, height: int) -> Image.Image:
        """Create aged paper texture with noise and sepia tone."""
        # Base paper color
        paper = Image.new("RGBA", (width, height), (*SEPIA_PAPER, 255))

        # Add noise for texture
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise, "RGB")
        paper = Image.blend(paper.convert("RGB"), noise_img, 0.15)
        paper = paper.convert("RGBA")

        # Add slight grain
        paper = paper.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Add vintage stains (darker spots)
        stain_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        stain_draw = ImageDraw.Draw(stain_layer)
        for _ in range(5):
            stain_x = np.random.randint(0, width)
            stain_y = np.random.randint(0, height)
            stain_r = np.random.randint(20, 60)
            stain_alpha = np.random.randint(10, 30)
            stain_draw.ellipse(
                [stain_x - stain_r, stain_y - stain_r, stain_x + stain_r, stain_y + stain_r],
                fill=(139, 90, 43, stain_alpha),
            )
        stain_layer = stain_layer.filter(ImageFilter.GaussianBlur(radius=15))
        paper = Image.alpha_composite(paper, stain_layer)

        return paper

    paper_texture = create_paper_texture(overlay_w, overlay_h)

    def render_text_with_highlights(
        img: Image.Image,
        text: str,
        font: ImageFont.FreeTypeFont,
        start_y: int,
        highlighted_phrases: Optional[List[str]] = None,
    ) -> int:
        """Render text with optional green highlights, return final y position."""
        if not text:
            return start_y

        draw = ImageDraw.Draw(img)
        line_spacing = 10
        max_width = overlay_w - 40
        words = text.split()
        lines = []
        current_line = []

        # Word wrap
        for word in words:
            test_line = " ".join(current_line + [word])
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]
            except Exception:
                line_width = len(test_line) * (body_fontsize // 2)

            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        # Render lines with highlights
        y = start_y
        for line in lines:
            x = 20

            # Check if line contains highlighted phrases
            should_highlight = False
            if highlighted_phrases:
                for phrase in highlighted_phrases:
                    if phrase.lower() in line.lower():
                        should_highlight = True
                        break

            if should_highlight:
                # Draw highlight background
                try:
                    line_bbox = draw.textbbox((x, y), line, font=font)
                    highlight_rect = [
                        line_bbox[0] - 5,
                        line_bbox[1] - 3,
                        line_bbox[2] + 5,
                        line_bbox[3] + 3,
                    ]
                    draw.rectangle(highlight_rect, fill=(*NEWS_GREEN_HIGHLIGHT, 150))
                except Exception:
                    pass

            # Draw text
            draw.text((x, y), line, font=font, fill=TEXT_BLACK)

            try:
                line_bbox = draw.textbbox((x, y), line, font=font)
                y += (line_bbox[3] - line_bbox[1]) + line_spacing
            except Exception:
                y += body_fontsize + line_spacing

        return y

    def make_overlay_frame(t: float):
        """Generate news overlay frame."""
        if t < start_t or t > start_t + overlay_dur:
            # Outside active time range
            return np.zeros((h, w, 4), dtype=np.uint8)

        local_t = t - start_t

        # Calculate fade alpha
        if local_t < fade_in_duration:
            alpha = ease_in_out_cubic(local_t / fade_in_duration) if fade_in_duration > 0 else 1.0
        elif local_t > overlay_dur - fade_out_duration:
            fade_progress = (overlay_dur - local_t) / fade_out_duration if fade_out_duration > 0 else 1.0
            alpha = ease_in_out_cubic(fade_progress)
        else:
            alpha = 1.0

        # Create overlay with paper texture
        overlay = paper_texture.copy()
        draw = ImageDraw.Draw(overlay)

        # Draw border
        border_thickness = 4
        draw.rectangle(
            [0, 0, overlay_w - 1, overlay_h - 1],
            outline=TEXT_BLACK,
            width=border_thickness,
        )

        # Render headline
        current_y = 20
        if headline:
            # Center headline
            try:
                headline_bbox = draw.textbbox((0, 0), headline, font=headline_font)
                headline_w = headline_bbox[2] - headline_bbox[0]
                headline_x = (overlay_w - headline_w) // 2
            except Exception:
                headline_x = 20

            draw.text((headline_x, current_y), headline, font=headline_font, fill=TEXT_BLACK)
            try:
                headline_bbox = draw.textbbox((headline_x, current_y), headline, font=headline_font)
                current_y = headline_bbox[3] + 15
            except Exception:
                current_y += headline_fontsize + 15

            # Underline
            draw.line([20, current_y, overlay_w - 20, current_y], fill=TEXT_BLACK, width=2)
            current_y += 15

        # Render subheadline
        if subheadline:
            current_y = render_text_with_highlights(overlay, subheadline, body_font, current_y, highlighted_phrases)
            current_y += 10

        # Render body text
        if body_text:
            render_text_with_highlights(overlay, body_text, body_font, current_y, highlighted_phrases)

        # Apply overall opacity and fade
        overlay_alpha = int(255 * paper_opacity * alpha)
        overlay.putalpha(overlay_alpha)

        # Create full-size frame
        full_frame = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        full_frame.paste(overlay, (pos_x, pos_y))

        return np.array(full_frame)

    # Create overlay clip
    overlay_clip = _VideoClip(make_overlay_frame, duration=clip_dur)

    # Composite
    result = CompositeVideoClip([clip, overlay_clip], size=(w, h))
    return result.with_duration(clip_dur)

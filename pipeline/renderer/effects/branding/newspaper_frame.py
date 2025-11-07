"""Newspaper frame effect - frames video in vintage newspaper with cut-out."""
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy import VideoClip, CompositeVideoClip

from config import RENDER_SIZE

logger = logging.getLogger(__name__)


def apply_newspaper_frame(
    clip: VideoClip,
    headline: str = "BREAKING NEWS",
    subheadline: str = "Historic moment captured on film",
    body_text: str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore.",
    cutout_width_ratio: float = 0.6,
    cutout_height_ratio: float = 0.6,
    paper_color: tuple = (245, 235, 210),
    text_color: tuple = (30, 30, 30),
    zoom_to_fullscreen: bool = False,
    zoom_start_time: float = 0.0,
    zoom_duration: float = 1.5,
) -> VideoClip:
    """Frame video clip inside vintage newspaper layout with cut-out box.

    Creates full-screen newspaper background with headlines, body text columns,
    and a cut-out rectangle in the center where the video clip is displayed.
    Output is always RENDER_SIZE (1920x1080).

    The clip passed in should already have geometric effects applied (zoom_on_subject, etc.).
    This effect takes the final transformed frame and places it in the newspaper cut-out.

    Args:
        clip: Input video clip to frame inside newspaper (should be pre-transformed).
        headline: Large headline text at top.
        subheadline: Smaller headline below main headline.
        body_text: Body text for newspaper columns around the frame.
        cutout_width_ratio: Width of video cut-out as ratio of frame width (0-1).
        cutout_height_ratio: Height of video cut-out as ratio of frame height (0-1).
        paper_color: RGB color for aged newspaper background.
        text_color: RGB color for newspaper text.
        zoom_to_fullscreen: If True, animate from newspaper frame to full-screen video.
        zoom_start_time: When to start the zoom-to-fullscreen animation (seconds).
        zoom_duration: Duration of the zoom-to-fullscreen animation (seconds).

    Returns:
        Composited VideoClip with newspaper frame and video in cut-out at RENDER_SIZE.
    """
    # Always output at RENDER_SIZE
    w, h = RENDER_SIZE
    duration = clip.duration or 1.0

    # Calculate cut-out dimensions
    cutout_w = int(w * cutout_width_ratio)
    cutout_h = int(h * cutout_height_ratio)

    # Calculate cut-out position (centered)
    cutout_x = (w - cutout_w) // 2
    cutout_y = (h - cutout_h) // 2

    logger.info(
        f"[newspaper_frame] Creating newspaper frame {w}x{h} "
        f"with {cutout_w}x{cutout_h} cut-out at ({cutout_x}, {cutout_y})"
    )

    # Load fonts
    try:
        headline_font = ImageFont.truetype("arial.ttf", size=72)
        subheadline_font = ImageFont.truetype("arial.ttf", size=36)
        body_font = ImageFont.truetype("arial.ttf", size=20)
    except Exception:
        logger.warning("[newspaper_frame] Failed to load TTF fonts, using default")
        headline_font = ImageFont.load_default()
        subheadline_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    def make_newspaper_frame(t):
        """Create newspaper background frame at time t."""
        # Create aged paper background
        paper = Image.new("RGB", (w, h), paper_color)
        draw = ImageDraw.Draw(paper)

        # Add paper texture noise
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        paper_arr = np.array(paper, dtype=np.int16)
        paper_arr = np.clip(paper_arr + noise, 0, 255).astype(np.uint8)
        paper = Image.fromarray(paper_arr)
        draw = ImageDraw.Draw(paper)

        # Add border
        border_width = 20
        draw.rectangle(
            [border_width, border_width, w - border_width, h - border_width],
            outline=text_color,
            width=3,
        )

        # Draw headline at top
        y_pos = 40
        headline_wrapped = _wrap_text(headline, headline_font, w - 100, draw)
        for line in headline_wrapped:
            bbox = draw.textbbox((0, 0), line, font=headline_font)
            line_w = bbox[2] - bbox[0]
            x_pos = (w - line_w) // 2
            draw.text((x_pos, y_pos), line, fill=text_color, font=headline_font)
            y_pos += bbox[3] - bbox[1] + 10

        # Draw subheadline
        y_pos += 20
        subheadline_wrapped = _wrap_text(subheadline, subheadline_font, w - 100, draw)
        for line in subheadline_wrapped:
            bbox = draw.textbbox((0, 0), line, font=subheadline_font)
            line_w = bbox[2] - bbox[0]
            x_pos = (w - line_w) // 2
            draw.text((x_pos, y_pos), line, fill=text_color, font=subheadline_font)
            y_pos += bbox[3] - bbox[1] + 5

        # Draw photo cut-out box with border
        draw.rectangle(
            [cutout_x - 5, cutout_y - 5, cutout_x + cutout_w + 5, cutout_y + cutout_h + 5],
            outline=text_color,
            width=3,
        )

        # Add caption placeholder below cut-out
        caption_y = cutout_y + cutout_h + 15
        caption_text = "Photo caption area - Historic scene depicted above"
        caption_wrapped = _wrap_text(caption_text, body_font, cutout_w, draw)
        for line in caption_wrapped:
            bbox = draw.textbbox((0, 0), line, font=body_font)
            line_w = bbox[2] - bbox[0]
            x_pos = cutout_x + (cutout_w - line_w) // 2
            draw.text((x_pos, caption_y), line, fill=text_color, font=body_font)
            caption_y += bbox[3] - bbox[1] + 5

        # Draw body text columns on left side (above and below cut-out)
        left_column_x = 40
        left_column_w = cutout_x - 60

        # Left top column
        if cutout_y > 200:
            _draw_text_column(
                draw,
                body_text,
                left_column_x,
                y_pos + 40,
                left_column_w,
                cutout_y - y_pos - 60,
                body_font,
                text_color,
            )

        # Left bottom column
        if caption_y + 40 < h - 60:
            _draw_text_column(
                draw,
                body_text,
                left_column_x,
                max(cutout_y + cutout_h + 100, caption_y + 40),
                left_column_w,
                h - max(cutout_y + cutout_h + 100, caption_y + 40) - 60,
                body_font,
                text_color,
            )

        # Draw body text columns on right side (above and below cut-out)
        right_column_x = cutout_x + cutout_w + 20
        right_column_w = w - right_column_x - 40

        # Right top column
        if cutout_y > 200:
            _draw_text_column(
                draw,
                body_text,
                right_column_x,
                y_pos + 40,
                right_column_w,
                cutout_y - y_pos - 60,
                body_font,
                text_color,
            )

        # Right bottom column
        if caption_y + 40 < h - 60:
            _draw_text_column(
                draw,
                body_text,
                right_column_x,
                max(cutout_y + cutout_h + 100, caption_y + 40),
                right_column_w,
                h - max(cutout_y + cutout_h + 100, caption_y + 40) - 60,
                body_font,
                text_color,
            )

        return np.array(paper)

    # Create newspaper background clip
    newspaper_clip = VideoClip(make_newspaper_frame, duration=duration)

    # Detect face anchor point for content-aware cropping
    from pipeline.renderer.subject_detection import detect_adaptive_face_anchor, detect_subject_shape, detect_anchor_feature
    from ..coords import denorm_bbox

    anchor_px = None

    try:
        # Get first frame to detect subject
        frame = clip.get_frame(0.001)
        frame_h, frame_w = clip.size[1], clip.size[0]

        # Try adaptive face anchor first (prioritize this)
        anchor_px = detect_adaptive_face_anchor(frame, target="person")
        if anchor_px:
            logger.info(f"[newspaper_frame] Detected adaptive face anchor at {anchor_px}")
        else:
            # Fallback to mask-based detection
            logger.info(f"[newspaper_frame] Adaptive face anchor failed, trying mask-based detection")
            result = detect_subject_shape(frame, target="person")
            if result and result.get("mask") is not None:
                subject_bbox_norm = result.get("bbox")

                # Convert normalized bbox to absolute pixel coordinates
                if subject_bbox_norm:
                    x, y, bw, bh = denorm_bbox(subject_bbox_norm, (frame_w, frame_h))
                    subject_bbox_abs = (x, y, x + bw, y + bh)

                    # Try to detect face anchor within subject
                    anchor_px = detect_anchor_feature(
                        frame,
                        subject_mask=result["mask"],
                        subject_bbox=subject_bbox_abs,
                        feature_type="face"
                    )

                    if anchor_px:
                        logger.info(f"[newspaper_frame] Detected face anchor at {anchor_px}")
                    else:
                        # Final fallback: Use upper portion of subject bbox
                        anchor_px = (int(x + bw / 2), int(y + bh * 0.25))
                        logger.info(f"[newspaper_frame] Face detection failed, using bbox fallback anchor at {anchor_px}")
    except Exception as e:
        logger.warning(f"[newspaper_frame] Subject detection failed: {e}")

    # Convert anchor to normalized coordinates if detected
    anchor_norm = None
    if anchor_px:
        frame_h, frame_w = clip.size[1], clip.size[0]
        anchor_norm = (anchor_px[0] / frame_w, anchor_px[1] / frame_h)
        logger.info(f"[newspaper_frame] Normalized anchor: {anchor_norm}")

    # Resize clip to cut-out size using zoom-to-cover with face anchor
    from pipeline.renderer.image_utils import resize_and_crop

    def resize_to_cutout_frame(get_frame, t):
        """Resize frame to cover cut-out dimensions with face-aware cropping."""
        from PIL import Image as PILImage

        frame = get_frame(t)

        # Convert to PIL Image
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img = PILImage.fromarray(frame)

        # Convert normalized anchor to pixel coordinates for this frame
        anchor_px_current = None
        if anchor_norm:
            anchor_px_current = (
                int(anchor_norm[0] * img.width),
                int(anchor_norm[1] * img.height)
            )

        # Use existing resize_and_crop function with anchor
        img_resized, _ = resize_and_crop(
            img,
            cutout_w,
            cutout_h,
            anchor_point=anchor_px_current
        )

        return np.array(img_resized)

    video_resized = clip.transform(resize_to_cutout_frame)

    # Position video in cut-out
    video_positioned = video_resized.with_position((cutout_x, cutout_y))

    # Composite newspaper background with video in cut-out
    composite = CompositeVideoClip([newspaper_clip, video_positioned], size=(w, h))

    # If zoom_to_fullscreen is enabled, animate from newspaper frame to full-screen video
    if zoom_to_fullscreen:
        logger.info(
            f"[newspaper_frame] Adding zoom-to-fullscreen animation "
            f"starting at {zoom_start_time}s for {zoom_duration}s"
        )
        result = _apply_zoom_to_fullscreen(
            composite,
            clip,
            cutout_x,
            cutout_y,
            cutout_w,
            cutout_h,
            zoom_start_time,
            zoom_duration,
        )
    else:
        result = composite

    logger.info(f"[newspaper_frame] Newspaper frame effect applied")

    return result


def _wrap_text(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list:
    """Wrap text to fit within max_width.

    Args:
        text: Text to wrap.
        font: Font to use for measuring.
        max_width: Maximum line width in pixels.
        draw: ImageDraw instance for text measurement.

    Returns:
        List of wrapped text lines.
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
        except Exception:
            # Fallback estimation
            line_width = len(test_line) * 10

        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _draw_text_column(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    max_height: int,
    font: ImageFont.ImageFont,
    color: tuple,
) -> None:
    """Draw wrapped text column in specified area.

    Args:
        draw: ImageDraw instance to draw on.
        text: Text content to draw.
        x: Column x position.
        y: Column y position (top).
        width: Column width.
        max_height: Maximum column height.
        font: Font to use.
        color: Text color.
    """
    wrapped_lines = _wrap_text(text, font, width, draw)

    y_pos = y
    for line in wrapped_lines:
        # Check if we've exceeded max height
        bbox = draw.textbbox((0, 0), line, font=font)
        line_h = bbox[3] - bbox[1]

        if y_pos + line_h > y + max_height:
            break

        draw.text((x, y_pos), line, fill=color, font=font)
        y_pos += line_h + 3

    return None


def _apply_zoom_to_fullscreen(
    newspaper_composite: VideoClip,
    original_clip: VideoClip,
    cutout_x: int,
    cutout_y: int,
    cutout_w: int,
    cutout_h: int,
    zoom_start_time: float,
    zoom_duration: float,
) -> VideoClip:
    """Animate from newspaper frame to full-screen video.

    Zooms from the cut-out position to fill the entire frame,
    while fading out the newspaper background.

    Args:
        newspaper_composite: Composited newspaper frame with video in cut-out.
        original_clip: Original video clip (for full-screen display).
        cutout_x: X position of cut-out.
        cutout_y: Y position of cut-out.
        cutout_w: Width of cut-out.
        cutout_h: Height of cut-out.
        zoom_start_time: When to start the zoom animation (seconds).
        zoom_duration: Duration of the zoom animation (seconds).

    Returns:
        VideoClip with zoom-to-fullscreen animation applied.
    """
    from PIL import Image as PILImage
    from pipeline.renderer.effects.easing import ease_in_out_cubic

    w, h = RENDER_SIZE
    duration = newspaper_composite.duration

    # Calculate scale factor needed to go from cut-out size to full-screen
    scale_x = w / cutout_w
    scale_y = h / cutout_h
    scale = max(scale_x, scale_y)  # Use max to ensure full coverage

    def transform_frame(get_frame, t):
        """Transform frame with zoom animation."""
        # Before zoom starts: show newspaper composite
        if t < zoom_start_time:
            return get_frame(t)

        # After zoom completes: show full-screen video
        zoom_end_time = zoom_start_time + zoom_duration
        if t >= zoom_end_time:
            # Show original clip resized to fill frame
            video_frame = original_clip.get_frame(min(t, original_clip.duration - 0.001))
            if video_frame.shape[:2] != (h, w):
                img = PILImage.fromarray(video_frame.astype('uint8'))
                img = img.resize((w, h), PILImage.Resampling.LANCZOS)
                video_frame = np.array(img)
            return video_frame

        # During zoom: interpolate from cut-out to full-screen
        progress = (t - zoom_start_time) / zoom_duration
        progress_eased = ease_in_out_cubic(progress)

        # Get both frames
        newspaper_frame = get_frame(t)
        video_frame = original_clip.get_frame(min(t, original_clip.duration - 0.001))

        # Resize video frame to full-screen size
        if video_frame.shape[:2] != (h, w):
            img = PILImage.fromarray(video_frame.astype('uint8'))
            img = img.resize((w, h), PILImage.Resampling.LANCZOS)
            video_frame = np.array(img)

        # Calculate current scale and position
        current_scale = 1.0 + (scale - 1.0) * progress_eased

        # Calculate zoom center (center of cut-out)
        center_x = cutout_x + cutout_w // 2
        center_y = cutout_y + cutout_h // 2

        # Transform newspaper frame (zoom from cut-out center)
        img_newspaper = PILImage.fromarray(newspaper_frame.astype('uint8'))
        zoom_w = int(w * current_scale)
        zoom_h = int(h * current_scale)
        img_newspaper_zoomed = img_newspaper.resize((zoom_w, zoom_h), PILImage.Resampling.LANCZOS)

        # Calculate crop position (zoom from cut-out center)
        crop_x = int(center_x * current_scale - w // 2)
        crop_y = int(center_y * current_scale - h // 2)
        crop_x = max(0, min(crop_x, zoom_w - w))
        crop_y = max(0, min(crop_y, zoom_h - h))

        img_newspaper_cropped = img_newspaper_zoomed.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        newspaper_frame_transformed = np.array(img_newspaper_cropped).astype(np.float32)

        # Fade from newspaper to video
        newspaper_opacity = 1.0 - progress_eased
        video_opacity = progress_eased

        # Composite
        result = (
            newspaper_frame_transformed * newspaper_opacity +
            video_frame.astype(np.float32) * video_opacity
        ).astype(np.uint8)

        return result

    return newspaper_composite.transform(transform_frame)

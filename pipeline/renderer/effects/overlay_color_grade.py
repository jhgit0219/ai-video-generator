"""
Color grading effects for cinematic look.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from moviepy.video.VideoClip import VideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("[overlay_color_grade] MoviePy not available")


def apply_color_grade(
    clip: VideoClip,
    temperature: float = 0.0,
    saturation: float = 1.0,
    contrast: float = 1.0,
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> VideoClip:
    """
    Apply cinematic color grading to the clip.

    Args:
        clip: Input video clip
        temperature: Color temperature shift (-1.0=cooler/blue, 0.0=neutral, +1.0=warmer/orange)
        saturation: Color saturation multiplier (0.0=grayscale, 1.0=normal, 1.5=vivid)
        contrast: Contrast multiplier (0.5=low contrast, 1.0=normal, 1.5=high contrast)
        tint: RGB multipliers for color tint (e.g., (1.1, 1.0, 0.9) for warm tint)

    Returns:
        Color graded video clip
    """
    if not MOVIEPY_AVAILABLE:
        logger.warning("[overlay_color_grade] MoviePy not available, returning clip unchanged")
        return clip

    def apply_grade_frame(image):
        """Apply color grading to a single frame."""
        img_pil = Image.fromarray(image)

        # Apply saturation
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(saturation)

        # Apply contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(contrast)

        # Convert to numpy for temperature and tint adjustments
        img_arr = np.array(img_pil).astype(np.float32)

        # Apply temperature shift (adjust blue/orange balance)
        if temperature != 0.0:
            # Positive temperature: add orange, reduce blue
            # Negative temperature: add blue, reduce orange
            temp_factor = temperature * 30  # Scale to visible range
            img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + temp_factor, 0, 255)  # Red
            img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] + temp_factor * 0.5, 0, 255)  # Green
            img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] - temp_factor, 0, 255)  # Blue

        # Apply tint (RGB multipliers)
        if tint != (1.0, 1.0, 1.0):
            img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] * tint[0], 0, 255)
            img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] * tint[1], 0, 255)
            img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] * tint[2], 0, 255)

        return img_arr.astype(np.uint8)

    logger.info(f"[overlay_color_grade] Applying color grade (temp={temperature}, sat={saturation}, contrast={contrast})")
    return clip.fl_image(apply_grade_frame)


def apply_warm_grade(clip: VideoClip, intensity: float = 0.5) -> VideoClip:
    """Apply warm/golden color grade (like sunset or vintage film)."""
    return apply_color_grade(
        clip,
        temperature=intensity * 0.7,
        saturation=1.1,
        contrast=1.05,
        tint=(1.1, 1.0, 0.85),
    )


def apply_cool_grade(clip: VideoClip, intensity: float = 0.5) -> VideoClip:
    """Apply cool/teal color grade (like thriller or sci-fi)."""
    return apply_color_grade(
        clip,
        temperature=-intensity * 0.7,
        saturation=1.15,
        contrast=1.1,
        tint=(0.9, 1.0, 1.1),
    )


def apply_desaturated_grade(clip: VideoClip, intensity: float = 0.6) -> VideoClip:
    """Apply desaturated look (like documentary or dramatic scene)."""
    return apply_color_grade(
        clip,
        saturation=1.0 - (intensity * 0.6),  # Reduce saturation
        contrast=1.1,  # Slight contrast boost to compensate
    )


def apply_cinematic_teal_orange(clip: VideoClip) -> VideoClip:
    """Apply the classic Hollywood teal-and-orange color grade."""
    return apply_color_grade(
        clip,
        temperature=0.3,  # Slight warm bias
        saturation=1.2,
        contrast=1.15,
        tint=(1.15, 1.0, 1.05),  # Boost orange in highlights, teal in shadows
    )

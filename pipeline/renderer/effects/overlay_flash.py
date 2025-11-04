"""
Flash/pulse overlay effects for dramatic moments.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageEnhance
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from moviepy.video.VideoClip import VideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("[overlay_flash] MoviePy not available")


def apply_flash_pulse(
    clip: VideoClip,
    intensity: float = 0.8,
    frequency: float = 2.0,
    flash_at: Optional[float] = None,
) -> VideoClip:
    """
    Apply a flash/pulse effect that brightens the image rhythmically or at a specific moment.

    Args:
        clip: Input video clip
        intensity: Flash brightness intensity (0.0-1.0), 0.8 = 80% brighter at peak
        frequency: Pulses per second (e.g., 2.0 = 2 pulses/second, 0.5 = one pulse every 2s)
        flash_at: Optional specific time (seconds) to flash, overrides frequency-based pulse

    Returns:
        Video clip with flash/pulse overlay
    """
    if not MOVIEPY_AVAILABLE:
        logger.warning("[overlay_flash] MoviePy not available, returning clip unchanged")
        return clip

    duration = clip.duration

    def apply_flash_frame(image, t):
        """Apply flash brightness modulation to a frame."""
        # Convert to PIL for easier brightness adjustment
        img_pil = Image.fromarray(image)

        # Calculate flash intensity at time t
        if flash_at is not None:
            # Single flash at specified time
            # Use gaussian-like curve centered at flash_at
            time_diff = abs(t - flash_at)
            flash_width = 0.15  # Flash lasts ~0.3s total (0.15s on each side)
            if time_diff < flash_width:
                # Gaussian falloff
                flash_strength = intensity * np.exp(-((time_diff / flash_width) ** 2) * 4)
            else:
                flash_strength = 0.0
        else:
            # Periodic pulse based on frequency
            # Use sine wave for smooth pulsing
            pulse = np.sin(2 * np.pi * frequency * t)
            # Map sine wave [-1,1] to [0, intensity]
            # Pulse peaks at intensity, valleys at 0 (no change)
            flash_strength = intensity * max(0, pulse)  # Only positive half of sine

        # Apply brightness boost
        if flash_strength > 0.01:
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(1.0 + flash_strength)

        return np.array(img_pil)

    logger.info(f"[overlay_flash] Applying flash effect (intensity={intensity}, frequency={frequency}, flash_at={flash_at})")
    return clip.fl_image(apply_flash_frame)


def apply_quick_flash(
    clip: VideoClip,
    flash_time: float = 0.0,
    intensity: float = 1.0,
) -> VideoClip:
    """
    Apply a single quick flash at a specific moment (like a camera flash or explosion).

    Args:
        clip: Input video clip
        flash_time: Time in seconds when flash occurs
        intensity: Flash brightness (1.0 = double brightness, 1.5 = 2.5x brightness)

    Returns:
        Video clip with quick flash at specified time
    """
    return apply_flash_pulse(clip, intensity=intensity, flash_at=flash_time)


def apply_strobe_effect(
    clip: VideoClip,
    frequency: float = 4.0,
    intensity: float = 0.6,
) -> VideoClip:
    """
    Apply a strobe/flickering effect for dramatic or tense moments.

    Args:
        clip: Input video clip
        frequency: Strobe flashes per second
        intensity: Flash intensity (0.0-1.0)

    Returns:
        Video clip with strobe effect
    """
    return apply_flash_pulse(clip, intensity=intensity, frequency=frequency)

"""Branded transition effects using brand colors and textures.

Provides transition presets using purple/violet and neon green brand colors
with various styles: wipes, flashes, noise, graffiti, luminance effects.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from moviepy import CompositeVideoClip, ColorClip
from moviepy.video.VideoClip import VideoClip as _VideoClip
from utils.logger import setup_logger

from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)

# Brand colors for transitions
PURPLE_BRAND = (139, 0, 255)
VIOLET_BRAND = (148, 0, 211)
NEON_GREEN = (0, 255, 0)
CYAN_BRAND = (0, 255, 255)

TransitionType = Literal[
    "leak_1", "leak_2", "neon_green", "dashed_graffiti",
    "noise_2", "old", "luminance", "purple_wipe", "green_flash"
]


def apply_branded_transition(
    clip: _VideoClip,
    transition_type: TransitionType = "purple_wipe",
    direction: Literal["in", "out", "both"] = "in",
    duration: float = 1.0,
    intensity: float = 1.0,
) -> _VideoClip:
    """Apply branded transition effect.

    Creates various transition effects using brand colors and styles:
    - leak_1, leak_2: Purple liquid wipe effects
    - neon_green: Bright green flash transition
    - dashed_graffiti: Graffiti-style wipe with dashed lines
    - noise_2: Noise/static transition
    - old: Vintage film style transition
    - luminance: Brightness-based fade
    - purple_wipe: Simple purple color wipe
    - green_flash: Quick neon green flash

    :param clip: Input video clip.
    :param transition_type: Type of transition to apply.
    :param direction: "in" (fade in), "out" (fade out), or "both".
    :param duration: Duration of transition in seconds.
    :param intensity: Intensity of effect (0-1).
    :return: Video clip with branded transition applied.
    """
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    if transition_type == "purple_wipe":
        return _apply_purple_wipe(clip, direction, duration, intensity)
    elif transition_type == "green_flash":
        return _apply_green_flash(clip, direction, duration, intensity)
    elif transition_type in ("leak_1", "leak_2"):
        return _apply_leak_transition(clip, direction, duration, intensity, variant=transition_type)
    elif transition_type == "neon_green":
        return _apply_neon_transition(clip, direction, duration, intensity)
    elif transition_type == "dashed_graffiti":
        return _apply_graffiti_transition(clip, direction, duration, intensity)
    elif transition_type == "noise_2":
        return _apply_noise_transition(clip, direction, duration, intensity)
    elif transition_type == "old":
        return _apply_old_film_transition(clip, direction, duration, intensity)
    elif transition_type == "luminance":
        return _apply_luminance_transition(clip, direction, duration, intensity)
    else:
        logger.warning(f"[branded_transition] Unknown transition type: {transition_type}, using purple_wipe")
        return _apply_purple_wipe(clip, direction, duration, intensity)


def _apply_purple_wipe(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Simple purple color wipe transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    def make_mask_frame(t: float):
        """Generate wipe mask."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:  # both
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)
        alpha = int(255 * progress)

        # Create vertical wipe
        mask = np.zeros((h, w, 4), dtype=np.uint8)
        wipe_x = int(w * progress)
        mask[:, :wipe_x] = (*PURPLE_BRAND, alpha)

        return mask

    overlay_clip = _VideoClip(make_mask_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_green_flash(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Quick neon green flash transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    flash_dur = min(duration, 0.3)  # Quick flash

    def make_flash_frame(t: float):
        """Generate flash overlay."""
        alpha = 0

        if direction in ("in", "both") and t < flash_dur:
            progress = 1.0 - (t / flash_dur)
            alpha = int(255 * intensity * progress * progress)
        elif direction in ("out", "both") and t > clip_dur - flash_dur:
            progress = (clip_dur - t) / flash_dur
            alpha = int(255 * intensity * (1.0 - progress * progress))

        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        if alpha > 0:
            overlay[:, :] = (*NEON_GREEN, alpha)

        return overlay

    overlay_clip = _VideoClip(make_flash_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_leak_transition(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
    variant: str,
) -> _VideoClip:
    """Purple liquid leak-style transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    # Variant determines leak pattern
    is_leak_2 = (variant == "leak_2")

    def make_leak_frame(t: float):
        """Generate liquid leak effect."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:  # both
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)

        # Create leak pattern
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        base_alpha = int(255 * intensity * (1.0 - progress))
        if base_alpha > 0:
            # Draw dripping pattern
            num_drips = 8 if is_leak_2 else 5
            for i in range(num_drips):
                drip_x = int(w * (i + 0.5) / num_drips)
                drip_length = int(h * (1.0 - progress) * (0.5 + np.random.random() * 0.5))

                color = VIOLET_BRAND if is_leak_2 else PURPLE_BRAND
                drip_color = (*color, base_alpha)

                # Draw drip shape
                drip_width = int(w / num_drips * 0.8)
                draw.ellipse(
                    [drip_x - drip_width // 2, 0, drip_x + drip_width // 2, drip_length],
                    fill=drip_color,
                )

            # Apply blur for liquid effect
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=8))

        return np.array(overlay)

    overlay_clip = _VideoClip(make_leak_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_neon_transition(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Neon green glow transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    def make_neon_frame(t: float):
        """Generate neon glow."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)
        alpha = int(255 * intensity * (1.0 - progress) * 0.7)

        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        if alpha > 0:
            overlay[:, :] = (*NEON_GREEN, alpha)

        return overlay

    overlay_clip = _VideoClip(make_neon_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_graffiti_transition(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Dashed graffiti-style transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    def make_graffiti_frame(t: float):
        """Generate graffiti pattern."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        base_alpha = int(255 * intensity * (1.0 - progress))
        if base_alpha > 0:
            # Draw dashed lines pattern
            dash_length = 20
            gap_length = 10
            num_lines = int(h * (1.0 - progress))

            for y in range(0, num_lines, dash_length + gap_length):
                for x in range(0, w, dash_length * 2):
                    draw.line(
                        [(x, y), (x + dash_length, y)],
                        fill=(*PURPLE_BRAND, base_alpha),
                        width=3,
                    )

        return np.array(overlay)

    overlay_clip = _VideoClip(make_graffiti_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_noise_transition(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Noise/static transition effect."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    def make_noise_frame(t: float):
        """Generate noise pattern."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)

        base_alpha = int(255 * intensity * (1.0 - progress))
        if base_alpha > 0:
            # Generate random noise
            noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[:, :, :3] = noise
            overlay[:, :, 3] = base_alpha
            return overlay
        else:
            return np.zeros((h, w, 4), dtype=np.uint8)

    overlay_clip = _VideoClip(make_noise_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_old_film_transition(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Vintage film-style transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    def make_old_film_frame(t: float):
        """Generate vintage film effect."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)

        # Create sepia overlay with vignette
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        base_alpha = int(255 * intensity * (1.0 - progress) * 0.6)
        if base_alpha > 0:
            # Sepia tint
            sepia_color = (112, 66, 20, base_alpha)
            draw.rectangle([0, 0, w, h], fill=sepia_color)

        return np.array(overlay)

    overlay_clip = _VideoClip(make_old_film_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)


def _apply_luminance_transition(
    clip: _VideoClip,
    direction: str,
    duration: float,
    intensity: float,
) -> _VideoClip:
    """Luminance-based brightness transition."""
    w, h = clip.size
    clip_dur = float(clip.duration or 1.0)

    def make_luminance_frame(t: float):
        """Generate brightness overlay."""
        if direction == "in":
            if t < duration:
                progress = t / duration
            else:
                progress = 1.0
        elif direction == "out":
            if t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0
        else:
            if t < duration:
                progress = t / duration
            elif t > clip_dur - duration:
                progress = (clip_dur - t) / duration
            else:
                progress = 1.0

        progress = ease_in_out_cubic(progress)

        # Brightness overlay (white fade)
        base_alpha = int(255 * intensity * (1.0 - progress))
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        if base_alpha > 0:
            overlay[:, :] = (255, 255, 255, base_alpha)

        return overlay

    overlay_clip = _VideoClip(make_luminance_frame, duration=clip_dur)
    return CompositeVideoClip([clip, overlay_clip], size=(w, h)).with_duration(clip_dur)

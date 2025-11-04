from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter
from moviepy.video.VideoClip import VideoClip
from utils.logger import setup_logger

logger = setup_logger(__name__)


def apply_nightvision(
    clip: VideoClip,
    green_tint: Tuple[int, int, int] = (30, 180, 30),
    intensity: float = 0.65,
    scanline_opacity: float = 0.12,
    scanline_spacing: int = 4,
    vignette_strength: float = 0.35,
    noise_amount: float = 0.06,
    animate_scanlines: bool = True,
    blur_amount: float = 0.4,
) -> VideoClip:
    """Apply military-style nightvision goggle effect with green tint, scan lines, and grain.

    Args:
        clip: Input video clip
        green_tint: RGB color for the green overlay (default military green)
        intensity: Strength of the green tint overlay (0-1)
        scanline_opacity: Opacity of horizontal scan lines (0-1)
        scanline_spacing: Pixel spacing between scan lines
        vignette_strength: Edge darkening strength (0-1)
        noise_amount: Amount of film grain/noise (0-1)
        animate_scanlines: If True, scan lines slowly scroll downward
        blur_amount: Amount of blur for fuzzy CRT effect (0-2)
    """
    w, h = clip.size
    dur = float(clip.duration or 1.0)

    # Pre-compute vignette mask (circular darkening like goggle view)
    vignette_mask = _create_vignette_mask(w, h, vignette_strength)

    # Pre-compute static scanline pattern
    base_scanline_pattern = _create_scanline_pattern(w, h, scanline_spacing, scanline_opacity)

    def apply_effect(get_frame, t):
        frame = get_frame(t)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Convert to float for processing
        img_arr = frame.astype(np.float32)

        # 1. Apply green tint overlay using screen blend
        green_layer = np.array(green_tint, dtype=np.float32)
        F = img_arr / 255.0
        C = (green_layer / 255.0) * intensity
        # Screen blend: 1 - (1-F) * (1-C)
        img_arr = (1.0 - (1.0 - F) * (1.0 - C)) * 255.0

        # 2. Enhance contrast (nightvision has high contrast)
        img_arr = np.clip((img_arr - 128) * 1.25 + 128, 0, 255)

        # 3. Apply vignette (darker edges like goggle view)
        for c in range(3):
            img_arr[:, :, c] = img_arr[:, :, c] * vignette_mask

        # 4. Add animated scan lines
        if animate_scanlines:
            # Slowly scroll scan lines downward
            offset = int((t * 15) % scanline_spacing)  # 15px/sec scroll speed
            scanlines = _create_scanline_pattern(w, h, scanline_spacing, scanline_opacity, offset)
        else:
            scanlines = base_scanline_pattern

        # Apply scanlines (darken where scanlines are)
        img_arr = img_arr * (1.0 - scanlines[:, :, np.newaxis])

        # 5. Add film grain/noise for fuzzy texture
        if noise_amount > 0:
            noise = np.random.normal(0, noise_amount * 20, (h, w, 3))
            img_arr = img_arr + noise

        # Clamp and convert back to uint8
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

        # 6. Apply slight blur for fuzzy CRT effect
        if blur_amount > 0:
            img = Image.fromarray(img_arr)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount))
            img_arr = np.array(img)

        return img_arr

    return clip.transform(apply_effect)


def _create_vignette_mask(w: int, h: int, strength: float) -> np.ndarray:
    """Create a radial vignette mask (1.0 at center, darker at edges)."""
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2

    # Distance from center (normalized)
    dist_x = (x - cx) / (w / 2)
    dist_y = (y - cy) / (h / 2)
    dist = np.sqrt(dist_x**2 + dist_y**2)

    # Smooth falloff
    vignette = 1.0 - (dist * strength)
    vignette = np.clip(vignette, 0.4, 1.0)  # Don't go too dark

    return vignette


def _create_scanline_pattern(
    w: int,
    h: int,
    spacing: int,
    opacity: float,
    offset: int = 0
) -> np.ndarray:
    """Create horizontal scanline pattern (0 = no effect, opacity = dark line)."""
    pattern = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        if (y + offset) % spacing == 0:
            pattern[y, :] = opacity

    return pattern

"""Organic slime splatter overlay effect.

Displays an irregular blob shape resembling paint splatter or slime
at a specified location on the frame.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from moviepy import CompositeVideoClip
from moviepy.video.VideoClip import VideoClip as _VideoClip

from utils.logger import setup_logger
from ..easing import ease_in_out_cubic

logger = setup_logger(__name__)


def generate_organic_slime_shape(width: int, height: int, seed: int = 42) -> Image.Image:
    """Generate an organic slime splatter shape.

    Creates an irregular blob shape resembling paint splatter or slime.

    :param width: Width of the shape in pixels.
    :param height: Height of the shape in pixels.
    :param seed: Random seed for reproducible shapes.
    :return: PIL Image with slime shape (alpha channel).
    """
    np.random.seed(seed)

    # Create base mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Center point
    cx, cy = width // 2, height // 2

    # Draw irregular blob using multiple overlapping circles
    num_blobs = np.random.randint(8, 15)
    for _ in range(num_blobs):
        # Random offset from center
        offset_x = np.random.randint(-width // 3, width // 3)
        offset_y = np.random.randint(-height // 3, height // 3)

        # Random radius
        radius = np.random.randint(min(width, height) // 6, min(width, height) // 3)

        # Draw filled circle
        x = cx + offset_x
        y = cy + offset_y
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=255
        )

    # Add irregular edges with smaller splatter droplets
    num_droplets = np.random.randint(15, 25)
    for _ in range(num_droplets):
        # Random position around the main blob
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(min(width, height) * 0.2, min(width, height) * 0.5)

        x = int(cx + distance * np.cos(angle))
        y = int(cy + distance * np.sin(angle))
        radius = np.random.randint(10, 40)

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=255
        )

    # Apply blur for organic edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=15))

    # Threshold to create sharper but still organic edges
    mask_array = np.array(mask)
    mask_array = np.where(mask_array > 50, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_array)

    # Apply slight blur again for softer edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

    return mask


def apply_slime_splatter(
    clip: _VideoClip,
    position: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    opacity: float = 0.7,
    seed: Optional[int] = None,
    animate_in: float = 0.4,
) -> _VideoClip:
    """Apply organic slime splatter overlay at specified position.

    :param clip: Input video clip.
    :param position: Normalized (x, y, w, h) coordinates for splatter placement.
    :param color: RGB color tuple (default: bright green).
    :param opacity: Splatter opacity 0-1 (default: 0.7).
    :param seed: Random seed for shape generation (default: based on position hash).
    :param animate_in: Fade-in duration in seconds (default: 0.4).
    :return: Video clip with slime splatter overlay.
    """
    w, h = clip.size
    dur = float(clip.duration or 1.0)

    # Convert normalized position to pixels
    bx, by, bw, bh = position
    box_x = int(bx * w)
    box_y = int(by * h)
    box_w = int(bw * w)
    box_h = int(bh * h)

    # Generate slime shape (use position hash as seed if not provided)
    if seed is None:
        seed = hash(position) % 1000
    slime_shape = generate_organic_slime_shape(box_w, box_h, seed=seed)

    # Calculate alpha value
    alpha = int(255 * max(0.0, min(1.0, opacity)))

    def make_overlay_frame(t: float):
        """Generate slime splatter overlay frame."""
        # Fade-in animation
        fade_in_progress = min(1.0, t / animate_in) if animate_in > 0 else 1.0
        fade_in_progress = ease_in_out_cubic(fade_in_progress)
        current_alpha = int(alpha * fade_in_progress)

        # Create transparent base
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # Create slime splatter
        slime_splatter = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        slime_pixels = np.array(slime_splatter)
        mask_array = np.array(slime_shape)

        # Apply color and alpha
        for i in range(3):
            slime_pixels[:, :, i] = color[i]
        slime_pixels[:, :, 3] = (mask_array * current_alpha / 255).astype(np.uint8)

        slime_splatter = Image.fromarray(slime_pixels)
        overlay.paste(slime_splatter, (box_x, box_y), slime_splatter)

        return np.array(overlay)

    # Create overlay clip with proper alpha handling
    def make_frame_rgb(t):
        rgba = make_overlay_frame(t)
        return rgba[:, :, :3]

    def make_mask_frame(t):
        rgba = make_overlay_frame(t)
        return rgba[:, :, 3] / 255.0

    overlay_clip = _VideoClip(make_frame_rgb, duration=dur)
    mask_clip = _VideoClip(make_mask_frame, duration=dur, is_mask=True)
    overlay_clip = overlay_clip.with_mask(mask_clip)

    # Composite
    result = CompositeVideoClip([clip, overlay_clip], size=(w, h))
    return result.with_duration(dur)

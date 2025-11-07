"""Organic slime splatter overlay effect.

Displays an irregular blob shape resembling paint splatter or slime
at a specified location on the frame.

NOTE: Shape generation needs improvement for more organic appearance.
Currently disabled from production use - available for manual testing only.
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
        current_alpha = alpha * fade_in_progress

        # Create RGB overlay (no alpha channel in the numpy array)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Get slime mask as numpy array
        mask_array = np.array(slime_shape)

        # Create colored slime region
        slime_colored = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        for i in range(3):
            slime_colored[:, :, i] = color[i]

        # Apply mask to color (mask is 0-255, normalize to 0-1)
        mask_normalized = mask_array.astype(np.float32) / 255.0
        for i in range(3):
            slime_colored[:, :, i] = (slime_colored[:, :, i] * mask_normalized).astype(np.uint8)

        # Paste slime into overlay at position
        overlay[box_y:box_y+box_h, box_x:box_x+box_w] = slime_colored

        return overlay

    # Create overlay clip
    overlay_clip = _VideoClip(make_overlay_frame, duration=dur)

    # Composite with opacity
    result = CompositeVideoClip([clip, overlay_clip.with_opacity(opacity)], size=(w, h))
    return result.with_duration(dur)

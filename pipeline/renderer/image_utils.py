"""Image processing utilities for video rendering.

Provides functions for:
- Aspect-ratio-preserving resizing and cropping with anchor-aware positioning
- Visual query refinement for historical figures (prefer illustrations over bust statues)
"""

from typing import Optional, Tuple, List, Set
from PIL import Image
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Historical figures that should prefer illustrations/full-body over bust statues
HISTORICAL_FIGURES: Set[str] = {
    # Ancient historians
    "herodotus", "thucydides", "xenophon", "polybius", "livy", "tacitus",
    "plutarch", "josephus", "suetonius", "ammianus marcellinus",

    # Ancient philosophers
    "socrates", "plato", "aristotle", "epicurus", "zeno", "pythagoras",
    "diogenes", "seneca", "marcus aurelius", "epictetus",

    # Ancient leaders/figures
    "alexander the great", "julius caesar", "cleopatra", "nero", "augustus",
    "hannibal", "spartacus", "pericles", "leonidas",

    # Medieval figures
    "charlemagne", "william the conqueror", "richard the lionheart",
    "saladin", "genghis khan", "marco polo", "joan of arc",

    # Renaissance figures
    "leonardo da vinci", "michelangelo", "raphael", "dante", "machiavelli",
    "copernicus", "galileo", "shakespeare", "columbus",

    # Other historical figures
    "confucius", "sun tzu", "ibn battuta", "avicenna", "averroes",
}

# Keywords that indicate a bust/statue query
BUST_INDICATORS: Set[str] = {
    "bust", "statue", "sculpture", "marble", "bronze", "monument",
}


def resize_and_crop(
    img: Image.Image,
    target_w: int,
    target_h: int,
    anchor_point: Optional[Tuple[int, int]] = None,
    anchor_margin: float = 0.1
) -> Tuple[Image.Image, Optional[Tuple[int, int]]]:
    """Resize image to cover target dimensions and crop to exact size.

    Uses uniform scaling to preserve aspect ratio, then crops to target size.
    If anchor_point is provided, adjusts crop to keep the anchor visible.

    :param img: Input PIL Image.
    :param target_w: Target width in pixels.
    :param target_h: Target height in pixels.
    :param anchor_point: Optional (x, y) pixel coordinates to keep visible.
    :param anchor_margin: Margin from edges (0-0.5) when positioning anchor.
    :return: Tuple of (cropped_image, transformed_anchor_point)
             If anchor was provided, returns new anchor coordinates.
             If anchor was None or out of bounds, returns None for anchor.
    """
    img_w, img_h = img.size

    # Calculate uniform scale to COVER target (not fit)
    scale = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize uniformly
    img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Determine crop position
    if anchor_point is not None:
        # Anchor-aware cropping
        ax, ay = anchor_point
        ax_scaled = int(ax * scale)
        ay_scaled = int(ay * scale)

        # Calculate crop position to keep anchor visible
        # Try to keep anchor in the center, but respect margins
        margin_x = int(target_w * anchor_margin)
        margin_y = int(target_h * anchor_margin)

        # Ideal position: center the anchor
        left = ax_scaled - target_w // 2
        top = ay_scaled - target_h // 2

        # Clamp to keep within bounds
        max_left = new_w - target_w
        max_top = new_h - target_h

        left = max(0, min(max_left, left))
        top = max(0, min(max_top, top))

        # Check if anchor is still visible after clamping
        anchor_x_in_crop = ax_scaled - left
        anchor_y_in_crop = ay_scaled - top

        # If anchor is outside the crop or too close to edges, log warning
        if (anchor_x_in_crop < margin_x or anchor_x_in_crop > target_w - margin_x or
            anchor_y_in_crop < margin_y or anchor_y_in_crop > target_h - margin_y):
            logger.warning(
                f"[image_utils] Anchor at ({ax}, {ay}) after scaling ({ax_scaled}, {ay_scaled}) "
                f"is near edge or outside crop region ({target_w}x{target_h}). "
                f"Final position in crop: ({anchor_x_in_crop}, {anchor_y_in_crop})"
            )

        # If anchor ended up outside the crop entirely, return None
        if (anchor_x_in_crop < 0 or anchor_x_in_crop >= target_w or
            anchor_y_in_crop < 0 or anchor_y_in_crop >= target_h):
            logger.warning(
                f"[image_utils] Anchor is completely outside crop region, "
                f"falling back to center crop"
            )
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            new_anchor = None
        else:
            new_anchor = (anchor_x_in_crop, anchor_y_in_crop)
            logger.debug(
                f"[image_utils] Anchor-aware crop: original ({ax}, {ay}) -> "
                f"crop ({anchor_x_in_crop}, {anchor_y_in_crop})"
            )
    else:
        # Center crop (default)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        new_anchor = None

    # Crop to target size
    right = left + target_w
    bottom = top + target_h
    img_cropped = img_scaled.crop((left, top, right, bottom))

    return img_cropped, new_anchor


def resize_mask(
    mask: np.ndarray,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    crop_offset: Tuple[int, int]
) -> np.ndarray:
    """Resize and crop a mask to match a transformed image.

    :param mask: Binary mask (H, W) with values 0/255.
    :param original_size: (width, height) of original image.
    :param target_size: (width, height) of target image.
    :param crop_offset: (left, top) crop offset in scaled coordinates.
    :return: Transformed mask matching target size.
    """
    import cv2

    orig_w, orig_h = original_size
    target_w, target_h = target_size

    # Calculate scale
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize mask
    mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Crop mask
    left, top = crop_offset
    mask_cropped = mask_scaled[top:top+target_h, left:left+target_w]

    return mask_cropped


def refine_visual_query_for_historical_figure(
    visual_query: str,
    person_name: str,
) -> str:
    """Refine visual query for historical figures to prefer illustrations over busts.

    When querying for historical figures, Google Images often returns bust statues
    (head/shoulders only) which are poorly framed for video effects. This function
    detects such queries and adds qualifiers to prefer:
    - Artistic illustrations
    - Full-body depictions
    - Historical recreations

    :param visual_query: Original visual query (e.g., "Herodotus ancient Greek historian portrait")
    :param person_name: Name of historical figure being queried
    :return: Refined visual query with style preferences
    """
    query_lower = visual_query.lower()
    name_lower = person_name.lower()

    # Check if this is a known historical figure
    is_historical = any(fig in name_lower for fig in HISTORICAL_FIGURES)

    if not is_historical:
        # Not a historical figure, return original query
        return visual_query

    # Check if query already has bust indicators (avoid adding if already there)
    has_bust_indicator = any(word in query_lower for word in BUST_INDICATORS)

    # Check if query already has preferred style indicators
    has_illustration_indicator = any(word in query_lower for word in [
        "illustration", "drawing", "painting", "artwork", "depiction",
        "full body", "full-body", "scene", "recreation"
    ])

    if has_illustration_indicator:
        # Query already refined, return as-is
        logger.debug(f"[image_utils] Visual query already has illustration indicators: '{visual_query}'")
        return visual_query

    # Refine the query to prefer illustrations/full-body
    # Strategy: Add qualifiers that push search away from bust statues
    if has_bust_indicator:
        # Query explicitly mentions bust/statue - warn and try to override
        logger.warning(f"[image_utils] Query contains bust indicators: '{visual_query}' - attempting to override")
        refined = visual_query + " -bust -statue illustration full body"
    else:
        # Add positive qualifiers for better image style
        refined = visual_query + " illustration full body painting"

    logger.info(f"[image_utils] Refined visual query for historical figure '{person_name}':")
    logger.info(f"[image_utils]   Original: '{visual_query}'")
    logger.info(f"[image_utils]   Refined:  '{refined}'")

    return refined


def detect_historical_figure_in_query(visual_query: str) -> List[str]:
    """Detect if visual query mentions any known historical figures.

    :param visual_query: Visual query to analyze
    :return: List of detected historical figure names
    """
    query_lower = visual_query.lower()
    detected = []

    for figure in HISTORICAL_FIGURES:
        if figure in query_lower:
            detected.append(figure)

    return detected

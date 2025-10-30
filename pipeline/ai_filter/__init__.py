"""
AI filter subpackage.
Expose CLIP ranker and other visual-semantic utilities.
"""
from .clip_ranker import rank_images

__all__ = ["rank_images"]

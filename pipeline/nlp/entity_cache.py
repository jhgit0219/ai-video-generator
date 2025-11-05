"""LRU cache for entity extraction results.

Caches entity extraction results to avoid redundant spaCy processing
for repeated transcript text.
"""

from typing import Dict, Any, Optional
from collections import OrderedDict
import hashlib
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EntityCache:
    """LRU cache for entity extraction results.

    Uses text hash as key to cache entity extraction results,
    avoiding redundant spaCy NER processing.
    """

    def __init__(self, max_size: int = 100):
        """Initialize entity cache.

        :param max_size: Maximum number of cached results.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        logger.debug(f"[entity_cache] Initialized with max_size={max_size}")

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text.

        :param text: Text to hash.
        :return: SHA256 hash string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached result for text.

        :param text: Text to look up.
        :return: Cached result or None if not found.
        """
        key = self._hash_text(text)

        if key in self._cache:
            self._cache.move_to_end(key)
            logger.debug(f"[entity_cache] Cache hit for text: {text[:50]}...")
            return self._cache[key]

        logger.debug(f"[entity_cache] Cache miss for text: {text[:50]}...")
        return None

    def set(self, text: str, result: Dict[str, Any]) -> None:
        """Store result in cache.

        :param text: Text that was processed.
        :param result: Entity extraction result.
        """
        key = self._hash_text(text)

        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"[entity_cache] Evicted oldest entry (size={len(self._cache)})")

        self._cache[key] = result
        logger.debug(f"[entity_cache] Cached result for text: {text[:50]}...")

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.debug("[entity_cache] Cache cleared")

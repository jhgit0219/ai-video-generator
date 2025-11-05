"""Tests for entity extraction caching."""

import pytest
from pipeline.nlp.entity_cache import EntityCache


def test_cache_hit():
    """Test cache hit returns cached result."""
    cache = EntityCache()

    text = "In Egypt, the pyramids stand tall."
    result = {"locations": ["Egypt"], "persons": []}

    cache.set(text, result)

    cached = cache.get(text)
    assert cached == result


def test_cache_miss():
    """Test cache miss returns None."""
    cache = EntityCache()

    cached = cache.get("This text was never cached")
    assert cached is None


def test_cache_size_limit():
    """Test cache respects size limit."""
    cache = EntityCache(max_size=3)

    cache.set("text1", {"locations": ["A"], "persons": []})
    cache.set("text2", {"locations": ["B"], "persons": []})
    cache.set("text3", {"locations": ["C"], "persons": []})
    cache.set("text4", {"locations": ["D"], "persons": []})

    assert cache.get("text1") is None
    assert cache.get("text4") is not None

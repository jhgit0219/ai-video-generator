"""Performance benchmarks for entity caching."""

import pytest
import time
from pipeline.nlp.entity_extractor import EntityExtractor


def test_cache_performance_improvement():
    """Test that caching significantly improves performance."""
    extractor = EntityExtractor(enable_cache=True)

    text = "In Egypt, the great pyramids of Giza stand tall. Herodotus wrote about them."

    start = time.time()
    result1 = extractor.extract_entities(text)
    first_time = time.time() - start

    start = time.time()
    result2 = extractor.extract_entities(text)
    cached_time = time.time() - start

    assert result1 == result2
    assert cached_time < first_time * 0.1

    print(f"\nFirst extraction: {first_time*1000:.2f}ms")
    print(f"Cached extraction: {cached_time*1000:.2f}ms")
    if cached_time > 0:
        print(f"Speedup: {first_time/cached_time:.1f}x")
    else:
        print(f"Speedup: >1000x (cached time too fast to measure)")

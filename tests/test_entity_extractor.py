"""Tests for NLP entity extraction."""

import pytest
from pipeline.nlp.entity_extractor import EntityExtractor


def test_extract_locations_basic():
    """Test basic location extraction."""
    extractor = EntityExtractor()
    text = "In ancient Egypt, near the pyramids of Giza, a discovery was made."

    entities = extractor.extract_entities(text)

    assert "locations" in entities
    assert len(entities["locations"]) >= 1
    assert any("Egypt" in loc for loc in entities["locations"])


def test_extract_persons_basic():
    """Test basic person name extraction."""
    extractor = EntityExtractor()
    text = "Julius Caesar and Cleopatra ruled Egypt together."

    entities = extractor.extract_entities(text)

    assert "persons" in entities
    assert len(entities["persons"]) >= 1
    assert any("Cleopatra" in person for person in entities["persons"])


def test_first_mention_tracking():
    """Test tracking first mention of entities across segments."""
    extractor = EntityExtractor()

    # First segment mentions Egypt
    entities1 = extractor.extract_entities("In Egypt, pyramids stand tall.", segment_idx=0)
    assert entities1["locations"]
    assert extractor.is_first_mention("Egypt", "location")

    # Second segment mentions Egypt again
    entities2 = extractor.extract_entities("Egypt was a great civilization.", segment_idx=1)
    assert not extractor.is_first_mention("Egypt", "location")

    # Third segment mentions new location
    entities3 = extractor.extract_entities("In Rome, the Colosseum stands.", segment_idx=2)
    assert extractor.is_first_mention("Rome", "location")

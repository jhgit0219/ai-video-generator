"""Tests for content-aware effects director."""

import pytest
from pipeline.renderer.content_aware_effects_director import ContentAwareEffectsDirector
from pipeline.parser import VideoSegment


def test_detect_first_location_mention():
    """Test detection of first location mention in segments."""
    director = ContentAwareEffectsDirector()

    segments = [
        VideoSegment(
            start_time=0.0, end_time=5.0, duration=5.0,
            transcript="Welcome to our story about ancient civilizations."
        ),
        VideoSegment(
            start_time=5.0, end_time=10.0, duration=5.0,
            transcript="In Egypt, the pyramids were built thousands of years ago."
        ),
        VideoSegment(
            start_time=10.0, end_time=15.0, duration=5.0,
            transcript="The Egyptian pharaohs ruled for centuries."
        ),
    ]

    director.analyze_segments(segments)

    # First segment has no locations
    assert not segments[0].is_first_location_mention

    # Second segment has first mention of Egypt
    assert segments[1].is_first_location_mention
    assert segments[1].first_mentioned_location == "Egypt"

    # Third segment mentions Egypt again but not first
    assert not segments[2].is_first_location_mention


def test_modify_visual_query_for_location():
    """Test visual query modification for location segments."""
    director = ContentAwareEffectsDirector()

    segment = VideoSegment(
        start_time=0.0, end_time=5.0, duration=5.0,
        transcript="In Rome, the Colosseum stands as a testament to ancient engineering.",
        visual_query="ancient Roman architecture"
    )
    segment.is_first_location_mention = True
    segment.first_mentioned_location = "Rome"

    modified_query = director.modify_visual_query_for_location(segment)

    assert "3d globe" in modified_query.lower() or "map" in modified_query.lower()
    assert "Rome" in modified_query or "rome" in modified_query


def test_inject_branding_effects():
    """Test injection of branding effect tools into effect plans."""
    director = ContentAwareEffectsDirector()

    # Location segment
    location_segment = VideoSegment(
        start_time=0.0, end_time=5.0, duration=5.0,
        transcript="In Egypt..."
    )
    location_segment.is_first_location_mention = True
    location_segment.first_mentioned_location = "Egypt"

    effect_plan = {"motion": "pan_up", "overlays": [], "tools": []}
    director.inject_branding_effects(location_segment, effect_plan)

    assert any(tool.get("name") == "map_highlight" for tool in effect_plan["tools"])
    map_tool = next(t for t in effect_plan["tools"] if t.get("name") == "map_highlight")
    assert map_tool["params"]["location_name"] == "EGYPT"

    # Character segment
    person_segment = VideoSegment(
        start_time=5.0, end_time=10.0, duration=5.0,
        transcript="Herodotus wrote about..."
    )
    person_segment.is_first_person_mention = True
    person_segment.first_mentioned_person = "Herodotus"

    effect_plan2 = {"motion": "ken_burns_in", "overlays": [], "tools": []}
    director.inject_branding_effects(person_segment, effect_plan2)

    assert any(tool.get("name") == "character_highlight" for tool in effect_plan2["tools"])
    char_tool = next(t for t in effect_plan2["tools"] if t.get("name") == "character_highlight")
    assert char_tool["params"]["character_name"] == "HERODOTUS"

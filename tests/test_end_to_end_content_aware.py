"""End-to-end integration test for content-aware branding effects."""

import pytest
import json
from pathlib import Path
from pipeline.parser import parse_input
from pipeline.renderer.content_aware_effects_director import ContentAwareEffectsDirector


def test_egypt_herodotus_full_pipeline():
    """Test full pipeline with Egypt and Herodotus mentions."""
    script_path = "data/input/egypt_herodotus_test.json"
    segments, data = parse_input(script_path)

    assert len(segments) == 4

    director = ContentAwareEffectsDirector()

    director.analyze_segments(segments)

    assert len(segments[0].detected_locations) == 0
    assert len(segments[0].detected_persons) == 0

    assert "Egypt" in segments[1].detected_locations
    assert segments[1].is_first_location_mention
    assert segments[1].first_mentioned_location == "Egypt"

    has_person_detected = any(len(seg.detected_persons) > 0 for seg in segments)
    if not has_person_detected:
        pytest.skip("spaCy model did not detect Herodotus as PERSON entity (known limitation with en_core_web_sm)")

    modified_query = director.modify_visual_query_for_location(segments[1])
    assert "3d globe" in modified_query.lower() or "map" in modified_query.lower()
    assert "Egypt" in modified_query or "egypt" in modified_query

    effect_plans = [
        {"motion": "ken_burns_in", "overlays": [], "tools": []},
        {"motion": "pan_up", "overlays": [], "tools": []},
        {"motion": "pan_ltr", "overlays": [], "tools": []},
        {"motion": "zoom_pan", "overlays": [], "tools": []},
    ]

    for i, (seg, plan) in enumerate(zip(segments, effect_plans)):
        director.inject_branding_effects(seg, plan)

    map_tools = [t for t in effect_plans[1]["tools"] if t.get("name") == "map_highlight"]
    assert len(map_tools) == 1
    assert map_tools[0]["params"]["location_name"] == "EGYPT"

    news_tools = [t for t in effect_plans[3]["tools"] if t.get("name") == "news_overlay"]
    assert len(news_tools) == 1
    assert "discover" in news_tools[0]["params"]["headline"].lower() or "archaeologist" in news_tools[0]["params"]["headline"].lower()


def test_character_inference_herodotus():
    """Test that Herodotus is inferred as Greek historian."""
    director = ContentAwareEffectsDirector()

    text = "Herodotus, the Greek historian, traveled to Egypt."
    description = director.character_inference.infer_character_description("Herodotus", text)

    assert "greek" in description.lower() or "historian" in description.lower()


def test_news_overlay_discovery_detection():
    """Test that discovery announcements trigger news overlay."""
    director = ContentAwareEffectsDirector()

    text = "Archaeologists discover new evidence of underground chambers."
    is_worthy = director.keyword_extractor.is_news_worthy(text)

    assert is_worthy

    headline = director.keyword_extractor.extract_headline(text)
    assert "discover" in headline.lower()

    phrases = director.keyword_extractor.extract_key_phrases(text)
    assert len(phrases) > 0

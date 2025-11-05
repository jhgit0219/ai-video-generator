"""Tests for character name inference from context."""

import pytest
from pipeline.nlp.character_inference import CharacterInferenceEngine


def test_infer_gender_from_name():
    """Test gender inference from character names."""
    engine = CharacterInferenceEngine()

    assert engine.infer_gender("Cleopatra") == "female"
    assert engine.infer_gender("Julius") == "male"
    assert engine.infer_gender("Alexander") == "male"
    assert engine.infer_gender("Nefertiti") == "female"


def test_infer_character_description_from_context():
    """Test inferring character description from surrounding context."""
    engine = CharacterInferenceEngine()

    # Female character in historical context
    context = "Queen Cleopatra ruled Egypt with wisdom and strength."
    description = engine.infer_character_description("Cleopatra", context)

    assert "queen" in description.lower() or "female" in description.lower()
    assert "egypt" in description.lower() or "egyptian" in description.lower()

    # Male philosopher
    context = "The philosopher Socrates taught in ancient Athens."
    description = engine.infer_character_description("Socrates", context)

    assert "philosopher" in description.lower() or "male" in description.lower()
    assert "athens" in description.lower() or "greek" in description.lower()


def test_enhance_person_with_context():
    """Test enhancing bare person name with context."""
    engine = CharacterInferenceEngine()

    persons = ["Herodotus"]
    context = "Herodotus was a Greek historian who wrote about the Persian Wars."

    enhanced = engine.enhance_persons_with_context(persons, context)

    assert len(enhanced) == 1
    assert "historian" in enhanced[0].lower() or "greek" in enhanced[0].lower()

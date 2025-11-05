"""Tests for keyword extraction and highlight detection."""

import pytest
from pipeline.nlp.keyword_extractor import KeywordExtractor


def test_extract_key_phrases_basic():
    """Test basic key phrase extraction."""
    extractor = KeywordExtractor()

    text = "Underground City? Giza Pyramids Yield New Secrets"
    phrases = extractor.extract_key_phrases(text, max_phrases=3)

    assert len(phrases) >= 2
    assert any("Underground City" in p or "underground city" in p.lower() for p in phrases)
    assert any("New Secrets" in p or "secrets" in p.lower() for p in phrases)


def test_detect_news_worthy_segment():
    """Test detection of news-worthy segments."""
    extractor = KeywordExtractor()

    # News-worthy: discovery announcement
    news_text = "Italian researchers claim discovery of underground chambers beneath the pyramids."
    assert extractor.is_news_worthy(news_text)

    # News-worthy: quote
    quote_text = 'As Herodotus wrote: "Egypt is the gift of the Nile."'
    assert extractor.is_news_worthy(quote_text)

    # Not news-worthy: narrative description
    narrative_text = "The pyramids stood tall against the desert sky."
    assert not extractor.is_news_worthy(narrative_text)


def test_extract_headline_from_text():
    """Test extracting headline from text."""
    extractor = KeywordExtractor()

    text = "Archaeologists discover ancient tomb. The tomb contains treasures from the Old Kingdom."
    headline = extractor.extract_headline(text)

    assert "discover" in headline.lower() or "ancient tomb" in headline.lower()
    assert len(headline) < len(text)  # Headline should be shorter

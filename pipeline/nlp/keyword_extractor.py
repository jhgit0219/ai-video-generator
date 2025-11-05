"""Keyword extraction and highlight detection for news overlay effects.

Detects news-worthy segments and extracts key phrases to highlight in
vintage newspaper-style overlays.
"""

from typing import List
import re
from utils.logger import setup_logger

logger = setup_logger(__name__)


# News-worthy indicators
NEWS_INDICATORS = [
    "discover", "found", "reveal", "claim", "announce", "report",
    "new evidence", "breakthrough", "unprecedented", "first time",
    "researchers", "scientists", "archaeologists", "experts"
]

# Quote indicators
QUOTE_PATTERNS = [
    r'"[^"]+"',  # Double quotes
    r"'[^']+'",  # Single quotes
    r":\s*[\"']",  # Colon followed by quote
]


class KeywordExtractor:
    """Extract keywords and detect news-worthy content for overlay effects."""

    def __init__(self):
        """Initialize keyword extractor."""
        logger.debug("[keyword_extractor] Initialized")

    def extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from text for highlighting.

        Uses simple heuristics: capitalized phrases, noun phrases,
        and emphasized words.

        :param text: Text to analyze.
        :param max_phrases: Maximum number of phrases to extract.
        :return: List of key phrases to highlight.
        """
        phrases = []

        # Extract capitalized phrases (e.g., "Underground City", "New Secrets")
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        cap_matches = re.findall(cap_pattern, text)
        phrases.extend(cap_matches)

        # Extract quoted phrases
        for pattern in QUOTE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                # Remove quotes
                clean = match.strip('"\'').strip()
                if clean:
                    phrases.append(clean)

        # Extract emphasized words (question marks, exclamation points)
        emphasis_pattern = r'(\b\w+\b)[?!]'
        emphasis_matches = re.findall(emphasis_pattern, text)
        phrases.extend(emphasis_matches)

        # Remove duplicates and limit
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower not in seen and len(phrase) > 3:
                unique_phrases.append(phrase)
                seen.add(phrase_lower)
                if len(unique_phrases) >= max_phrases:
                    break

        logger.debug(f"[keyword_extractor] Extracted {len(unique_phrases)} key phrases: {unique_phrases}")
        return unique_phrases

    def is_news_worthy(self, text: str) -> bool:
        """Determine if segment is news-worthy (suitable for news overlay).

        :param text: Segment transcript.
        :return: True if news-worthy, False otherwise.
        """
        text_lower = text.lower()

        # Check for news indicators
        has_news_indicator = any(indicator in text_lower for indicator in NEWS_INDICATORS)

        # Check for quotes
        has_quote = any(re.search(pattern, text) for pattern in QUOTE_PATTERNS)

        # Check for question format (headlines often use questions)
        has_question = "?" in text

        is_worthy = has_news_indicator or has_quote or has_question

        logger.debug(f"[keyword_extractor] News-worthy: {is_worthy} "
                    f"(indicators={has_news_indicator}, quotes={has_quote}, questions={has_question})")

        return is_worthy

    def extract_headline(self, text: str, max_length: int = 60) -> str:
        """Extract or generate headline from text.

        :param text: Full text.
        :param max_length: Maximum headline length.
        :return: Headline string.
        """
        # Use first sentence as headline
        sentences = re.split(r'[.!?]', text)
        if sentences:
            headline = sentences[0].strip()

            # Truncate if too long
            if len(headline) > max_length:
                words = headline.split()
                truncated = []
                length = 0
                for word in words:
                    if length + len(word) + 1 > max_length:
                        break
                    truncated.append(word)
                    length += len(word) + 1
                headline = " ".join(truncated) + "..."

            logger.debug(f"[keyword_extractor] Extracted headline: '{headline}'")
            return headline

        return text[:max_length]

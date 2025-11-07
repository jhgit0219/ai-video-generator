"""Content-Aware Effects Director for automated branding effect application.

Analyzes video segments to detect first mentions of locations and characters,
then automatically applies appropriate branding effects (map_highlight,
character_highlight) without manual configuration.
"""

from typing import List, Dict, Any, Optional
import re
from pipeline.parser import VideoSegment
from pipeline.nlp.entity_extractor import EntityExtractor
from pipeline.nlp.character_inference import CharacterInferenceEngine
from pipeline.nlp.keyword_extractor import KeywordExtractor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ContentAwareEffectsDirector:
    """Automatically applies branding effects based on content analysis.

    Detects locations and character names from transcripts, identifies first
    mentions, and injects appropriate branded effects (map_highlight for
    locations, character_highlight for persons).
    """

    def __init__(self):
        """Initialize content-aware effects director."""
        self.entity_extractor = EntityExtractor()
        self.character_inference = CharacterInferenceEngine()
        self.keyword_extractor = KeywordExtractor()
        logger.info("[content_aware_effects] Initialized with NER, inference, and keyword extraction")

    def analyze_segments(self, segments: List[VideoSegment]) -> None:
        """Analyze segments for entity mentions and mark first occurrences.

        Updates segment objects in-place with entity detection results.

        :param segments: List of video segments to analyze.
        """
        logger.info(f"[content_aware_effects] Analyzing {len(segments)} segments for entities")

        # Reset entity tracking for new script
        self.entity_extractor.reset()

        for idx, segment in enumerate(segments):
            # Extract entities from transcript
            entities = self.entity_extractor.extract_entities(
                segment.transcript,
                segment_idx=idx
            )

            segment.detected_locations = entities["locations"]
            segment.detected_persons = entities["persons"]

            # Enhance person names with context (e.g., "Herodotus" -> "Greek historian Herodotus")
            if segment.detected_persons:
                segment.detected_persons_enhanced = self.character_inference.enhance_persons_with_context(
                    segment.detected_persons,
                    segment.transcript
                )
                logger.debug(f"[content_aware_effects] Segment {idx}: Enhanced persons: "
                            f"{segment.detected_persons} -> {segment.detected_persons_enhanced}")

            # Check for first mentions
            if segment.detected_locations:
                first_location = segment.detected_locations[0]
                if self.entity_extractor.is_first_mention(first_location, "location"):
                    segment.is_first_location_mention = True
                    segment.first_mentioned_location = first_location
                    logger.info(f"[content_aware_effects] Segment {idx}: "
                               f"First mention of location '{first_location}'")

            if segment.detected_persons:
                first_person = segment.detected_persons[0]
                if self.entity_extractor.is_first_mention(first_person, "person"):
                    segment.is_first_person_mention = True
                    segment.first_mentioned_person = first_person
                    logger.info(f"[content_aware_effects] Segment {idx}: "
                               f"First mention of person '{first_person}'")

    def modify_visual_query_for_location(self, segment: VideoSegment) -> str:
        """Modify visual query to scrape 3D globe/map for location segments.

        :param segment: Segment with first location mention.
        :return: Modified visual query targeting globe/map imagery.
        """
        if not segment.is_first_location_mention or not segment.first_mentioned_location:
            return segment.visual_query or ""

        location = segment.first_mentioned_location

        # Generate map-focused query
        modified_query = (
            f"3D globe showing {location} highlighted, "
            f"world map with {location} marked, "
            f"geographic location of {location} on Earth"
        )

        logger.info(f"[content_aware_effects] Modified visual query for location: "
                   f"'{segment.visual_query}' -> '{modified_query}'")

        return modified_query

    def inject_branding_effects(
        self,
        segment: VideoSegment,
        effect_plan: Dict[str, Any]
    ) -> None:
        """Inject branding effect tools into effect plan based on entities.

        Modifies effect_plan in-place by adding map_highlight,
        character_highlight, or news_overlay tools as appropriate.

        :param segment: Segment with entity detection results.
        :param effect_plan: Effect plan dict to modify.
        """
        if "tools" not in effect_plan:
            effect_plan["tools"] = []

        # Add map_highlight for first location mention
        if segment.is_first_location_mention and segment.first_mentioned_location:
            location_upper = segment.first_mentioned_location.upper()

            # Calculate normalized box position (center of frame)
            # TODO: Could use geocoding API to get actual lat/lon for positioning
            box_position = [0.35, 0.25, 0.3, 0.2]  # Center region

            map_tool = {
                "name": "map_highlight",
                "params": {
                    "location_name": location_upper,
                    "box_position": box_position,
                    "cps": 8,
                    "highlight_duration": 2.5,
                    "fade_to_normal_duration": 1.5,
                    "text_position": "center"
                }
            }

            effect_plan["tools"].append(map_tool)
            logger.info(f"[content_aware_effects] Injected map_highlight for {location_upper}")

        # Add character_highlight or newspaper_frame for first person mention
        if segment.is_first_person_mention and segment.first_mentioned_person:
            person_name = segment.first_mentioned_person
            person_upper = person_name.upper()

            # Check if this person warrants newspaper frame via LLM
            is_newspaper_worthy = self.character_inference.is_newspaper_worthy(
                person_name,
                segment.transcript
            )

            if is_newspaper_worthy:
                # Use newspaper_frame for significant/newsworthy persons
                # Extract enhanced description for headline/body text
                enhanced_desc = segment.detected_persons_enhanced[0] if segment.detected_persons_enhanced else person_name

                newspaper_tool = {
                    "name": "newspaper_frame",
                    "params": {
                        "headline": person_upper,
                        "subheadline": enhanced_desc,
                        "body_text": segment.transcript[:200] + "..." if len(segment.transcript) > 200 else segment.transcript,
                        "cutout_width_ratio": 0.6,
                        "cutout_height_ratio": 0.6,
                        "zoom_to_fullscreen": True,
                        "zoom_start_time": 2.0,
                        "zoom_duration": 1.5
                    }
                }
                effect_plan["tools"].append(newspaper_tool)
                logger.info(f"[content_aware_effects] Injected newspaper_frame for newsworthy person {person_upper}")
            else:
                # Use character_highlight for modern/generic persons
                char_tool = {
                    "name": "character_highlight",
                    "params": {
                        "character_name": person_upper,
                        "glow_color": [0, 255, 200],
                        "glow_stages": [1.0, 0.6, 0.3],
                        "stage_durations": [0.5, 1.2, 1.8],
                        "label_position": "top",
                        "cps": 10
                    }
                }
                effect_plan["tools"].append(char_tool)
                logger.info(f"[content_aware_effects] Injected character_highlight for {person_upper}")

        # Add news_overlay for news-worthy segments
        self.detect_and_inject_news_overlay(segment, effect_plan)

    def detect_and_inject_news_overlay(
        self,
        segment: VideoSegment,
        effect_plan: Dict[str, Any]
    ) -> None:
        """Detect news-worthy segments and inject news overlay effect.

        :param segment: Segment to analyze.
        :param effect_plan: Effect plan to modify.
        """
        if not self.keyword_extractor.is_news_worthy(segment.transcript):
            return

        # Extract headline and key phrases
        headline = self.keyword_extractor.extract_headline(segment.transcript)
        key_phrases = self.keyword_extractor.extract_key_phrases(segment.transcript)

        if not headline:
            return

        # Split transcript into sentences for body text
        sentences = re.split(r'[.!?]', segment.transcript)
        body_text = ". ".join(sentences[1:3]) if len(sentences) > 1 else ""

        news_tool = {
            "name": "news_overlay",
            "params": {
                "headline": headline,
                "body_text": body_text,
                "highlighted_phrases": key_phrases,
                "position": [0.05, 0.6],
                "size": [0.9, 0.35],
                "start_time": 0.5,
                "duration": min(segment.duration - 0.5, 4.0),
                "fade_in_duration": 0.5,
                "fade_out_duration": 0.5
            }
        }

        if "tools" not in effect_plan:
            effect_plan["tools"] = []

        effect_plan["tools"].append(news_tool)
        logger.info(f"[content_aware_effects] Injected news_overlay: '{headline}'")

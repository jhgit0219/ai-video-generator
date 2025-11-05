"""Entity extraction using spaCy NER for location and person detection."""

from typing import Dict, List, Set, Optional
import spacy
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EntityExtractor:
    """Extract named entities (locations, persons) from text using spaCy.

    Tracks first mentions of entities across multiple segments to enable
    content-aware effect application (e.g., map highlight on first location mention).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize entity extractor with spaCy model.

        :param model_name: spaCy model to use for NER.
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"[entity_extractor] Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"[entity_extractor] spaCy model '{model_name}' not found. "
                        f"Run: python -m spacy download {model_name}")
            raise

        # Track seen entities for first mention detection
        self._seen_entities: Dict[str, Set[str]] = {
            "location": set(),
            "person": set(),
        }

    def extract_entities(
        self,
        text: str,
        segment_idx: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """Extract named entities from text.

        :param text: Text to analyze.
        :param segment_idx: Optional segment index for logging.
        :return: Dict with 'locations' and 'persons' lists.
        """
        doc = self.nlp(text)

        locations = []
        persons = []

        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                # GPE: Geo-political entity, LOC: Location, FAC: Facility
                location = ent.text.strip()
                if location and location not in locations:
                    locations.append(location)
                    logger.debug(f"[entity_extractor] Found location: {location}")

            elif ent.label_ == "PERSON":
                person = ent.text.strip()
                if person and person not in persons:
                    persons.append(person)
                    logger.debug(f"[entity_extractor] Found person: {person}")

        if segment_idx is not None:
            logger.info(f"[entity_extractor] Segment {segment_idx}: "
                       f"{len(locations)} locations, {len(persons)} persons")

        return {
            "locations": locations,
            "persons": persons,
        }

    def is_first_mention(self, entity: str, entity_type: str) -> bool:
        """Check if this is the first mention of an entity.

        Marks entity as seen if it's the first mention.

        :param entity: Entity name (e.g., "Egypt", "Herodotus").
        :param entity_type: Type of entity ("location" or "person").
        :return: True if first mention, False otherwise.
        """
        entity_normalized = entity.lower().strip()

        if entity_type not in self._seen_entities:
            logger.warning(f"[entity_extractor] Unknown entity type: {entity_type}")
            return False

        if entity_normalized in self._seen_entities[entity_type]:
            return False

        # Mark as seen
        self._seen_entities[entity_type].add(entity_normalized)
        logger.debug(f"[entity_extractor] First mention of {entity_type}: {entity}")
        return True

    def reset(self):
        """Reset first mention tracking (for new video scripts)."""
        self._seen_entities = {
            "location": set(),
            "person": set(),
        }
        logger.debug("[entity_extractor] Reset first mention tracking")

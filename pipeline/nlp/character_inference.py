"""Character inference engine for enriching person entities with context.

Infers gender, occupation, and descriptive attributes from character names
and surrounding text to improve search accuracy for historical figures.
"""

from typing import List, Optional
import re
import asyncio
from utils.logger import setup_logger
from utils.llm import ollama_chat

logger = setup_logger(__name__)


# Common name gender associations (historical figures)
MALE_NAMES = {
    "alexander", "julius", "caesar", "herodotus", "socrates", "plato",
    "aristotle", "napoleon", "leonardo", "michelangelo", "galileo",
    "shakespeare", "homer", "virgil", "augustus", "constantine"
}

FEMALE_NAMES = {
    "cleopatra", "nefertiti", "hatshepsut", "joan", "marie", "catherine",
    "elizabeth", "victoria", "ada", "florence", "amelia"
}

# Historical figures that should use newspaper_frame instead of character_highlight
HISTORICAL_FIGURES = MALE_NAMES | FEMALE_NAMES

# Title/occupation patterns
TITLE_PATTERNS = [
    r"\b(king|queen|emperor|empress|pharaoh|caesar)\s+(\w+)",
    r"\b(\w+)\s+the\s+(great|magnificent|terrible|wise)",
    r"\b(saint|sir|lord|lady|duke|duchess)\s+(\w+)",
]

OCCUPATION_KEYWORDS = {
    "historian": ["historian", "chronicler", "documented"],
    "philosopher": ["philosopher", "thinker", "taught", "wisdom"],
    "scientist": ["scientist", "discovered", "invented", "experiment"],
    "artist": ["painter", "sculptor", "artist", "painted"],
    "writer": ["writer", "author", "poet"],
    "ruler": ["ruled", "reigned", "kingdom", "empire"],
    "military": ["general", "commander", "conquered", "battle"],
}


class CharacterInferenceEngine:
    """Infer character attributes from names and context.

    Enhances bare person names (e.g., "Herodotus") with descriptive
    attributes (e.g., "Greek historian Herodotus") to improve image
    search accuracy.
    """

    def __init__(self):
        """Initialize character inference engine."""
        logger.debug("[character_inference] Initialized")

    def is_newspaper_worthy(self, name: str, context: str) -> bool:
        """Check if person warrants newspaper frame effect via LLM.

        Determines if the person is historically significant, newsworthy,
        or important enough to warrant dramatic newspaper introduction.

        :param name: Character name.
        :param context: Surrounding text (transcript).
        :return: True if newspaper-worthy, False otherwise.
        """
        system_prompt = """You are an expert at determining if a person mentioned in a documentary script warrants a dramatic newspaper-style introduction effect.

Answer with ONLY "YES" or "NO" - no explanation needed.

A person is newspaper-worthy if they are:
- Historical figures (ancient to modern history)
- Famous scientists, inventors, philosophers, artists, writers
- Important political/military leaders
- Significant cultural or religious figures
- Anyone who would be featured in historical newspapers or documentaries

NOT newspaper-worthy:
- Generic modern people
- Unnamed individuals
- Common occupations without historical significance"""

        user_prompt = f"""Person mentioned: {name}

Context from script:
{context[:300]}

Should this person get a dramatic newspaper frame introduction?
Answer: """

        try:
            # Run async LLM call in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                ollama_chat(system_prompt, user_prompt, timeout=15)
            )
            loop.close()

            response_clean = response.strip().upper()
            is_worthy = "YES" in response_clean

            logger.info(f"[character_inference] '{name}' newspaper-worthy: {is_worthy} (LLM: {response_clean[:50]})")
            return is_worthy

        except Exception as e:
            logger.warning(f"[character_inference] LLM check failed for '{name}': {e}, defaulting to False")
            return False

    def infer_gender(self, name: str) -> Optional[str]:
        """Infer gender from character name using historical name database.

        :param name: Character name.
        :return: "male", "female", or None if unknown.
        """
        name_lower = name.lower().strip()

        # Check first name (handle "Julius Caesar" -> "Julius")
        first_name = name_lower.split()[0] if " " in name_lower else name_lower

        if first_name in MALE_NAMES:
            return "male"
        if first_name in FEMALE_NAMES:
            return "female"

        # Default heuristics (not perfect but helps)
        if name_lower.endswith("a") and len(name_lower) > 3:
            return "female"  # Cleopatra, Nefertiti, etc.

        return None

    def infer_character_description(self, name: str, context: str) -> str:
        """Infer character description from surrounding context.

        :param name: Character name.
        :param context: Surrounding text (transcript).
        :return: Enriched description (e.g., "Greek historian Herodotus").
        """
        context_lower = context.lower()
        name_lower = name.lower()

        descriptors = []

        # Extract nationality/location first (most important for context)
        # Map locations to nationalities
        location_to_nationality = {
            "greek": "greek",
            "greece": "greek",
            "athens": "greek",
            "roman": "roman",
            "rome": "roman",
            "egyptian": "egyptian",
            "egypt": "egyptian",
            "persian": "persian",
            "persia": "persian",
            "chinese": "chinese",
            "china": "chinese",
            "indian": "indian",
            "india": "indian",
        }

        for location, nationality in location_to_nationality.items():
            if location in context_lower:
                if nationality not in descriptors:
                    descriptors.append(nationality)
                break

        # Extract title if present
        for pattern in TITLE_PATTERNS:
            match = re.search(pattern, context_lower)
            if match:
                title = match.group(1)
                descriptors.append(title)
                break

        # Extract occupation from keywords (skip if title was found and is a ruler title)
        ruler_titles = ["king", "queen", "emperor", "empress", "pharaoh", "caesar"]
        has_ruler_title = len(descriptors) > 1 and descriptors[-1] in ruler_titles

        if not has_ruler_title:
            for occupation, keywords in OCCUPATION_KEYWORDS.items():
                if any(kw in context_lower for kw in keywords):
                    descriptors.append(occupation)
                    break

        # Combine descriptors with name
        if descriptors:
            description = " ".join(descriptors) + " " + name
            logger.debug(f"[character_inference] Inferred: '{name}' -> '{description}'")
            return description

        # Fallback: add gender if known
        gender = self.infer_gender(name)
        if gender:
            descriptor = "male figure" if gender == "male" else "female figure"
            description = f"{descriptor} {name}"
            logger.debug(f"[character_inference] Inferred gender: '{name}' -> '{description}'")
            return description

        # Last resort: return name as-is
        return name

    def enhance_persons_with_context(
        self,
        persons: List[str],
        context: str
    ) -> List[str]:
        """Enhance list of person names with contextual descriptions.

        :param persons: List of person names extracted via NER.
        :param context: Full text context (transcript).
        :return: List of enhanced person descriptions.
        """
        enhanced = []

        for person in persons:
            description = self.infer_character_description(person, context)
            enhanced.append(description)

        return enhanced

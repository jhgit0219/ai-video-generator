"""Segment Analysis Agents for enriching content understanding via API calls.

Provides specialized agents that make external API calls to gather context
about entities, events, and concepts mentioned in video segments.
"""

from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
from utils.logger import setup_logger
from utils.llm import ollama_chat

logger = setup_logger(__name__)


class WikipediaEnrichmentAgent:
    """Enriches entity information using Wikipedia API.

    Queries Wikipedia for persons, places, and events to gather
    structured context (dates, significance, categories, etc.).
    """

    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

    def __init__(self):
        """Initialize Wikipedia enrichment agent."""
        logger.debug("[wikipedia_agent] Initialized")

    async def enrich_person(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch biographical data for a person from Wikipedia.

        :param name: Person's name.
        :return: Dict with person data or None if not found.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Search for the page
                search_params = {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": name,
                    "srlimit": 1
                }

                async with session.get(self.WIKIPEDIA_API_URL, params=search_params) as resp:
                    if resp.status != 200:
                        logger.warning(f"[wikipedia_agent] Search failed for '{name}': HTTP {resp.status}")
                        return None

                    data = await resp.json()
                    results = data.get("query", {}).get("search", [])

                    if not results:
                        logger.debug(f"[wikipedia_agent] No Wikipedia page found for '{name}'")
                        return None

                    page_title = results[0]["title"]
                    logger.debug(f"[wikipedia_agent] Found Wikipedia page: '{page_title}' for '{name}'")

                # Get page extract and categories
                page_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts|categories|pageprops",
                    "titles": page_title,
                    "exintro": True,
                    "explaintext": True,
                    "cllimit": 10
                }

                async with session.get(self.WIKIPEDIA_API_URL, params=page_params) as resp:
                    if resp.status != 200:
                        logger.warning(f"[wikipedia_agent] Page fetch failed: HTTP {resp.status}")
                        return None

                    data = await resp.json()
                    pages = data.get("query", {}).get("pages", {})

                    if not pages:
                        return None

                    page_data = next(iter(pages.values()))
                    extract = page_data.get("extract", "")
                    categories = [cat["title"].replace("Category:", "") for cat in page_data.get("categories", [])]

                    # Parse extract for key info using LLM
                    person_data = await self._parse_person_data(name, extract, categories)

                    logger.info(f"[wikipedia_agent] Enriched '{name}': {person_data.get('occupation', 'unknown occupation')}, "
                               f"{person_data.get('era', 'unknown era')}, significance={person_data.get('significance_score', 0)}")

                    return person_data

        except asyncio.TimeoutError:
            logger.warning(f"[wikipedia_agent] Timeout fetching data for '{name}'")
            return None
        except Exception as e:
            logger.error(f"[wikipedia_agent] Error enriching '{name}': {e}")
            return None

    async def _parse_person_data(self, name: str, extract: str, categories: List[str]) -> Dict[str, Any]:
        """Parse Wikipedia extract and categories to extract structured person data.

        :param name: Person's name.
        :param extract: Wikipedia intro text.
        :param categories: Wikipedia categories.
        :return: Structured person data.
        """
        # Use LLM to extract structured data from Wikipedia text
        system_prompt = """You are a data extraction expert. Extract key biographical information from Wikipedia text.

Return a JSON object with these fields:
- occupation: person's primary occupation/role (e.g., "historian", "philosopher", "emperor")
- era: time period (e.g., "Ancient Greece", "Renaissance", "20th century")
- birth_year: approximate birth year (number or null)
- death_year: approximate death year (number or null)
- significance_score: importance rating 0-10 (10=extremely significant historical figure)
- is_historical: true if person lived before 1900, false otherwise
- notable_work: their most famous work/achievement (brief)

Return ONLY valid JSON, no explanation."""

        user_prompt = f"""Person: {name}

Wikipedia extract:
{extract[:500]}

Categories: {', '.join(categories[:5])}

Extract biographical data as JSON:"""

        try:
            response = await ollama_chat(system_prompt, user_prompt, timeout=20)

            # Parse JSON response
            import json
            # Try to extract JSON from response (LLM might add explanation)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                person_data = json.loads(json_str)
                person_data["name"] = name
                person_data["wikipedia_extract"] = extract[:200]
                return person_data
            else:
                logger.warning(f"[wikipedia_agent] Failed to parse JSON from LLM response for '{name}'")
                return self._fallback_parse(name, extract, categories)

        except Exception as e:
            logger.warning(f"[wikipedia_agent] LLM parsing failed for '{name}': {e}")
            return self._fallback_parse(name, extract, categories)

    def _fallback_parse(self, name: str, extract: str, categories: List[str]) -> Dict[str, Any]:
        """Fallback parsing using simple heuristics.

        :param name: Person's name.
        :param extract: Wikipedia extract.
        :param categories: Wikipedia categories.
        :return: Basic person data.
        """
        # Simple heuristic: check categories for historical markers
        is_historical = any(marker in " ".join(categories).lower() for marker in
                           ["ancient", "medieval", "renaissance", "century bc", "century ad"])

        return {
            "name": name,
            "occupation": "unknown",
            "era": "unknown",
            "birth_year": None,
            "death_year": None,
            "significance_score": 5 if is_historical else 3,
            "is_historical": is_historical,
            "notable_work": "unknown",
            "wikipedia_extract": extract[:200]
        }


class VisualStyleAgent:
    """Determines appropriate visual style and effects for segment content.

    Analyzes segment transcript to suggest visual treatments, color grading,
    and effect choices that match the content's tone and subject matter.
    """

    def __init__(self):
        """Initialize visual style agent."""
        logger.debug("[visual_style_agent] Initialized")

    async def analyze_segment(self, transcript: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze segment to determine visual style recommendations.

        :param transcript: Segment transcript text.
        :param entities: Detected entities (persons, locations, etc.).
        :return: Visual style recommendations.
        """
        system_prompt = """You are a cinematography expert specializing in documentary visual styles.

Analyze the transcript and suggest appropriate visual treatments.

Return a JSON object with:
- era: historical era if applicable (e.g., "Ancient", "Medieval", "Modern", "Contemporary")
- tone: emotional tone (e.g., "mysterious", "dramatic", "celebratory", "somber", "ominous")
- color_grade: suggested color treatment (e.g., "sepia", "desaturated", "warm", "cool", "high_contrast")
- visual_style: overall style (e.g., "ancient_manuscript", "vintage_newspaper", "modern_sleek", "conspiracy_dark")
- recommended_effects: list of 2-3 effect names that would enhance this segment

Return ONLY valid JSON, no explanation."""

        user_prompt = f"""Transcript:
{transcript}

Detected entities:
- Persons: {', '.join(entities.get('persons', []))}
- Locations: {', '.join(entities.get('locations', []))}

Analyze and suggest visual style as JSON:"""

        try:
            response = await ollama_chat(system_prompt, user_prompt, timeout=20)

            import json
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                style_data = json.loads(json_str)

                logger.info(f"[visual_style_agent] Suggested style: {style_data.get('visual_style', 'unknown')}, "
                           f"tone: {style_data.get('tone', 'unknown')}, "
                           f"grade: {style_data.get('color_grade', 'unknown')}")

                return style_data
            else:
                logger.warning("[visual_style_agent] Failed to parse JSON from LLM response")
                return self._default_style()

        except Exception as e:
            logger.error(f"[visual_style_agent] Analysis failed: {e}")
            return self._default_style()

    def _default_style(self) -> Dict[str, Any]:
        """Return default visual style recommendations.

        :return: Default style dict.
        """
        return {
            "era": "unknown",
            "tone": "neutral",
            "color_grade": "natural",
            "visual_style": "documentary_standard",
            "recommended_effects": []
        }


class SegmentAnalysisOrchestrator:
    """Orchestrates multiple analysis agents to enrich segment understanding.

    Coordinates Wikipedia enrichment, visual style analysis, and other
    agents to build comprehensive segment context.
    """

    def __init__(self):
        """Initialize segment analysis orchestrator."""
        self.wikipedia_agent = WikipediaEnrichmentAgent()
        self.visual_style_agent = VisualStyleAgent()
        logger.info("[segment_analysis] Initialized orchestrator with Wikipedia and VisualStyle agents")

    async def analyze_segment(
        self,
        transcript: str,
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all analysis agents on a segment.

        :param transcript: Segment transcript.
        :param entities: Detected entities (persons, locations).
        :return: Enriched segment analysis.
        """
        logger.debug(f"[segment_analysis] Analyzing segment with {len(entities.get('persons', []))} persons, "
                    f"{len(entities.get('locations', []))} locations")

        # Run agents concurrently
        tasks = []

        # Enrich persons via Wikipedia
        person_enrichments = {}
        for person in entities.get("persons", []):
            task = self.wikipedia_agent.enrich_person(person)
            tasks.append(("person", person, task))

        # Analyze visual style
        tasks.append(("visual_style", None, self.visual_style_agent.analyze_segment(transcript, entities)))

        # Execute all tasks concurrently
        results = {}
        for task_type, key, task in tasks:
            try:
                result = await task
                if task_type == "person":
                    if result:
                        person_enrichments[key] = result
                elif task_type == "visual_style":
                    results["visual_style"] = result
            except Exception as e:
                logger.error(f"[segment_analysis] Task failed ({task_type}, {key}): {e}")

        results["enriched_persons"] = person_enrichments

        logger.info(f"[segment_analysis] Analysis complete: enriched {len(person_enrichments)} persons, "
                   f"visual_style={results.get('visual_style', {}).get('visual_style', 'unknown')}")

        return results

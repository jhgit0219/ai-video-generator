"""
Content Analyzer Agent - Iterative genre detection and subject extraction.

Analyzes video segments to determine genre, extract required subjects,
and provide style guidance with iterative refinement based on critique.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.llm import ollama_chat
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Load content guidelines using modular loader (prevents example contamination)
# IMPORTANT: Do NOT load CONTENT_GUIDELINES.md directly as it contains specific
# examples (Mia, Whiskers) that will bias ALL analyses. Use modular loader instead.
try:
    from utils.guideline_loader import get_cached_guidelines
    CONTENT_GUIDELINES = get_cached_guidelines('content_analyzer')
    logger.info("[analyzer_agent] Loaded modular content guidelines (subject_rules only)")
except Exception as e:
    logger.warning(f"[analyzer_agent] Could not load content guidelines: {e}")
    CONTENT_GUIDELINES = ""


CONTENT_ANALYZER_SYSTEM_PROMPT = """
You are a Content Analyzer for an AI video pipeline. You analyze video segments to determine:
1. Genre/style (documentary, fiction, fantasy, sci-fi, etc.)
2. Required subjects that must appear in images (with detailed attributes: gender, age, appearance)
3. Visual style guidance for image search

Your goal is to preserve story accuracy while enabling successful image searches.

CRITICAL RULES FOR SUBJECT EXTRACTION:

1. **Preserve ALL Character Attributes**:
   - Gender: "young girl" NOT "child", "woman" NOT "person"
   - Age: "young girl", "elderly man", "teenage boy"
   - Species: "cat", "dog" NOT "pet" or "animal"
   - Appearance: "orange tabby cat", "long-haired girl"

2. **Named Character Conversion**:
   - Infer attributes from story context
   - "Mia" in child story → "young girl" (feminine name + child context)
   - "Whiskers" the guardian → "cat" (context: guardian cat)
   - "King Arthur" → "medieval king" or "armored knight"
   - "Optimus Prime" → "transforming robot"

3. **NEVER Use Generic Terms**:
   - WRONG: "child" (loses gender)
   - WRONG: "person" (loses gender + age)
   - WRONG: "pet" (loses species)
   - WRONG: "animal" (loses species)

4. **Gender-Specific Occupations**:
   - If character is female: "girl explorer" NOT "pirate" (implies male)
   - If character is male: "boy adventurer" NOT generic "adventurer"
   - Always prefix occupation with gender/age when known

5. **Priority Order**:
   - Subject preservation (gender, age, species) > Story accuracy > Search popularity
   - Better to have specific low-popularity terms than generic high-popularity terms
"""


CONTENT_ANALYZER_INITIAL_PROMPT = """
Analyze this SPECIFIC video segment ONLY and provide structured metadata.

**CRITICAL**: Extract subjects ONLY from this segment's transcript. Do NOT infer subjects from other segments or story context.

TRANSCRIPT (ANALYZE THIS ONLY):
{transcript}

TOPIC: {topic}
CONTENT TYPE: {content_type}
VISUAL QUERY: {visual_query}
VISUAL DESCRIPTION: {visual_description}

Analyze and return JSON with:

1. GENRE: What type of content is this? **Use the VISUAL_QUERY and VISUAL_DESCRIPTION to determine genre.**
   - "documentary_historical" - real historical events/figures, archival footage, PHOTOREALISTIC
   - "documentary_modern" - real modern events/science/nature, PHOTOREALISTIC
   - "fiction_realistic" - fictional but realistic (drama, thriller, no magic), PHOTOREALISTIC
   - "fiction_fantasy" - fantasy/magical elements, mythical creatures, OR poetic/storybook language
   - "fiction_scifi" - science fiction, futuristic technology
   - "animated_stylized" - clearly illustrated/animated/artistic style

   **CRITICAL**: If visual_query includes "fantasy", "illustration", "storybook", "digital painting", "mystical"
   → MUST be "fiction_fantasy" or "animated_stylized", NEVER documentary!

2. REQUIRED_SUBJECTS: Array of strings for subjects MENTIONED IN THIS SEGMENT'S TRANSCRIPT (max 4)

   **CRITICAL RULE**: ONLY include subjects explicitly mentioned or clearly implied in THIS segment's transcript.
   - If the transcript doesn't mention a character, DON'T include them.
   - If the transcript only mentions "she" or "he", infer from visual_description or leave out.

   Characters: Convert fictional names to descriptive types with gender/age/appearance
     - Preserve gender, age, and distinguishing features!
     - Examples: Feminine name in child story -> "young girl" (NOT "child" or "person")
                 Animal companion -> "cat" or "dog" with attributes (NOT "pet" or "animal")
                 Hero character -> "transforming robot leader" (NOT "machine")
                 Professional -> "elderly male scientist" or "female doctor in lab coat"

   Objects: Use descriptors from transcript
     - "ancient treasure map" (if transcript says "ancient map" or "treasure map")
     - "glowing artifact" (if transcript mentions glowing object)
     - "medieval sword" (if transcript references sword)

   Locations: Only if they're primary visual subject
     - "mystical forest", "futuristic city", "stone pyramid"

   MUST be an array of STRINGS, not objects:
   - CORRECT: ["young boy", "ancient book", "castle"]
   - WRONG: [{{"subject": "boy", "attributes": ["young"]}}]

   FORBIDDEN: Generic terms that lose critical attributes
   - "person" (loses gender/age)
   - "child" (loses gender)
   - "pet" or "animal" (loses species)
   - "document" instead of "map"

3. STYLE_GUIDANCE: Single phrase describing how images should look
   - For documentary: "photorealistic documentary footage"
   - For historical: "historical archival photography"
   - For realistic fiction: "cinematic film still"
   - For fantasy: "illustrated storybook art" or "fantasy concept art"
   - For sci-fi: "sci-fi concept art" or "futuristic photography"

4. VISUAL_STYLE_QUALIFIERS: Keywords to add to search query (2-3 max)
   - Examples: ["illustrated", "storybook art"] for fantasy
              ["archival photo", "black and white"] for historical
              ["concept art", "futuristic"] for sci-fi

5. REASONING: Explain your genre classification and subject choices based ONLY on this segment (2-3 sentences)
   - Reference ONLY what is mentioned in THIS segment's transcript
   - Do NOT mention characters/objects from other segments

Return ONLY valid JSON (no markdown, no explanation):
{{
  "genre": "documentary_historical",
  "required_subjects": ["ancient ruins", "stone columns"],
  "style_guidance": "historical photography",
  "visual_style_qualifiers": ["archival", "sepia"],
  "reasoning": "The transcript describes ancient ruins and architecture, suggesting historical documentary content. Focus on archaeological imagery with stone structures."
}}
"""


CONTENT_ANALYZER_CRITIQUE_PROMPT = """
Critique this content analysis for a video segment.

SEGMENT:
- Transcript: {transcript}
- Topic: {topic}
- Visual Query: {visual_query}

ANALYSIS:
{analysis}

SCRAPING RESULTS (if available):
{scraping_results}

Evaluate the analysis on these criteria:

1. GENRE ACCURACY (0-3): Is the genre classification correct?
   - Does it match the content style and subject matter?
   - Will it lead to appropriate image search results?

2. SUBJECT COMPLETENESS (0-3): Are all critical subjects identified?
   - Are key characters/objects/locations captured?
   - Are subjects properly converted (names -> types)?
   - Are there too many or too few subjects?

3. STYLE APPROPRIATENESS (0-2): Is the style guidance appropriate?
   - Will it help find relevant images?
   - Does it match the genre?

4. SEARCHABILITY (0-2): Will this lead to successful image searches?
   - Are subjects concrete and photographable?
   - Are style qualifiers helpful or confusing?

IMPORTANT: Return ONLY valid JSON, no markdown headers, no explanation, no text before or after.

{{
  "score": 8,
  "genre_accuracy": 3,
  "subject_completeness": 3,
  "style_appropriateness": 2,
  "searchability": 2,
  "strengths": "what works well",
  "weaknesses": "what needs improvement",
  "suggestions": "specific improvements for next iteration",
  "satisfied": true
}}
"""


CONTENT_ANALYZER_REFINE_PROMPT = """
Refine the content analysis based on critique feedback.

SEGMENT:
- Transcript: {transcript}
- Topic: {topic}
- Content Type: {content_type}
- Visual Query: {visual_query}

PREVIOUS ANALYSIS:
{previous_analysis}

CRITIQUE FEEDBACK:
- Score: {score}/10
- Weaknesses: {weaknesses}
- Suggestions: {suggestions}

ITERATION HISTORY:
{history}

Generate an IMPROVED analysis addressing the critique. Focus on:
1. Fixing identified weaknesses
2. Improving searchability
3. Ensuring all critical subjects are captured
4. Refining genre classification if needed

Return ONLY valid JSON in the same format:
{{
  "genre": "...",
  "required_subjects": [...],
  "style_guidance": "...",
  "visual_style_qualifiers": [...],
  "reasoning": "..."
}}
"""


class ContentAnalyzerAgent:
    """
    Iterative content analyzer with critique-based refinement.

    Analyzes video segments to extract genre, required subjects, and style guidance.
    Uses iterative refinement with critique loop to improve analysis quality.
    """

    def __init__(self, model: str = "llama3", max_iterations: int = 3):
        """
        Initialize ContentAnalyzer agent.

        Args:
            model: LLM model to use for analysis
            max_iterations: Maximum refinement iterations
        """
        self.model = model
        self.max_iterations = max_iterations
        self.history = []

    async def analyze_segment(
        self,
        segment,
        script_data: Dict[str, Any],
        scraping_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze segment with iterative refinement.

        Args:
            segment: VideoSegment object
            script_data: Full script metadata
            scraping_results: Optional scraping results for feedback

        Returns:
            Analysis dict with genre, subjects, style guidance
        """
        logger.info(f"[analyzer_agent] Starting content analysis for segment")
        logger.info(f"[analyzer_agent] Transcript: {segment.transcript[:80]}...")

        # CRITICAL: Reset history for each segment to prevent cross-contamination
        # Without this, segment N inherits context from segments 0..N-1, causing
        # wrong genre/subjects to be applied (e.g., Mia story data in lost_labyrinth)
        segment_history = []

        best_analysis = None
        best_score = 0.0

        for iteration in range(self.max_iterations):
            logger.info(f"[analyzer_agent] --- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Generate analysis
            if iteration == 0:
                analysis = await self._generate_initial_analysis(segment, script_data)
            else:
                analysis = await self._generate_refinement(
                    segment,
                    script_data,
                    segment_history
                )

            if not analysis:
                logger.error("[analyzer_agent] Failed to generate analysis")
                continue

            logger.info(f"[analyzer_agent] Generated analysis:")
            logger.info(f"  Genre: {analysis.get('genre')}")
            logger.info(f"  Subjects: {analysis.get('required_subjects')}")
            logger.info(f"  Style: {analysis.get('style_guidance')}")

            # Validate structure
            if not self._validate_analysis(analysis):
                logger.error("[analyzer_agent] Invalid analysis structure")
                continue

            # Critique
            critique = await self._critique_analysis(
                analysis,
                segment,
                scraping_results
            )

            if not critique:
                logger.error("[analyzer_agent] Failed to generate critique")
                continue

            logger.info(f"[analyzer_agent] Critique:")
            logger.info(f"  Score: {critique.get('score', 0)}/10")
            logger.info(f"  Strengths: {critique.get('strengths', 'N/A')}")
            logger.info(f"  Weaknesses: {critique.get('weaknesses', 'N/A')}")

            segment_history.append({
                "iteration": iteration + 1,
                "analysis": analysis,
                "critique": critique
            })

            score = critique.get('score', 0)
            if score > best_score:
                best_score = score
                best_analysis = analysis

            # Break if satisfied
            if critique.get('satisfied', False) and score >= 8:
                logger.info(f"[analyzer_agent] Satisfied with analysis after {iteration + 1} iteration(s)")
                break

        if best_analysis:
            logger.info(f"[analyzer_agent] Final best score: {best_score}/10")
            logger.info(f"[analyzer_agent] Final analysis: {json.dumps(best_analysis, indent=2)}")
        else:
            logger.warning("[analyzer_agent] No valid analysis generated, using fallback")
            best_analysis = self._fallback_analysis(segment)

        return best_analysis

    def _format_prompt(self, segment, script_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Format prompt with safe field access and default values.

        Safely formats the CONTENT_ANALYZER_INITIAL_PROMPT template by:
        1. Using getattr() with defaults for optional segment fields
        2. Using dict.get() with defaults for optional script_data fields
        3. Preventing KeyError when fields are missing

        Args:
            segment: VideoSegment with required (transcript, visual_query, topic)
                     and optional fields (visual_description, content_type)
            script_data: Optional script metadata dict

        Returns:
            Formatted prompt string with no unreplaced placeholders

        Raises:
            ValueError: If prompt template has invalid placeholders
        """
        if script_data is None:
            script_data = {}

        # Safe field access with defaults
        data = {
            'transcript': getattr(segment, 'transcript', ''),
            'topic': getattr(segment, 'topic', 'N/A'),
            'content_type': getattr(segment, 'content_type', 'N/A'),
            'generation_method': script_data.get('generation_method', 'N/A'),
            'visual_query': getattr(segment, 'visual_query', ''),
            'visual_description': getattr(segment, 'visual_description', 'N/A')
        }

        # Format with safe dict access
        try:
            return CONTENT_ANALYZER_INITIAL_PROMPT.format(**data)
        except KeyError as e:
            logger.error(f"[analyzer_agent] Prompt template missing field: {e}")
            raise ValueError(f"Invalid prompt template: {e}")

    async def _generate_initial_analysis(
        self,
        segment,
        script_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate initial content analysis."""
        try:
            prompt = self._format_prompt(segment, script_data)

            logger.debug(f"[analyzer_agent] System prompt length: {len(CONTENT_ANALYZER_SYSTEM_PROMPT)} chars")
            logger.debug(f"[analyzer_agent] User prompt length: {len(prompt)} chars")

            response = await ollama_chat(
                CONTENT_ANALYZER_SYSTEM_PROMPT,
                prompt,
                model=self.model
            )

            logger.debug(f"[analyzer_agent] LLM response length: {len(response)} chars")
            logger.debug(f"[analyzer_agent] LLM response preview: {response[:200]}...")

            # Extract JSON from response
            analysis = self._extract_json(response)
            return analysis

        except ValueError as e:
            logger.error(f"[analyzer_agent] Prompt formatting failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[analyzer_agent] Error generating initial analysis: {e}")
            import traceback
            logger.error(f"[analyzer_agent] Traceback: {traceback.format_exc()}")
            return None

    async def _generate_refinement(
        self,
        segment,
        script_data: Dict[str, Any],
        history: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Generate refined analysis based on critique."""
        try:
            previous = history[-1]
            previous_analysis = previous['analysis']
            critique = previous['critique']

            # Format history for context
            history_str = "\n".join([
                f"Iteration {h['iteration']}: Score {h['critique'].get('score', 0)}/10"
                for h in history
            ])

            prompt = CONTENT_ANALYZER_REFINE_PROMPT.format(
                transcript=segment.transcript,
                topic=getattr(segment, 'topic', 'N/A'),
                content_type=getattr(segment, 'content_type', 'N/A'),
                visual_query=segment.visual_query,
                previous_analysis=json.dumps(previous_analysis, indent=2),
                score=critique.get('score', 0),
                weaknesses=critique.get('weaknesses', 'N/A'),
                suggestions=critique.get('suggestions', 'N/A'),
                history=history_str
            )

            response = await ollama_chat(
                CONTENT_ANALYZER_SYSTEM_PROMPT,
                prompt,
                model=self.model
            )

            analysis = self._extract_json(response)
            return analysis

        except Exception as e:
            logger.error(f"[analyzer_agent] Error generating refinement: {e}")
            return None

    async def _critique_analysis(
        self,
        analysis: Dict[str, Any],
        segment,
        scraping_results: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Generate critique of analysis."""
        try:
            scraping_str = "Not yet available" if not scraping_results else json.dumps(scraping_results, indent=2)

            prompt = CONTENT_ANALYZER_CRITIQUE_PROMPT.format(
                transcript=segment.transcript,
                topic=getattr(segment, 'topic', 'N/A'),
                visual_query=segment.visual_query,
                analysis=json.dumps(analysis, indent=2),
                scraping_results=scraping_str
            )

            response = await ollama_chat(
                CONTENT_ANALYZER_SYSTEM_PROMPT,
                prompt,
                model=self.model
            )

            critique = self._extract_json(response)
            return critique

        except Exception as e:
            logger.error(f"[analyzer_agent] Error generating critique: {e}")
            return None

    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate analysis structure."""
        required_keys = [
            'genre',
            'required_subjects',
            'style_guidance',
            'visual_style_qualifiers',
            'reasoning'
        ]

        if not all(k in analysis for k in required_keys):
            logger.error(f"[analyzer_agent] Missing keys: {set(required_keys) - set(analysis.keys())}")
            return False

        if not isinstance(analysis['required_subjects'], list):
            logger.error("[analyzer_agent] required_subjects must be a list")
            return False

        if not isinstance(analysis['visual_style_qualifiers'], list):
            logger.error("[analyzer_agent] visual_style_qualifiers must be a list")
            return False

        return True

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with robust parsing."""
        try:
            if not response or not response.strip():
                logger.error("[analyzer_agent] Empty response from LLM")
                return None

            # Log first 500 chars for debugging
            logger.debug(f"[analyzer_agent] Raw LLM response (first 500 chars): {response[:500]}")

            # Try multiple extraction strategies
            original_response = response

            # Strategy 1: Remove markdown code blocks if present
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()

            # Strategy 2: Find first { and last }
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                response = response[start:end]

            # Strategy 3: Try to parse
            parsed = json.loads(response)
            logger.debug(f"[analyzer_agent] Successfully parsed JSON: {list(parsed.keys())}")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"[analyzer_agent] JSON decode error: {e}")
            logger.error(f"[analyzer_agent] Failed to parse response (first 500 chars):")
            logger.error(f"  {response[:500]}")
            logger.error(f"[analyzer_agent] Full original response length: {len(original_response)} chars")

            # Try to show where parsing failed
            if hasattr(e, 'pos'):
                context_start = max(0, e.pos - 50)
                context_end = min(len(response), e.pos + 50)
                logger.error(f"[analyzer_agent] Error at position {e.pos}:")
                logger.error(f"  ...{response[context_start:context_end]}...")

            return None
        except Exception as e:
            logger.error(f"[analyzer_agent] Unexpected error extracting JSON: {e}")
            logger.error(f"[analyzer_agent] Response type: {type(response)}, length: {len(response) if response else 0}")
            return None

    def _fallback_analysis(self, segment) -> Dict[str, Any]:
        """Generate fallback analysis if iterations fail."""
        logger.warning("[analyzer_agent] Using fallback analysis")
        return {
            "genre": "documentary_modern",
            "required_subjects": [],
            "style_guidance": "photorealistic documentary footage",
            "visual_style_qualifiers": ["cinematic", "documentary"],
            "reasoning": "Fallback analysis due to iteration failure"
        }

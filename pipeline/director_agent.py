"""
Director Agent - Supervisory LLM controller for adaptive video generation.
Evaluates ranked results, refines queries, and triggers re-scraping if quality is low.
Offline edition powered by Ollama (local models such as phi3, llama3, mistral).
"""

import asyncio
import subprocess
import json
from utils.logger import setup_logger
from pipeline.scraper import collect_images_for_segments
from pipeline.ai_filter import rank_images
from pipeline.ai_filter.clip_ranker import _generate_contextual_labels
from pipeline.analyzer_agent import ContentAnalyzerAgent
from config import DIRECTOR_MAX_RETRIES
from pipeline.prompts import (
    DIRECTOR_SYSTEM_PROMPT,
    DIRECTOR_INITIAL_QUERY_TEMPLATE,
    get_director_refine_query_prompt,
)

logger = setup_logger(__name__)

# -------------- Configurable thresholds --------------
MIN_CONFIDENCE = 0.28      # re-trigger if final_score below this (lowered from 0.35 to reduce false retries)
MAX_RETRIES = int(DIRECTOR_MAX_RETRIES)            # max refinement cycles per segment
OLLAMA_MODEL = "llama3"      # change to "llama3", "mistral", etc.
# ------------------------------------------------------


def filter_subjects_for_segment(segment, global_subjects: list) -> list:
    """
    Filter required_subjects to only include those relevant to THIS specific segment.

    Prevents query contamination where global subjects (e.g., "young girl", "cat", "map")
    get injected into segments where they don't appear (e.g., landscape-only segment).

    Args:
        segment: Video segment with transcript, visual_query, visual_description
        global_subjects: List of subjects from content analysis

    Returns:
        Filtered list of subjects that are actually relevant to this segment
    """
    if not global_subjects:
        return []

    # Build segment context from all text fields
    segment_text = " ".join([
        getattr(segment, 'transcript', ''),
        getattr(segment, 'visual_description', ''),
        getattr(segment, 'visual_query', ''),
        getattr(segment, 'topic', '')
    ]).lower()

    if not segment_text.strip():
        # No text context - return all subjects as fallback
        return global_subjects

    # Filter subjects based on keyword matching
    relevant = []
    for subject in global_subjects:
        subject_lower = subject.lower()

        # Extract meaningful keywords from subject (ignore articles/prepositions)
        skip_words = {'with', 'the', 'a', 'an', 'of', 'and', 'or', 'in', 'on', 'at'}
        subject_keywords = [
            word for word in subject_lower.split()
            if len(word) > 3 and word not in skip_words
        ]

        # Check if ANY subject keyword appears in segment text
        for keyword in subject_keywords:
            if keyword in segment_text:
                relevant.append(subject)
                logger.debug(f"[subject_filter] Kept '{subject}' (matched keyword: '{keyword}')")
                break

    # If no subjects matched, log warning but don't force-inject them
    if not relevant and global_subjects:
        logger.info(f"[subject_filter] No subjects relevant to segment: '{segment_text[:80]}...'")
        logger.info(f"[subject_filter] Global subjects were: {global_subjects}")
        logger.info(f"[subject_filter] This is OK - segment may be landscape/setting-focused")

    return relevant


def detect_art_style(segment) -> str:
    """
    Detect if segment needs stylized/illustrated imagery vs photorealistic.

    Analyzes segment text to determine visual rendering style that should be used.
    This helps prevent generic photorealistic results for fantasy/mystical content.

    Args:
        segment: Video segment with genre, style_guidance, visual_query, transcript

    Returns:
        Art style category: "illustration", "cinematic", "documentary", or "photorealistic"
    """
    # Priority 1: Use style_guidance from content analysis if available
    style_guidance = getattr(segment, 'style_guidance', '').lower()
    if 'illustration' in style_guidance or 'storybook' in style_guidance:
        return "illustration"
    if 'painting' in style_guidance or 'art' in style_guidance:
        return "illustration"

    # Priority 2: Check genre
    genre = getattr(segment, 'genre', '').lower()
    if 'fiction_fantasy' in genre or 'fantasy' in genre:
        return "illustration"
    if 'animation' in genre or 'cartoon' in genre:
        return "illustration"

    # Priority 3: Analyze text for style indicators
    indicators = {
        "illustration": ["fantasy", "mystical", "magical", "storybook", "ethereal", "guardian", "destiny", "enchanted", "fairy"],
        "cinematic": ["dramatic", "cinematic", "movie", "film", "noir", "thriller"],
        "documentary": ["realistic", "documentary", "historical", "archival", "news"],
    }

    # Build searchable text from segment
    text_fields = " ".join([
        getattr(segment, 'visual_query', ''),
        getattr(segment, 'visual_description', ''),
        getattr(segment, 'transcript', ''),
        style_guidance
    ]).lower()

    # Check indicators in priority order
    for style, keywords in indicators.items():
        if any(kw in text_fields for kw in keywords):
            logger.debug(f"[art_style] Detected '{style}' style from keywords")
            return style

    # Default to photorealistic
    return "photorealistic"


def inject_art_style_modifiers(query: str, art_style: str) -> str:
    """
    Add art style modifiers to search query to get appropriate imagery.

    Args:
        query: Original search query
        art_style: Art style category from detect_art_style()

    Returns:
        Query with art style modifiers appended
    """
    # Don't inject if style modifiers already present
    query_lower = query.lower()

    if art_style == "illustration":
        # Check if already has illustration-style modifiers
        existing_modifiers = ['illustration', 'painting', 'art', 'storybook', 'drawing', 'digital art']
        if any(mod in query_lower for mod in existing_modifiers):
            logger.debug(f"[art_style] Query already has illustration modifiers")
            return query

        # Inject illustration modifiers
        modifiers = "fantasy illustration storybook art digital painting"
        logger.info(f"[art_style] Injecting illustration modifiers: '{modifiers}'")
        return f"{query} {modifiers}"

    elif art_style == "cinematic":
        existing_modifiers = ['cinematic', 'movie', 'film', 'photography']
        if any(mod in query_lower for mod in existing_modifiers):
            return query

        modifiers = "cinematic photography movie still professional"
        logger.info(f"[art_style] Injecting cinematic modifiers: '{modifiers}'")
        return f"{query} {modifiers}"

    # For documentary/photorealistic, don't inject - let query be natural
    return query


def validate_llm_query_response(response: str) -> bool:
    """
    Validate that LLM returned an actual query, not a preamble.

    Bug fix: LLM sometimes returns "Here is the transformed visual query:"
    instead of actual content, causing wrong images to be scraped.

    Args:
        response: LLM's response text

    Returns:
        True if valid query, False if incomplete/invalid
    """
    if not response or not response.strip():
        return False

    # Reject common preamble phrases
    preambles = [
        "here is the",
        "the transformed query",
        "here's the",
        "visual search query:",
        "the refined search"
    ]

    lower = response.strip().lower()
    if any(preamble in lower for preamble in preambles):
        return False

    # Require minimum meaningful content (at least 3 words of 3+ chars)
    words = [w for w in response.split() if len(w) >= 3]
    if len(words) < 3:
        return False

    return True


async def llm_initial_query(segment, script_context: str = ""):
    """
    Transforms the initial visual query into a cinematic, documentary-style search phrase.
    Called BEFORE scraping to ensure all queries are cinematically refined from the start.

    IMPORTANT: Passes visual_description to help LLM distinguish between CONTEXT words
    (e.g., "archaeology" in a documentary) and VISUAL INTENT (what the image should show).
    """
    user_prompt = DIRECTOR_INITIAL_QUERY_TEMPLATE.format(
        visual_query=getattr(segment, 'visual_query', ''),
        visual_description=getattr(segment, 'visual_description', 'N/A'),
        topic=getattr(segment, 'topic', 'N/A'),
        content_type=getattr(segment, 'content_type', 'N/A'),
        transcript=getattr(segment, 'transcript', '')[:300]
    )

    prompt = f"{DIRECTOR_SYSTEM_PROMPT}\n\n{user_prompt}"

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60
        )
        refined = result.stdout.strip().split("\n")[0].strip('" ')
        if not refined:
            logger.warning("[director_agent] Empty initial refinement result; using original query.")
            return segment.visual_query
        if not validate_llm_query_response(refined):
            logger.warning(f"[director_agent] LLM returned invalid query: '{refined[:80]}...'; using original query.")
            return segment.visual_query

        # CRITICAL FIX: Detect and inject art style modifiers for fantasy/illustration content
        art_style = detect_art_style(segment)
        refined = inject_art_style_modifiers(refined, art_style)

        logger.info(f"[director_agent] Initial query refined: '{segment.visual_query}' -> '{refined}'")
        return refined
    except subprocess.TimeoutExpired:
        logger.warning("[director_agent] Ollama timed out on initial query.")
    except Exception as e:
        logger.warning(f"[director_agent] LLM initial query failed: {e}")
    return segment.visual_query


def enrich_query_with_agent_data(segment, base_query: str) -> str:
    """Enrich visual query with Wikipedia/agent analysis data.

    Uses segment analysis agents data to add era, occupation, and style context
    to improve image search accuracy.

    :param segment: Video segment with agent_analysis attribute.
    :param base_query: Original visual query.
    :return: Enriched query with Wikipedia context.
    """
    if not hasattr(segment, 'agent_analysis') or not segment.agent_analysis:
        return base_query

    enrichments = []

    # Add person era/occupation from Wikipedia
    enriched_persons = segment.agent_analysis.get('enriched_persons', {})
    for person_name, person_data in enriched_persons.items():
        era = person_data.get('era', '')
        occupation = person_data.get('occupation', '')

        if era and era != 'unknown':
            enrichments.append(era.lower())
        if occupation and occupation != 'unknown':
            enrichments.append(occupation.lower())

    # Add visual style suggestions
    visual_style = segment.agent_analysis.get('visual_style', {})
    style_type = visual_style.get('visual_style', '')
    if style_type and 'manuscript' in style_type:
        enrichments.append('historical manuscript')
    elif style_type and 'newspaper' in style_type:
        enrichments.append('vintage newspaper')

    if enrichments:
        enriched_query = f"{base_query} {' '.join(enrichments[:3])}"  # Limit to 3 enrichments
        logger.info(f"[director_agent] Enriched query with agent data: '{base_query}' → '{enriched_query}'")
        return enriched_query

    return base_query


async def llm_refine_query(segment, top3, contextual_labels=None):
    """
    Suggests a better visual search query using the local LLM (Ollama).
    Activated when no images pass the CLIP text relevance filter.
    Refines the CURRENT query (which may already be refined from previous attempts).
    Uses the base query as context to prevent drift from original intent.

    Args:
        segment: The video segment to refine query for
        top3: Top 3 ranked images with their CLIP captions
        contextual_labels: Optional dict of scene/composition/quality labels for context
    """
    current_query = getattr(segment, 'visual_query', '')
    base_query = getattr(segment, 'base_visual_query', current_query)  # Fallback to current if no base stored

    # ENHANCEMENT: Enrich query with segment analysis agent data (Wikipedia, visual style)
    enriched_base = enrich_query_with_agent_data(segment, base_query)
    
    # Extract CLIP captions from top3 images
    top3_captions = []
    if top3:
        for img in top3[:3]:
            caption = img.get('clip_caption')
            score = img.get('final_score', 0)
            if caption:
                top3_captions.append(f"- {caption} (score: {score:.2f})")
    
    # Build scores text for template
    scores_text = "\n".join(top3_captions) if top3_captions else "No images scored well enough"
    
    # Add contextual labels if available
    label_context = ""
    if contextual_labels:
        scene_examples = ", ".join(contextual_labels.get('scene_labels', [])[:5])
        label_context = f"\n\nAvailable image types in this video's collection:\n{scene_examples}\nYour refined query should target one of these scene types."
    
    # Get content analysis metadata if available
    genre = getattr(segment, 'genre', 'documentary_modern')
    style_guidance = getattr(segment, 'style_guidance', 'photorealistic')
    global_subjects = getattr(segment, 'required_subjects', [])
    visual_style_qualifiers = getattr(segment, 'visual_style_qualifiers', [])

    # CRITICAL FIX: Filter subjects to only those relevant to THIS segment
    # This prevents query contamination where global subjects from content analysis
    # (e.g., "young girl", "cat", "map") get forced into landscape-only segments
    required_subjects = filter_subjects_for_segment(segment, global_subjects)

    # Log filtering results
    if global_subjects != required_subjects:
        logger.info(f"[subject_filter] Filtered subjects for segment:")
        logger.info(f"  Global: {global_subjects}")
        logger.info(f"  Filtered: {required_subjects}")
        logger.info(f"  Removed {len(global_subjects) - len(required_subjects)} irrelevant subject(s)")

    # Format required subjects for prompt generation
    subjects_str = ", ".join(required_subjects) if required_subjects else "None specified"
    qualifiers_str = ", ".join(visual_style_qualifiers) if visual_style_qualifiers else "cinematic"

    # IMPORTANT: Use condensed prompt for retries to prevent token overload
    # Retry queries use ultra-concise guidelines (~50 tokens vs ~1400 tokens)
    # This prevents Ollama context overwhelm that can cause bad refinements
    from pipeline.prompts import get_director_refine_query_prompt_retry

    # Add agent analysis context to prompt if available
    agent_context = ""
    if hasattr(segment, 'agent_analysis') and segment.agent_analysis:
        enriched_persons = segment.agent_analysis.get('enriched_persons', {})
        if enriched_persons:
            agent_hints = []
            for person_name, data in enriched_persons.items():
                era = data.get('era', '')
                occupation = data.get('occupation', '')
                if era != 'unknown' and occupation != 'unknown':
                    agent_hints.append(f"{era} {occupation}")
            if agent_hints:
                agent_context = f"\n\nContext from research: {', '.join(agent_hints)}\nUse this to refine imagery type."

    user_prompt = get_director_refine_query_prompt_retry(
        visual_query=enriched_base,  # Use enriched base query instead of current
        transcript=getattr(segment, 'transcript', '')[:200],  # Truncate to 200 chars for token efficiency
        required_subjects=required_subjects,  # Pass as list, not string
        scores=scores_text
    ) + label_context + agent_context

    logger.debug(f"[director_agent] Using condensed retry prompt (~500 tokens vs ~2500 full prompt)")

    # Clear Ollama context to prevent contamination from previous queries
    # This prevents the LLM from being influenced by previous segment analysis or retries
    try:
        from utils.ollama_cache import clear_ollama_model_context
        logger.debug(f"[director_agent] Clearing Ollama context before retry...")
        clear_ollama_model_context(OLLAMA_MODEL)
    except Exception as e:
        logger.warning(f"[director_agent] Failed to clear Ollama context: {e}")

    prompt = f"{DIRECTOR_SYSTEM_PROMPT}\n\n{user_prompt}"

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60
        )
        refined = result.stdout.strip().split("\n")[0].strip('" ')
        if not refined:
            logger.warning("[director_agent] Empty refinement result; using current query.")
            return current_query
        if not validate_llm_query_response(refined):
            logger.warning(f"[director_agent] LLM returned invalid query: '{refined[:80]}...'; using current query.")
            return current_query

        # CRITICAL FIX: Detect and inject art style modifiers for fantasy/illustration content
        art_style = detect_art_style(segment)
        refined = inject_art_style_modifiers(refined, art_style)

        logger.info(f"[director_agent] Retry refinement (base: '{base_query}'): '{current_query}' -> '{refined}'")
        return refined
    except subprocess.TimeoutExpired:
        logger.warning("[director_agent] Ollama timed out.")
    except Exception as e:
        logger.warning(f"[director_agent] LLM refine query failed: {e}")
    return current_query


async def analyze_content(segments, script_data: dict):
    """
    Analyze all segments using ContentAnalyzer agent to extract:
    - Genre classification (documentary, fiction, fantasy, etc.)
    - Required subjects that must appear in images
    - Visual style guidance for search queries

    This runs BEFORE query refinement to inform the refinement process.

    Args:
        segments: List of video segments
        script_data: Full script JSON data

    Returns:
        segments with analysis metadata attached
    """
    logger.info("[director_agent] Starting content analysis for all segments...")

    analyzer = ContentAnalyzerAgent(model="llama3", max_iterations=3)

    for idx, seg in enumerate(segments):
        logger.info(f"[director_agent] Analyzing segment {idx + 1}/{len(segments)}")

        try:
            analysis = await analyzer.analyze_segment(seg, script_data)

            # Attach analysis to segment
            seg.genre = analysis.get('genre', 'documentary_modern')
            seg.required_subjects = analysis.get('required_subjects', [])
            seg.style_guidance = analysis.get('style_guidance', 'photorealistic')
            seg.visual_style_qualifiers = analysis.get('visual_style_qualifiers', [])
            seg.analysis_reasoning = analysis.get('reasoning', '')

            logger.info(f"[director_agent] Segment {idx} analysis:")
            logger.info(f"  Genre: {seg.genre}")
            logger.info(f"  Required subjects: {seg.required_subjects}")
            logger.info(f"  Style: {seg.style_guidance}")

        except Exception as e:
            logger.error(f"[director_agent] Content analysis failed for segment {idx}: {e}")
            # Set fallback values
            seg.genre = "documentary_modern"
            seg.required_subjects = []
            seg.style_guidance = "photorealistic"
            seg.visual_style_qualifiers = ["cinematic"]

    logger.info("[director_agent] Content analysis complete.")
    return segments


async def refine_initial_queries(segments, script_context: str = ""):
    """
    Refines all segment queries BEFORE scraping using the cinematic director prompt.
    This ensures all queries are transformed from abstract to cinematic from the start.
    Also stores the original base query for future reference during retries.

    NOTE: Should be called AFTER analyze_content() to use content analysis metadata.

    Args:
        segments: List of video segments (with content analysis attached)
        script_context: Overarching theme/intent extracted from the script
    """
    logger.info("[director_agent] Refining initial queries with cinematic director...")
    logger.info(f"[director_agent] Script context:\n{script_context}")

    for idx, seg in enumerate(segments):
        original_query = seg.visual_query
        # Store the base query before refinement for retry context
        seg.base_visual_query = original_query
        refined_query = await llm_initial_query(seg, script_context)
        seg.visual_query = refined_query
        logger.info(f"[director_agent] Segment {idx}: '{original_query}' -> '{refined_query}'")

    logger.info("[director_agent] Initial query refinement complete.")
    return segments



async def supervise_segments(segments, temp_dir: str = None):
    """
    Main orchestration function.
    - Evaluates ranking results
    - Refines queries and re-scrapes when needed
    - Updates segments in place

    Args:
        segments: List of video segments to supervise
        temp_dir: Story-specific temp directory (e.g., data/temp_images/lost_labyrinth_script)
                  If None, falls back to default segment_N structure
    """
    logger.info("[director_agent] Starting supervision pass (offline Ollama mode)...")
    if temp_dir:
        logger.info(f"[director_agent] Using story-specific temp directory: {temp_dir}")

    # Load or generate contextual labels for query refinement
    try:
        contextual_labels = _generate_contextual_labels(segments)
        logger.info(f"[director_agent] Loaded contextual labels for query refinement: "
                   f"{len(contextual_labels.get('scene_labels', []))} scene types")
    except Exception as e:
        logger.warning(f"[director_agent] Failed to load contextual labels: {e}")
        contextual_labels = None

    for idx, seg in enumerate(segments):
        debug = getattr(seg, "ranking_debug", {})
        top3 = debug.get("top3", [])
        top5 = debug.get("top5", [])
        # Accumulator of best candidates across retries (by path/url)
        accumulated = {}
        for entry in (top5 or top3 or []):
            key = entry.get("path") or entry.get("url")
            if not key:
                continue
            prev = accumulated.get(key)
            if (not prev) or float(entry.get("final_score", 0)) > float(prev.get("final_score", 0)):
                accumulated[key] = entry
        best = top3[0] if top3 else {}
        score = best.get("final_score", 0)
        original_score = score  # Store original score to compare after requery
        original_selected_image = seg.selected_image  # Store original selected image
        logger.info(f"[director_agent] Segment {idx} top image score = {score:.3f}")

        retries = 0
        while score < MIN_CONFIDENCE and retries < MAX_RETRIES:
            retries += 1
            logger.info(f"[director_agent] Low score ({score:.2f}) — refining query (attempt {retries})...")

            new_query = await llm_refine_query(seg, top3, contextual_labels)
            if new_query == seg.visual_query:
                logger.info("[director_agent] Query unchanged; breaking out early.")
                break

            seg.visual_query = new_query

            # Re-scrape and re-rank
            # Run an incremental scrape for this single segment only (preserve other segments)
            # IMPORTANT: Pass temp_dir to preserve story-specific directory structure
            new_imgs = await collect_images_for_segments(
                [seg],
                max_concurrent=1,
                wipe_temp=False,
                base_index=idx,
                temp_dir=temp_dir  # Preserve story-specific temp dir
            )
            seg.images = new_imgs[0].images
            updated_seg = rank_images([seg], base_index=idx, temp_dir=temp_dir)[0]

            debug = getattr(updated_seg, "ranking_debug", {})
            top3 = debug.get("top3", [])
            new_top5 = debug.get("top5", [])
            # Merge new top5 into accumulator
            for entry in (new_top5 or top3 or []):
                key = entry.get("path") or entry.get("url")
                if not key:
                    continue
                prev = accumulated.get(key)
                if (not prev) or float(entry.get("final_score", 0)) > float(prev.get("final_score", 0)):
                    accumulated[key] = entry
            new_score = top3[0]["final_score"] if top3 else 0

            # Only update if new score is better than original
            if new_score > original_score:
                logger.info(f"[director_agent] Improvement: {original_score:.3f} -> {new_score:.3f}")
                seg.images = updated_seg.images
                seg.ranking_debug = updated_seg.ranking_debug
                seg.selected_image = updated_seg.selected_image
                score = new_score
                original_score = new_score  # Update baseline for next retry
                original_selected_image = updated_seg.selected_image
            else:
                logger.warning(f"[director_agent] Warning: new score {new_score:.3f} <= original {original_score:.3f}, keeping original selection")
                # Restore original selected image but keep new images for potential future retries
                seg.images = updated_seg.images  # Keep new images in pool
                seg.selected_image = original_selected_image  # But keep original selection
                score = new_score  # Update score to check if we should continue retrying

            if score >= MIN_CONFIDENCE:
                logger.info(f"[director_agent] Segment {idx} reached acceptable score {score:.2f} after {retries} retry(s).")
                break
        else:
            logger.warning(f"[director_agent] Segment {idx} stuck below threshold after {retries} retries (score={score:.2f})")

        # At the end of processing this segment, attach the top-5 accumulated candidates for debugging/reporting
        try:
            combined_top5 = sorted(accumulated.values(), key=lambda x: float(x.get("final_score", 0)), reverse=True)[:5]
            if not hasattr(seg, "ranking_debug") or not isinstance(seg.ranking_debug, dict):
                seg.ranking_debug = {}
            seg.ranking_debug["accumulated_top5"] = combined_top5
        except Exception:
            pass

        # Keep only the final selection in seg.images to reduce noise downstream
        if getattr(seg, "selected_image", None):
            seg.images = [seg.selected_image]

    logger.info("[director_agent] Supervision pass complete.")
    return segments


def choose_effects_system(segment) -> str:
    """
    Decide which effects system to use based on content analysis.

    Returns:
        "standard" or "iterative"

    Logic:
    - Standard effects: Fast, pre-built effects for documentary and realistic content
    - Iterative effects: Custom MoviePy code generation for fantasy/sci-fi/complex needs
    """
    genre = getattr(segment, 'genre', 'documentary_modern')

    # Use iterative effects for fantasy, sci-fi, and animated content
    if genre in ['fiction_fantasy', 'fiction_scifi', 'animated_stylized']:
        logger.info(f"[director_agent] Segment genre '{genre}' → using iterative effects system")
        return "iterative"

    # Use standard effects for documentary and realistic fiction
    logger.info(f"[director_agent] Segment genre '{genre}' → using standard effects system")
    return "standard"

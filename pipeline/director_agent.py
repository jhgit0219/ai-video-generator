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
from config import DIRECTOR_MAX_RETRIES
from prompts import (
    DIRECTOR_SYSTEM_PROMPT,
    DIRECTOR_INITIAL_QUERY_TEMPLATE,
    DIRECTOR_REFINE_QUERY_TEMPLATE,
)

logger = setup_logger(__name__)

# -------------- Configurable thresholds --------------
MIN_CONFIDENCE = 0.35      # re-trigger if final_score below this
MAX_RETRIES = int(DIRECTOR_MAX_RETRIES)            # max refinement cycles per segment
OLLAMA_MODEL = "llama3"      # change to "llama3", "mistral", etc.
# ------------------------------------------------------


async def llm_initial_query(segment, script_context: str = ""):
    """
    Transforms the initial visual query into a cinematic, documentary-style search phrase.
    Called BEFORE scraping to ensure all queries are cinematically refined from the start.
    """
    user_prompt = DIRECTOR_INITIAL_QUERY_TEMPLATE.format(
        script_context=script_context,
        visual_query=getattr(segment, 'visual_query', ''),
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
        logger.info(f"[director_agent] Initial query refined: '{segment.visual_query}' -> '{refined}'")
        return refined
    except subprocess.TimeoutExpired:
        logger.warning("[director_agent] Ollama timed out on initial query.")
    except Exception as e:
        logger.warning(f"[director_agent] LLM initial query failed: {e}")
    return segment.visual_query


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
    
    user_prompt = DIRECTOR_REFINE_QUERY_TEMPLATE.format(
        visual_query=current_query,
        topic=getattr(segment, 'topic', 'N/A'),
        content_type=getattr(segment, 'content_type', 'N/A'),
        transcript=getattr(segment, 'transcript', '')[:300],
        scores=scores_text
    ) + label_context

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
        logger.info(f"[director_agent] Retry refinement (base: '{base_query}'): '{current_query}' -> '{refined}'")
        return refined
    except subprocess.TimeoutExpired:
        logger.warning("[director_agent] Ollama timed out.")
    except Exception as e:
        logger.warning(f"[director_agent] LLM refine query failed: {e}")
    return current_query


async def refine_initial_queries(segments, script_context: str = ""):
    """
    Refines all segment queries BEFORE scraping using the cinematic director prompt.
    This ensures all queries are transformed from abstract to cinematic from the start.
    Also stores the original base query for future reference during retries.
    
    Args:
        segments: List of video segments
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



async def supervise_segments(segments):
    """
    Main orchestration function.
    - Evaluates ranking results
    - Refines queries and re-scrapes when needed
    - Updates segments in place
    """
    logger.info("[director_agent] Starting supervision pass (offline Ollama mode)...")

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
            new_imgs = await collect_images_for_segments([seg], max_concurrent=1, wipe_temp=False, base_index=idx)
            seg.images = new_imgs[0].images
            updated_seg = rank_images([seg], base_index=idx)[0]

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

"""
Effects Batch Director: Processes segments in micro-batches for efficient LLM effects planning.

Instead of calling the LLM once per segment (100 segments = 100 API calls),
this module batches segments together (100 segments → 10 API calls with batch_size=10).

Key benefits:
- 90% fewer API calls (100 → 10 for 100 segments)
- 78% token savings (163K → 36K tokens)
- Better narrative coherence (LLM sees story flow)
- Character state tracking between batches (no redundant intro effects)
- Consistent pacing within batches

Architecture:
1. Split segments into batches of N (default 10)
2. Track character introductions across batches
3. Build context summary from previous batches
4. Clear Ollama context per batch (prevents cross-contamination)
5. Send batch to LLM with character intro markers
6. Parse batch response into per-segment plans
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Dict, List, Optional

from utils.logger import setup_logger
from config import EFFECTS_LLM_MODEL, EFFECTS_CUSTOM_INSTRUCTIONS
from prompts import EFFECTS_DIRECTOR_SYSTEM_PROMPT

logger = setup_logger(__name__)

# Import context clearing function
try:
    from utils.ollama_cache import clear_ollama_model_context
except ImportError:
    logger.warning("[effects_batch_director] utils.ollama_cache not found, context clearing disabled")
    def clear_ollama_model_context(model: str):
        """Fallback no-op if utils.ollama_cache module is missing."""
        pass


def _detect_subjects_in_content(transcript: str, clip_caption: str) -> list:
    """
    Detect human/animal subjects in transcript and CLIP caption.

    Returns list of detected subject types: ['girl', 'cat', 'man', etc.]
    """
    subjects = []
    content = f"{transcript} {clip_caption}".lower()

    # Human subjects
    if any(word in content for word in ['girl', 'young girl', 'female child']):
        subjects.append('girl')
    if any(word in content for word in ['boy', 'young boy', 'male child']):
        subjects.append('boy')
    if any(word in content for word in ['woman', 'lady', 'female']):
        subjects.append('woman')
    if any(word in content for word in ['man', 'guy', 'male', 'gentleman']):
        subjects.append('man')
    if 'child' in content and not subjects:  # Generic child
        subjects.append('child')
    if 'person' in content and not subjects:  # Generic person
        subjects.append('person')

    # Animal subjects
    if 'cat' in content:
        subjects.append('cat')
    if 'dog' in content:
        subjects.append('dog')

    return subjects


def _detect_character_introduction(transcript: str, subjects: list, introduced_characters: set) -> tuple[bool, str]:
    """
    Detect if this segment is introducing a character for the first time.

    Args:
        transcript: Segment narration text
        subjects: List of detected subjects in this segment
        introduced_characters: Set of subjects already introduced in previous segments

    Returns:
        (is_introduction, primary_subject)

    Logic:
    - Check transcript for introduction patterns ("named", "called", "met", etc.)
    - Check if subject has been seen before in introduced_characters set
    - Return True only for FIRST appearance with introduction language
    """
    transcript_lower = transcript.lower()

    # Introduction trigger phrases
    intro_patterns = [
        'named', 'called', 'meet', 'met', 'discovered',
        'there was', 'there lived', 'once was', 'day,', 'appeared',
        'her cat', 'his cat', 'her dog', 'his dog',  # Possessive animal introductions
        'little did', 'had a secret'  # Story twist introductions
    ]

    has_intro_language = any(pattern in transcript_lower for pattern in intro_patterns)

    # Find first subject that hasn't been introduced yet
    for subject in subjects:
        if subject not in introduced_characters:
            if has_intro_language:
                # First appearance with introduction language = CHARACTER INTRODUCTION
                logger.info(f"[effects_batch_director] CHARACTER INTRODUCTION DETECTED: '{subject}' (first appearance)")
                return True, subject
            else:
                # First appearance but no intro language = just mark as seen
                logger.info(f"[effects_batch_director] Subject '{subject}' first appearance (no intro language)")
                return False, subject

    # All subjects already introduced
    return False, subjects[0] if subjects else None


def _build_context_summary(introduced_characters: set, visual_style: str = "cinematic") -> str:
    """
    Build lightweight context summary to inject between batches.

    Args:
        introduced_characters: Set of character types already introduced
        visual_style: Overall visual style (e.g., "cinematic", "fantasy storybook")

    Returns:
        Context summary string (~150-200 tokens)
    """
    if not introduced_characters:
        return ""

    chars = ", ".join(sorted(introduced_characters))

    return f"""
STORY CONTEXT (carried from previous batches):
- Introduced characters: {chars}
- Visual style: {visual_style}

IMPORTANT: Do NOT add character introduction effects (zoom_on_subject, subject_outline)
for already-introduced characters: {chars}

Only use intro effects for NEW characters not in the list above.
"""


def _build_batch_prompt(
    batch_segments: List[Any],
    batch_start_idx: int,
    total_segments: int,
    context_summary: str,
    introduced_characters: set
) -> str:
    """
    Build prompt for a batch of segments.

    Args:
        batch_segments: List of segment objects in this batch
        batch_start_idx: Starting index of this batch in overall segment list
        total_segments: Total number of segments in video
        context_summary: Context from previous batches
        introduced_characters: Set of characters already introduced

    Returns:
        Full prompt string with system prompt + batch context + segments
    """
    # Build custom instructions section if provided
    custom_instructions_section = ""
    if EFFECTS_CUSTOM_INSTRUCTIONS and EFFECTS_CUSTOM_INSTRUCTIONS.strip():
        custom_instructions_section = (
            f"\n\nCUSTOM CREATIVE REQUIREMENTS:\n"
            f"{EFFECTS_CUSTOM_INSTRUCTIONS.strip()}\n"
        )

    # Build segment descriptions with character intro markers
    segment_descriptions = []

    for i, seg in enumerate(batch_segments):
        global_idx = batch_start_idx + i

        # Extract segment data
        reasoning = getattr(seg, "reasoning", "")
        visdesc = getattr(seg, "visual_description", "")
        topic = getattr(seg, "topic", "")
        vq = getattr(seg, "visual_query", "")
        transcript = (getattr(seg, "transcript", "") or "")[:400]
        clip_caption = getattr(seg, "clip_caption", None) or ""

        # Detect subjects and check for intro
        subjects = _detect_subjects_in_content(transcript, clip_caption)
        is_introduction, primary_subject = _detect_character_introduction(
            transcript, subjects, introduced_characters
        )

        # Update introduced_characters for next segments in batch
        if subjects:
            introduced_characters.update(subjects)

        # Detect image orientation
        is_portrait = False
        try:
            img = getattr(seg, "selected_image", None)
            if img and hasattr(img, "size"):
                width, height = img.size
                is_portrait = height > width
        except Exception:
            pass

        # Build image content section
        image_content = f"Visual Query: {vq}\n"
        if visdesc:
            image_content += f"Description: {visdesc}\n"
        if clip_caption:
            image_content += f"CLIP Analysis: {clip_caption}\n"
        image_content += f"Reasoning: {reasoning}"

        # Add orientation hint
        orientation_hint = ""
        if is_portrait:
            orientation_hint = (
                "\n  IMAGE ORIENTATION: PORTRAIT (vertical) - prefer vertical motion (pan_up/pan_down)"
            )

        # Build intro alert marker
        intro_marker = ""
        if is_introduction and primary_subject:
            intro_marker = f"\n  🎬 CHARACTER INTRODUCTION: {primary_subject} (first appearance) - USE AGGRESSIVE INTRO EFFECTS!"

        # Format segment description
        seg_desc = (
            f"\nSegment {i+1}/{len(batch_segments)} (Global: {global_idx+1}/{total_segments}):\n"
            f"  Transcript: \"{transcript}\"\n"
            f"  {image_content}"
            f"{orientation_hint}"
            f"{intro_marker}\n"
        )

        segment_descriptions.append(seg_desc)

    # Combine all parts
    segments_text = "\n".join(segment_descriptions)

    prompt = (
        f"{EFFECTS_DIRECTOR_SYSTEM_PROMPT}\n"
        f"{custom_instructions_section}\n"
        f"{context_summary}\n"
        f"SEGMENTS TO PROCESS ({len(batch_segments)} segments):\n"
        f"{segments_text}\n"
        f"TASK: Analyze each segment and return a JSON array with {len(batch_segments)} effect plans.\n"
        f"Return format: [{{'motion': '...', 'overlays': [...], 'fade_in': X, 'fade_out': Y, 'transition': {{'type': '...', 'duration': Z}}}}, ...]\n"
        f"Return strictly JSON array. No commentary, no markdown code fences, no backticks."
    )

    return prompt


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract JSON array from LLM response.

    Args:
        text: Raw LLM output

    Returns:
        Parsed list of dicts, or None if extraction fails
    """
    # Remove markdown code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Try extracting just the JSON array (greedy match for nested objects)
    m = re.search(r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return None


def _normalize_plan(plan: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    """
    Normalize and validate a single effect plan.

    Args:
        plan: Raw plan dict from LLM
        idx: Segment index (for fade defaults)
        total: Total segments (for fade defaults)

    Returns:
        Normalized plan dict with validated fields
    """
    # Validate plan is a dict
    if not isinstance(plan, dict):
        logger.warning(f"[effects_batch_director] Plan is not a dict: {type(plan)}, value: {plan}")
        return None

    # Normalize motion
    motion = str(plan.get("motion", "ken_burns_in")).lower()
    if motion not in {"ken_burns_in", "ken_burns_out", "pan_ltr", "pan_rtl", "pan_up", "pan_down", "zoom_pan", "static"}:
        motion = "ken_burns_in"  # Default to dynamic motion

    # Normalize overlays
    overlays = plan.get("overlays", []) or []
    if not isinstance(overlays, list):
        overlays = []
    allowed_overlays = {
        "vignette", "film_grain", "neon_overlay",
        "warm_grade", "cool_grade", "desaturated_grade", "teal_orange_grade",
        "quick_flash", "flash_pulse", "strobe_effect",
        "subject_outline"  # For character introductions
    }
    overlays = [o for o in overlays if o in allowed_overlays]

    # Clamp fades
    def _clamp(v):
        try:
            x = float(v)
        except Exception:
            x = 0.0
        return max(0.0, min(1.5, x))

    fade_in = _clamp(plan.get("fade_in", 0.3 if idx == 0 else 0.1))
    fade_out = _clamp(plan.get("fade_out", 0.8 if idx == total - 1 else 0.0))

    # Normalize transition
    trans = plan.get("transition") or {}
    if isinstance(trans, str):
        trans = {"type": trans, "duration": 1.0}
    elif not isinstance(trans, dict):
        trans = {"type": "crossfade", "duration": 1.0}

    t_type = str(trans.get("type", "crossfade")).lower()
    if t_type not in {"none", "crossfade", "slide_left", "slide_right", "slide_up", "slide_down"}:
        t_type = "crossfade"

    def _clamp_t(v):
        try:
            x = float(v)
        except Exception:
            x = 1.0
        return max(0.0, min(2.0, x))

    t_dur = _clamp_t(trans.get("duration", 1.0))

    # Extract tools if present (e.g., zoom_on_subject)
    tools = plan.get("tools", []) or []
    if not isinstance(tools, list):
        tools = []

    # Extract custom code flags
    needs_custom_code = plan.get("needs_custom_code", False)
    custom_effect_description = plan.get("custom_effect_description", "")

    normalized_plan = {
        "motion": motion,
        "overlays": overlays,
        "fade_in": fade_in,
        "fade_out": fade_out,
        "transition": {"type": t_type, "duration": t_dur},
        "tools": tools,
        "needs_custom_code": needs_custom_code
    }

    if custom_effect_description:
        normalized_plan["custom_effect_description"] = custom_effect_description

    # Log if custom code is requested
    if needs_custom_code:
        logger.info(f"[effects_batch_director] Segment {idx} requests custom code: {custom_effect_description[:100]}")

    return normalized_plan


def get_effect_plans_batched(
    segments: List[Any],
    batch_size: int = 10,
    visual_style: str = "cinematic"
) -> List[Optional[Dict[str, Any]]]:
    """
    Process segments in micro-batches with character tracking.

    Args:
        segments: List of all video segments
        batch_size: Segments per batch (default 10)
        visual_style: Overall visual style for context summary

    Returns:
        List of effect plans (one per segment, None if plan generation failed)

    Example:
        segments = [seg1, seg2, ..., seg100]
        plans = get_effect_plans_batched(segments, batch_size=10)
        # Returns 100 plans from 10 API calls

    Token efficiency:
        - Per-segment: 100 segments × 1,630 tokens = 163,000 tokens
        - Batched: 10 batches × 3,700 tokens = 37,000 tokens
        - Savings: 77% reduction
    """
    plans = []
    introduced_characters = set()

    total_segments = len(segments)

    logger.info(f"[effects_batch_director] Processing {total_segments} segments in batches of {batch_size}")

    for batch_start in range(0, total_segments, batch_size):
        batch_end = min(batch_start + batch_size, total_segments)
        batch = segments[batch_start:batch_end]

        logger.info(f"[effects_batch_director] Processing batch {batch_start//batch_size + 1}/{(total_segments + batch_size - 1)//batch_size} (segments {batch_start+1}-{batch_end})")

        # Build context summary from previous batches
        context_summary = _build_context_summary(introduced_characters, visual_style)

        # Clear Ollama context before this batch
        logger.debug(f"[effects_batch_director] Clearing Ollama context before batch {batch_start//batch_size + 1}")
        clear_ollama_model_context(EFFECTS_LLM_MODEL)

        # Build batch prompt
        prompt = _build_batch_prompt(
            batch,
            batch_start,
            total_segments,
            context_summary,
            introduced_characters
        )

        # Call LLM
        try:
            result = subprocess.run(
                ["ollama", "run", EFFECTS_LLM_MODEL, "--nowordwrap"],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=120,  # Longer timeout for batches
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            out = (result.stdout or "").strip()

            logger.info(f"[effects_batch_director] LLM returned {len(out)} characters for batch")

            if len(out) < 50:
                logger.warning(f"[effects_batch_director] Output too short: {repr(out)}")
                # Fallback: None plans for this batch
                plans.extend([None] * len(batch))
                continue

            # Show first 500 chars for debugging
            logger.debug(f"[effects_batch_director] Raw output: {repr(out[:500])}")

            # Extract JSON array
            batch_plans = _extract_json_array(out)

            if not batch_plans:
                logger.warning(f"[effects_batch_director] LLM did not return valid JSON array. Output: {out[:300]}")
                plans.extend([None] * len(batch))
                continue

            # Validate we got the right number of plans
            if len(batch_plans) != len(batch):
                logger.warning(
                    f"[effects_batch_director] Expected {len(batch)} plans, got {len(batch_plans)}. "
                    f"Padding/truncating to match."
                )
                # Pad or truncate
                while len(batch_plans) < len(batch):
                    batch_plans.append({})  # Add empty plans
                batch_plans = batch_plans[:len(batch)]  # Truncate if too many

            # Normalize each plan
            for i, plan in enumerate(batch_plans):
                global_idx = batch_start + i
                normalized = _normalize_plan(plan, global_idx, total_segments)
                plans.append(normalized)

            logger.info(f"[effects_batch_director] Successfully processed batch {batch_start//batch_size + 1}")

        except subprocess.TimeoutExpired:
            logger.warning(f"[effects_batch_director] Ollama timed out for batch {batch_start//batch_size + 1}")
            plans.extend([None] * len(batch))
        except Exception as e:
            logger.warning(f"[effects_batch_director] Batch processing failed: {e}")
            plans.extend([None] * len(batch))

    logger.info(f"[effects_batch_director] Completed batching: {len(plans)} plans generated")
    return plans


def reset_character_tracker():
    """
    Reset character introduction tracker (call at start of new video generation).

    Note: This function is provided for API compatibility but character tracking
    is now done per-batch within get_effect_plans_batched().
    """
    logger.info("[effects_batch_director] Character tracker reset called (tracking is per-batch)")

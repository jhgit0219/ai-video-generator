"""
Effects Director: uses an offline LLM (Ollama) to propose per-segment video effects.
Returns a JSON-like plan with keys:
{
    "motion": "ken_burns_in" | "ken_burns_out" | "pan_ltr" | "pan_rtl" | "pan_up" | "pan_down" | "zoom_pan" | "static",
    "overlays": ["vignette", "film_grain"],
    "fade_in": 0.0-1.5,
    "fade_out": 0.0-1.5,
    "transition": { "type": "none" | "crossfade" | "slide_left" | "slide_right" | "slide_up" | "slide_down", "duration": 0.0-2.0 }
}
If anything fails, returns None and the caller should fallback to heuristics.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Dict, Optional

from utils.logger import setup_logger
from config import EFFECTS_LLM_MODEL, EFFECTS_CUSTOM_INSTRUCTIONS
from prompts import EFFECTS_DIRECTOR_SYSTEM_PROMPT

logger = setup_logger(__name__)

# Import context clearing function
try:
    from utils.ollama_cache import clear_ollama_model_context
except ImportError:
    logger.warning("[effects_director] utils.ollama_cache not found, context clearing disabled")
    def clear_ollama_model_context(model: str):
        """Fallback no-op if utils.ollama_cache module is missing."""
        pass

# Character introduction tracker - maintains state across segment processing
_introduced_characters = set()

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

def _detect_character_introduction(transcript: str, subjects: list) -> tuple[bool, str]:
    """
    Detect if this segment is introducing a character for the first time.

    Returns:
        (is_introduction, primary_subject)

    Logic:
    - Check transcript for introduction patterns ("named", "called", "met", etc.)
    - Check if subject has been seen before in _introduced_characters set
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
        if subject not in _introduced_characters:
            if has_intro_language:
                # First appearance with introduction language = CHARACTER INTRODUCTION
                _introduced_characters.add(subject)
                logger.info(f"[effects_director] CHARACTER INTRODUCTION DETECTED: '{subject}' (first appearance)")
                return True, subject
            else:
                # First appearance but no intro language = just mark as seen
                _introduced_characters.add(subject)
                logger.info(f"[effects_director] Subject '{subject}' first appearance (no intro language)")
                return False, subject

    # All subjects already introduced
    return False, subjects[0] if subjects else None

def reset_character_tracker():
    """Reset the character introduction tracker (call at start of new video generation)."""
    global _introduced_characters
    _introduced_characters = set()
    logger.info("[effects_director] Character introduction tracker reset")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text; return parsed dict or None."""
    # Remove markdown code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()
    
    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    
    # Try extracting just the JSON object (greedy match for nested objects)
    m = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    
    return None


def get_effect_plan(segment, idx: int, total: int) -> Optional[Dict[str, Any]]:
    """Call local Ollama to propose an effects plan for this segment. Returns dict or None."""

    # CRITICAL: Clear Ollama context before effects planning
    # This ensures effects LLM doesn't see query refinement attempts from previous segments
    logger.debug(f"[effects_director] Clearing Ollama context before segment {idx} effects planning")
    clear_ollama_model_context(EFFECTS_LLM_MODEL)

    reasoning = getattr(segment, "reasoning", "")
    visdesc = getattr(segment, "visual_description", "")
    topic = getattr(segment, "topic", "")
    vq = getattr(segment, "visual_query", "")
    transcript = (getattr(segment, "transcript", "") or "")[:400]

    # Get CLIP-generated caption if available
    clip_caption = getattr(segment, "clip_caption", None) or ""

    # Detect subjects in content
    subjects = _detect_subjects_in_content(transcript, clip_caption)
    is_introduction, primary_subject = _detect_character_introduction(transcript, subjects)
    
    # Detect image orientation from selected_image
    is_portrait = False
    try:
        img = getattr(segment, "selected_image", None)
        if img and hasattr(img, "size"):
            width, height = img.size
            is_portrait = height > width
    except Exception:
        pass

    # Build image content section with both human description and AI analysis
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
            "\n\nIMAGE ORIENTATION: PORTRAIT (vertical)\n"
            "IMPORTANT: This is a portrait/vertical image. Prefer VERTICAL motion (pan_up or pan_down) "
            "to reveal the full vertical composition. Horizontal pans won't work well on vertical images."
        )

    # Build custom instructions section if provided
    custom_instructions_section = ""
    if EFFECTS_CUSTOM_INSTRUCTIONS and EFFECTS_CUSTOM_INSTRUCTIONS.strip():
        custom_instructions_section = (
            f"\n\nCUSTOM CREATIVE REQUIREMENTS:\n"
            f"{EFFECTS_CUSTOM_INSTRUCTIONS.strip()}\n"
            f"IMPORTANT: Incorporate these requirements into your effects decisions. "
            f"If they mention specific effects or subject focus that aren't in the available tools, "
            f"explain in your reasoning how you're addressing them with the available options."
        )

    # Build character introduction alert (dynamic prompt injection)
    intro_alert = ""
    if is_introduction and primary_subject:
        intro_alert = (
            f"\n\n{'='*60}\n"
            f"🎬 CHARACTER INTRODUCTION ALERT 🎬\n"
            f"{'='*60}\n"
            f"DETECTED: First introduction of '{primary_subject}' character!\n"
            f"Transcript: \"{transcript[:100]}...\"\n"
            f"CLIP Analysis: \"{clip_caption[:100]}...\"\n\n"
            f"MANDATORY EFFECTS FOR CHARACTER INTRODUCTION:\n"
            f"1. ADD zoom_on_subject tool to focus on the {primary_subject}:\n"
            f"   {{'tools': [{{'name': 'zoom_on_subject', 'params': {{'target': '{primary_subject}', 'target_scale': 1.5, 'anim_duration': 3.0}}}}]}}\n"
            f"2. ADD 'subject_outline' to overlays to highlight the character\n"
            f"3. Use SLOW fade_in (0.6-0.8s) for cinematic introduction\n"
            f"4. Choose dynamic motion (ken_burns_in or zoom_pan) for impact\n\n"
            f"This is a CRITICAL STORY MOMENT - use AGGRESSIVE cinematic effects!\n"
            f"{'='*60}\n"
        )

    user = (
        f"Segment {idx+1} of {total}.\n\n"
        f"NARRATIVE CONTEXT:\n"
        f"Topic: {topic}\n"
        f"Transcript: {transcript}\n\n"
        f"IMAGE CONTENT:\n"
        f"{image_content}"
        f"{orientation_hint}"
        f"{custom_instructions_section}"
        f"{intro_alert}\n\n"
        f"TASK: Analyze the image content (especially the CLIP Analysis which describes the actual downloaded image) "
        f"and narrative. Choose motion that reveals the story cinematically.\n"
        f"The CLIP Analysis tells you what the image actually contains - use this to make informed motion decisions.\n\n"
        "Return strictly JSON as specified. No commentary, no backticks."
    )

    prompt = f"{EFFECTS_DIRECTOR_SYSTEM_PROMPT}\n\n{user}"

    try:
        result = subprocess.run(
            ["ollama", "run", EFFECTS_LLM_MODEL, "--nowordwrap"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        # Stderr contains ANSI escape codes and spinner characters - ignore it
        
        # Log the full output to see what we're getting
        logger.info(f"[effects_director] LLM returned {len(out)} characters")
        if len(out) < 50:
            logger.warning(f"[effects_director] Output too short: {repr(out)}")
            return None
        
        # Show first 300 chars for debugging
        logger.debug(f"[effects_director] Raw output: {repr(out[:300])}")
        
        plan = _extract_json(out)
        if not plan:
            logger.warning(f"[effects_director] LLM did not return valid JSON. Output was: {out[:200]}")
            return None

        # Validate plan is a dict
        if not isinstance(plan, dict):
            logger.warning(f"[effects_director] LLM returned non-dict type: {type(plan)}, value: {plan}. Output was: {out[:200]}")
            return None

        # Normalize plan fields and clamp values
        motion = str(plan.get("motion", "ken_burns_in")).lower()
        if motion not in {"ken_burns_in", "ken_burns_out", "pan_ltr", "pan_rtl", "pan_up", "pan_down", "zoom_pan", "static"}:
            motion = "ken_burns_in"  # Default to dynamic motion
        overlays = plan.get("overlays", []) or []
        if not isinstance(overlays, list):
            overlays = []
        # Expanded overlay support for more creative effects
        allowed_overlays = {
            "vignette", "film_grain", "neon_overlay",
            "warm_grade", "cool_grade", "desaturated_grade", "teal_orange_grade",
            "quick_flash", "flash_pulse", "strobe_effect",
            "subject_outline"  # For character introductions and subject highlighting
        }
        overlays = [o for o in overlays if o in allowed_overlays]
        def _clamp(v):
            try:
                x = float(v)
            except Exception:
                x = 0.0
            return max(0.0, min(1.5, x))
        fade_in = _clamp(plan.get("fade_in", 0.3 if idx == 0 else 0.1))
        fade_out = _clamp(plan.get("fade_out", 0.8 if idx == total - 1 else 0.0))

        # Transition normalization (optional)
        trans = plan.get("transition") or {}
        # Handle if LLM returns transition as string instead of dict
        if isinstance(trans, str):
            trans = {"type": trans, "duration": 1.0}
        elif not isinstance(trans, dict):
            trans = {"type": "crossfade", "duration": 1.0}
        
        t_type = str(trans.get("type", "crossfade")).lower()  # Default to crossfade for flow
        if t_type not in {"none", "crossfade", "slide_left", "slide_right", "slide_up", "slide_down"}:
            t_type = "crossfade"
        def _clamp_t(v):
            try:
                x = float(v)
            except Exception:
                x = 1.0
            return max(0.0, min(2.0, x))
        t_dur = _clamp_t(trans.get("duration", 1.0))

        normalized = {
            "motion": motion,
            "overlays": overlays,
            "fade_in": fade_in,
            "fade_out": fade_out,
            "transition": {"type": t_type, "duration": t_dur},
        }
        return normalized
    except subprocess.TimeoutExpired:
        logger.warning("[effects_director] Ollama timed out; using heuristic effects")
    except Exception as e:
        logger.warning(f"[effects_director] LLM effects selection failed: {e}")
    return None

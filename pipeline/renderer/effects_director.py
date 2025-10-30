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
from config import EFFECTS_LLM_MODEL
from prompts import EFFECTS_DIRECTOR_SYSTEM_PROMPT

logger = setup_logger(__name__)


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
    reasoning = getattr(segment, "reasoning", "")
    visdesc = getattr(segment, "visual_description", "")
    topic = getattr(segment, "topic", "")
    vq = getattr(segment, "visual_query", "")
    transcript = (getattr(segment, "transcript", "") or "")[:400]
    
    # Get CLIP-generated caption if available
    clip_caption = getattr(segment, "clip_caption", None)
    
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

    user = (
        f"Segment {idx} of {total}.\n\n"
        f"NARRATIVE CONTEXT:\n"
        f"Topic: {topic}\n"
        f"Transcript: {transcript}\n\n"
        f"IMAGE CONTENT:\n"
        f"{image_content}"
        f"{orientation_hint}\n\n"
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
        overlays = [o for o in overlays if o in {"vignette", "film_grain"}]
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

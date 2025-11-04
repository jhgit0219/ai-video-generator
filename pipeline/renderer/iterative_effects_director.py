"""
Iterative Effects Planner with Self-Critique

Agentic effects planner that uses multiple iterations with self-feedback
to refine effects plans for fantasy/sci-fi/complex content.

Unlike standard effects_director.py (single-pass) or effects_batch_director.py (batch),
this planner iteratively refines its own output through self-critique.
"""
import json
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EffectsPlan:
    """Container for effects plan with confidence score"""
    motion: Optional[str]
    overlays: List[str]
    fade_in: float
    fade_out: float
    transition: Optional[Dict[str, Any]]
    confidence: float
    reasoning: str


class EnhancedIterativeEffectsPlanner:
    """
    Iterative effects planner with self-critique for complex content.

    Process:
    1. Generate initial effects plan
    2. Critique the plan (identify weaknesses)
    3. Refine based on critique
    4. Repeat until satisfied or max iterations reached
    """

    def __init__(
        self,
        model: str = "llama3",
        max_iterations: int = 3,
        allow_custom_code: bool = False
    ):
        """
        Initialize iterative planner.

        Args:
            model: Ollama model to use
            max_iterations: Maximum refinement iterations
            allow_custom_code: Whether to allow custom MoviePy code generation
        """
        self.model = model
        self.max_iterations = max_iterations
        self.allow_custom_code = allow_custom_code

    def plan_effects(
        self,
        segment: Dict[str, Any],
        segment_idx: int,
        total_segments: int
    ) -> Optional[Dict[str, Any]]:
        """
        Generate effects plan with iterative refinement.

        Args:
            segment: Segment dict with transcript, duration, content_type, etc.
            segment_idx: Index of this segment (0-based)
            total_segments: Total number of segments

        Returns:
            Effects plan dict or None if planning failed
        """
        logger.info(f"[iterative_planner] Starting iterative planning for segment {segment_idx}")

        best_plan = None
        best_confidence = 0.0

        for iteration in range(self.max_iterations):
            logger.debug(f"[iterative_planner] Iteration {iteration + 1}/{self.max_iterations}")

            # Generate plan (or refine previous)
            if iteration == 0:
                plan, confidence, reasoning = self._generate_initial_plan(segment, segment_idx, total_segments)
            else:
                plan, confidence, reasoning = self._refine_plan(
                    segment, segment_idx, total_segments, best_plan, critique
                )

            if plan and confidence > best_confidence:
                best_plan = plan
                best_confidence = confidence

            # Critique the plan
            critique = self._critique_plan(plan, segment, segment_idx, total_segments)

            # Stop if we're satisfied
            if confidence >= 0.9 or critique.get("satisfied", False):
                logger.info(f"[iterative_planner] Satisfied with plan after {iteration + 1} iteration(s)")
                break

        if best_plan:
            logger.info(f"[iterative_planner] Final plan confidence: {best_confidence:.2f}")
            return best_plan
        else:
            logger.warning(f"[iterative_planner] Failed to generate valid plan")
            return None

    def _generate_initial_plan(
        self,
        segment: Dict[str, Any],
        segment_idx: int,
        total_segments: int
    ) -> tuple[Optional[Dict], float, str]:
        """Generate initial effects plan."""
        prompt = self._build_planning_prompt(segment, segment_idx, total_segments)

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="ignore"
            )

            response = result.stdout.strip()
            plan_data = self._parse_plan_response(response)

            if plan_data:
                return plan_data["plan"], plan_data.get("confidence", 0.5), plan_data.get("reasoning", "")
            else:
                return None, 0.0, ""

        except Exception as e:
            logger.error(f"[iterative_planner] Error generating plan: {e}")
            return None, 0.0, ""

    def _refine_plan(
        self,
        segment: Dict[str, Any],
        segment_idx: int,
        total_segments: int,
        previous_plan: Dict,
        critique: Dict
    ) -> tuple[Optional[Dict], float, str]:
        """Refine plan based on critique."""
        prompt = f"""Refine this effects plan based on the critique below.

SEGMENT:
Transcript: {segment.get('transcript', '')}
Content Type: {segment.get('content_type', '')}
Visual Style: {segment.get('visual_description', '')}

PREVIOUS PLAN:
{json.dumps(previous_plan, indent=2)}

CRITIQUE:
{critique.get('feedback', 'No specific feedback')}

Generate an improved effects plan in JSON format."""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="ignore"
            )

            response = result.stdout.strip()
            plan_data = self._parse_plan_response(response)

            if plan_data:
                return plan_data["plan"], plan_data.get("confidence", 0.5), plan_data.get("reasoning", "")
            else:
                return previous_plan, 0.5, "Refinement failed, keeping previous plan"

        except Exception as e:
            logger.error(f"[iterative_planner] Error refining plan: {e}")
            return previous_plan, 0.5, ""

    def _critique_plan(
        self,
        plan: Dict,
        segment: Dict[str, Any],
        segment_idx: int,
        total_segments: int
    ) -> Dict[str, Any]:
        """Generate self-critique of the effects plan."""
        prompt = f"""Critique this effects plan for the given segment.

SEGMENT:
Transcript: {segment.get('transcript', '')}
Content Type: {segment.get('content_type', '')}

EFFECTS PLAN:
{json.dumps(plan, indent=2)}

Provide critique in JSON format:
{{
    "satisfied": true/false,
    "feedback": "specific improvements needed",
    "score": 0.0-1.0
}}"""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="ignore"
            )

            response = result.stdout.strip()

            # Try to parse JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                critique = json.loads(json_str)
                return critique
            else:
                return {"satisfied": True, "feedback": "", "score": 0.7}

        except Exception as e:
            logger.warning(f"[iterative_planner] Error generating critique: {e}")
            return {"satisfied": True, "feedback": "", "score": 0.7}

    def _build_planning_prompt(
        self,
        segment: Dict[str, Any],
        segment_idx: int,
        total_segments: int
    ) -> str:
        """Build prompt for effects planning."""
        position = "opening" if segment_idx == 0 else ("closing" if segment_idx == total_segments - 1 else "middle")

        prompt = f"""Plan cinematic effects for this video segment.

SEGMENT DETAILS:
Position: {position} (segment {segment_idx + 1} of {total_segments})
Transcript: {segment.get('transcript', '')}
Duration: {segment.get('duration', 5.0)}s
Content Type: {segment.get('content_type', '')}
Visual Style: {segment.get('visual_description', '')}

Generate effects plan in JSON format:
{{
    "motion": "pan_up|pan_down|pan_ltr|pan_rtl|ken_burns_in|ken_burns_out|zoom_pan|static",
    "overlays": ["vignette", "film_grain", "neon_overlay"],
    "fade_in": 0.0-2.0,
    "fade_out": 0.0-2.0,
    "transition": {{"type": "crossfade", "duration": 1.0}},
    "confidence": 0.0-1.0,
    "reasoning": "why these effects work for this content"
}}

Focus on effects that match the content's mood and pacing."""

        return prompt

    def _parse_plan_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into effects plan."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                plan_data = json.loads(json_str)
                return plan_data
            else:
                logger.warning("[iterative_planner] No JSON found in response")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"[iterative_planner] JSON parse error: {e}")
            return None

"""
DeepSeek Effects Director: Generates custom MoviePy effects code using DeepSeek Coder.

Unlike the standard effects_director.py which selects from pre-existing effects,
this director generates NEW custom effect code for each segment based on:
- Segment narrative context (transcript, topic, visual description)
- YOLO subject detection for anchor points
- Cinematic storytelling principles

The generated code is validated for safety and executed dynamically.
"""
import json
import subprocess
import re
import ast
import unicodedata
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from utils.logger import setup_logger
from config import EFFECTS_LLM_MODEL, EFFECTS_CUSTOM_INSTRUCTIONS

logger = setup_logger(__name__)

# Try to import MoviePy and detection modules
try:
    from moviepy.video.VideoClip import VideoClip
    from moviepy import ImageClip, CompositeVideoClip, ColorClip
    import numpy as np
    from PIL import Image
    MOVIEPY_AVAILABLE = True
except:
    MOVIEPY_AVAILABLE = False
    logger.warning("[deepseek_director] MoviePy not available")

try:
    from pipeline.renderer.subject_detection import detect_main_subject, detect_anchor_feature
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    logger.warning("[deepseek_director] YOLO detection not available")


@dataclass
class GeneratedEffect:
    """Container for generated custom effect code"""
    name: str
    code: str
    description: str
    segment_idx: int
    is_valid: bool = False
    error_message: Optional[str] = None


class DeepSeekEffectsDirector:
    """
    Effects director that generates custom MoviePy code for each segment.

    Uses DeepSeek Coder to create unique, context-aware effects that are:
    - Narratively appropriate
    - YOLO-aware for subject anchoring
    - Cinematically sophisticated
    - Safely validated and executed
    """

    def __init__(
        self,
        model: str = "deepseek-coder:6.7b",
        max_code_lines: int = 30,
        timeout: int = 120
    ):
        self.model = model
        self.max_code_lines = max_code_lines
        self.timeout = timeout
        self.effects_cache = {}  # Cache by segment description

    def generate_effect_for_segment(
        self,
        segment,
        segment_idx: int,
        total_segments: int,
        frame_sample: Optional[np.ndarray] = None
    ) -> Optional[GeneratedEffect]:
        """
        Generate custom effect code for a specific segment.

        Args:
            segment: VideoSegment with narrative context
            segment_idx: Index of current segment (0-based)
            total_segments: Total number of segments
            frame_sample: Optional sample frame for YOLO analysis

        Returns:
            GeneratedEffect with validated code, or None if generation failed
        """
        logger.info(f"[deepseek_director] Generating custom effect for segment {segment_idx + 1}/{total_segments}")

        # Extract segment context
        transcript = getattr(segment, "transcript", "")
        topic = getattr(segment, "topic", "")
        content_type = getattr(segment, "content_type", "")
        visual_query = getattr(segment, "visual_query", "")
        visual_description = getattr(segment, "visual_description", "")
        reasoning = getattr(segment, "reasoning", "")
        duration = float(getattr(segment, "duration", 3.0))

        # Detect subject and anchor points if YOLO is available and we have a frame
        yolo_info = ""
        if YOLO_AVAILABLE and frame_sample is not None:
            yolo_info = self._detect_subjects(frame_sample)

        # Build effect description from context
        effect_description = self._build_effect_description(
            transcript, topic, content_type, visual_description,
            reasoning, segment_idx, total_segments, duration
        )

        # Build prompt for code generation
        prompt = self._build_code_generation_prompt(
            effect_description=effect_description,
            segment_context={
                "transcript": transcript,
                "topic": topic,
                "content_type": content_type,
                "visual_query": visual_query,
                "visual_description": visual_description,
                "reasoning": reasoning,
                "duration": duration,
                "segment_idx": segment_idx,
                "total_segments": total_segments,
                "yolo_info": yolo_info
            }
        )

        # Generate code
        code_output = self._call_llm(prompt)

        if not code_output:
            logger.warning(f"[deepseek_director] Failed to generate code for segment {segment_idx}")
            return None

        raw_response = code_output.get("response", "")

        # Extract Python code
        code = self._extract_python_code(raw_response)

        if not code:
            logger.warning(f"[deepseek_director] Could not extract valid Python code for segment {segment_idx}")
            return None

        # Normalize Unicode characters
        code = self._normalize_code(code)

        logger.info(f"[deepseek_director] Generated {len(code)} chars of code")

        # Create effect object
        effect = GeneratedEffect(
            name=f"custom_seg{segment_idx}",
            code=code,
            description=effect_description,
            segment_idx=segment_idx
        )

        # Validate code
        is_valid, error = self._validate_custom_code(effect)
        effect.is_valid = is_valid
        effect.error_message = error

        if is_valid:
            logger.info(f"[deepseek_director] [OK] Code validated successfully for segment {segment_idx}")
        else:
            logger.warning(f"[deepseek_director] [ERROR] Validation failed for segment {segment_idx}: {error}")

        return effect

    def _detect_subjects(self, frame: np.ndarray) -> str:
        """
        Use YOLO to detect main subject in frame and return detection info.

        Returns:
            String with detection information for prompt
        """
        try:
            # Detect main subject (ANY class - person, vehicle, animal, building, etc.)
            subject = detect_main_subject(frame, prefer="center")

            if subject and subject["confidence"] > 0.2:
                bbox = subject["bbox"]
                class_name = subject["class_name"]
                confidence = subject["confidence"]

                x, y, w, h = bbox
                info = f"YOLO detected main subject: {class_name} (conf={confidence:.2f}) at normalized bbox({x:.2f}, {y:.2f}, {w:.2f}x{h:.2f})\n"

                # Try to detect anchor feature (class-aware: face for person, center for objects)
                try:
                    # Convert normalized bbox to absolute pixels
                    h_px, w_px = frame.shape[:2]
                    bbox_abs = (
                        int(x * w_px),
                        int(y * h_px),
                        int((x + w) * w_px),
                        int((y + h) * h_px),
                    )

                    anchor = detect_anchor_feature(
                        frame,
                        subject_mask=subject.get("mask"),
                        subject_bbox=bbox_abs,
                        feature_type="auto",
                        subject_class=class_name,
                    )
                    if anchor:
                        ax, ay = anchor
                        info += f"Anchor point at pixel coordinates ({ax}, {ay})\n"
                        if class_name == "person":
                            info += "(This is a person - anchor represents face/head center)\n"
                        else:
                            info += f"(This is a {class_name} - anchor represents geometric center)\n"
                except Exception as e:
                    logger.debug(f"[deepseek_director] Anchor detection failed: {e}")
                    pass

                info += "You can use these coordinates for subject-aware effects (zoom to subject, track motion, etc.)"
                return info
            else:
                return "YOLO detection: No clear subject detected. Use general effects."
        except Exception as e:
            logger.debug(f"[deepseek_director] YOLO detection error: {e}")
            return "YOLO detection: Not available for this frame."

    def _build_effect_description(
        self,
        transcript: str,
        topic: str,
        content_type: str,
        visual_description: str,
        reasoning: str,
        segment_idx: int,
        total_segments: int,
        duration: float
    ) -> str:
        """Build a natural language description of what effect is needed."""

        # Determine position in story
        if segment_idx == 0:
            position = "opening segment (establish tone, fade in)"
        elif segment_idx == total_segments - 1:
            position = "final segment (emotional climax, fade out)"
        else:
            position = f"middle segment ({segment_idx + 1} of {total_segments})"

        # Analyze content type for effect hints
        effect_hints = {
            "narrative_hook": "gentle zoom in, warm tones, inviting",
            "narrative_event": "dynamic motion, emphasize action, quick reveals",
            "narrative_shift": "subtle transitions, tonal changes, mystery",
            "narrative_reveal": "dramatic reveal, slow push in, emotional impact"
        }

        hint = effect_hints.get(content_type, "dynamic cinematic motion")

        description = f"""
Create a cinematic effect for this {position}:

NARRATIVE: {transcript}
TOPIC: {topic}
VISUAL: {visual_description}
PACING: {reasoning}
DURATION: {duration:.1f}s

EFFECT STYLE: {hint}
CONTENT TYPE: {content_type}

Generate MoviePy code that brings this moment to life cinematically.
"""
        return description.strip()

    def _build_code_generation_prompt(
        self,
        effect_description: str,
        segment_context: Dict[str, Any]
    ) -> str:
        """Build prompt for DeepSeek Coder with YOLO awareness."""

        # Build custom instructions section if provided
        custom_instructions_section = ""
        if EFFECTS_CUSTOM_INSTRUCTIONS and EFFECTS_CUSTOM_INSTRUCTIONS.strip():
            custom_instructions_section = f"""
CUSTOM CREATIVE REQUIREMENTS:
{EFFECTS_CUSTOM_INSTRUCTIONS.strip()}

IMPORTANT: Your generated code must incorporate these requirements. If they mention specific
effects or behaviors not shown in the examples, be creative and implement them using the
available MoviePy capabilities.
"""

        return f"""You are an expert cinematographer and Python/MoviePy developer. Generate custom cinematic effect code.

SEGMENT CONTEXT:
- Segment {segment_context['segment_idx'] + 1} of {segment_context['total_segments']}
- Duration: {segment_context['duration']:.2f}s
- Transcript: "{segment_context['transcript']}"
- Topic: {segment_context['topic']}
- Content Type: {segment_context['content_type']}
- Visual: {segment_context['visual_description']}
- Reasoning: {segment_context['reasoning']}

{segment_context.get('yolo_info', '')}
{custom_instructions_section}
EFFECT REQUIREMENT:
{effect_description}

AVAILABLE CAPABILITIES:
1. Frame-by-frame transformations (zoom, pan, rotate, warp)
2. Color grading (tints, contrast, saturation, curves)
3. Overlays (vignette, grain, light leaks, gradients)
4. Subject-aware effects (if YOLO bbox available above)
5. Time-based animations (ease in/out, bounce, elastic)

EXAMPLE 1 - Ken Burns Zoom In with Warm Tint:
```python
def apply_cinematic_intro(clip):
    '''Gentle zoom in with golden hour warmth'''
    w, h = clip.size
    duration = clip.duration

    def make_frame(get_frame, t):
        frame = get_frame(t)
        progress = t / duration

        # Zoom in smoothly
        zoom = 1.0 + (0.15 * progress)  # 1.0 to 1.15
        new_w = int(w * zoom)
        new_h = int(h * zoom)

        # Apply warm golden tint
        tint = np.array([255, 220, 180], dtype=np.float32)
        tinted = frame.astype(np.float32) * 0.8 + tint * 0.2
        frame = np.clip(tinted, 0, 255).astype(np.uint8)

        # Resize and center crop
        img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        cropped = np.array(img)[y1:y1+h, x1:x1+w]

        return cropped

    return clip.transform(make_frame)
```

EXAMPLE 2 - Pan Right with Vignette (Subject-Aware):
```python
def apply_mystery_pan(clip):
    '''Horizontal pan with darkening vignette for mystery'''
    w, h = clip.size
    duration = clip.duration
    scale = 1.2  # Pan room

    def make_frame(get_frame, t):
        frame = get_frame(t)
        progress = t / duration

        # Scale up for pan room (uniform)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)

        # Pan from left to right
        max_x = new_w - w
        x_offset = int(max_x * progress)
        y_offset = (new_h - h) // 2

        cropped = np.array(img)[y_offset:y_offset+h, x_offset:x_offset+w]

        # Add vignette
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        vignette = 1.0 - (dist / max_dist) * 0.4

        result = cropped * vignette[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    return clip.transform(make_frame)
```

CRITICAL RULES:
1. FUNCTION SIGNATURE: Use "def apply_EFFECTNAME(clip)" - ONLY take 'clip' as parameter
   - Extract duration from: clip.duration
   - Extract size from: w, h = clip.size
   - DO NOT add 'duration', 'bbox', 'w', 'h' as separate parameters!
2. MUST end with "return clip.transform(make_frame)" or "return clip.fl_image(make_frame)"
3. Keep it FOCUSED - one clear cinematic effect per function
4. Frame is numpy array (H, W, 3) uint8
5. Use numpy for speed, PIL for resize operations
6. If YOLO bbox is provided, you can use those coordinates for subject-aware effects
7. Consider segment duration and position in story
8. Match effect intensity to narrative tone
9. ALWAYS preserve aspect ratio (uniform scaling on both width and height)
10. Use smooth easing functions (progress ** 2 for ease-in, etc.)

Generate COMPLETE function NOW (must include return statement):"""

    def _extract_python_code(self, output: str) -> Optional[str]:
        """Extract Python code from LLM output"""
        # Try to extract from markdown code block first
        code_block_match = re.search(r'```python\s*(.*?)```', output, re.DOTALL)
        if code_block_match:
            code_content = code_block_match.group(1).strip()
            if 'def apply_' in code_content:
                output = code_content

        # Find the apply_ function with all its content
        lines = output.split('\n')
        start_idx = -1

        for i, line in enumerate(lines):
            if 'def apply_' in line and start_idx == -1:
                start_idx = i
                break

        if start_idx == -1:
            return None

        # Collect all lines from function start until we find the return statement
        code_lines = [lines[start_idx]]
        indent_level = None

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]

            # Detect indent level from first indented line
            if indent_level is None and line and line[0] in (' ', '\t'):
                indent_level = len(line) - len(line.lstrip())

            # Check if we've found the return statement
            if 'return clip.' in line:
                code_lines.append(line)
                break

            # Stop if we hit another top-level def (unindented)
            if line.startswith('def ') and not line.startswith(' ') and not line.startswith('\t') and i > start_idx:
                break

            code_lines.append(line)

        # Clean up trailing empty lines
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()

        result = '\n'.join(code_lines)
        return result if result else None

    def _normalize_code(self, code: str) -> str:
        """Normalize Unicode characters in code to ASCII equivalents."""
        # NFKC normalization converts fullwidth to halfwidth
        normalized = unicodedata.normalize('NFKC', code)
        # Ensure we only keep ASCII-compatible characters
        result = ''.join(char if ord(char) < 128 else ' ' for char in normalized)
        return result

    def _validate_custom_code(self, effect: GeneratedEffect) -> tuple[bool, Optional[str]]:
        """
        Validate generated code for safety and correctness.

        Returns: (is_valid, error_message)
        """
        code = effect.code

        # 1. Check for dangerous operations
        dangerous_patterns = [
            r'import\s+os\b', r'import\s+sys\b', r'import\s+subprocess\b',
            r'\bopen\s*\(', r'\bexec\s*\(', r'\beval\s*\(',
            r'__import__', r'\bcompile\s*\(',
            r'\brm\s+\-', r'\bdel\s+[a-z_]', r'\brmdir\b'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Dangerous operation detected: {pattern}"

        # 2. Check Python syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # 3. Check required structure
        if not re.search(r'def\s+apply_\w+', code):
            return False, "Missing apply_ function definition"

        if 'clip.transform' not in code and 'clip.fl_image' not in code and 'return clip' not in code:
            return False, "Function must return a transformed clip"

        # 4. Check allowed imports
        allowed_imports = ['numpy', 'PIL', 'Image', 'ImageFilter', 'ImageDraw', 'math', 're']
        imports = re.findall(r'(?:from\s+(\w+)|import\s+(\w+))', code)
        for imp in imports:
            module = imp[0] or imp[1]
            if module not in allowed_imports:
                return False, f"Disallowed import: {module}"

        # 5. Test compilation
        try:
            compile(code, '<string>', 'exec')
        except Exception as e:
            return False, f"Compilation error: {e}"

        return True, None

    def execute_effect(self, effect: GeneratedEffect, clip, segment=None) -> Optional[Any]:
        """
        Safely execute validated custom effect code on a clip.

        Args:
            effect: Validated GeneratedEffect
            clip: MoviePy VideoClip or ImageClip
            segment: Optional VideoSegment with metadata (for duration, bbox, etc.)

        Returns:
            Transformed clip or None if error
        """
        if not effect.is_valid:
            logger.error(f"[deepseek_director] Cannot execute invalid effect: {effect.error_message}")
            return None

        if not MOVIEPY_AVAILABLE:
            logger.error("[deepseek_director] MoviePy not available")
            return None

        try:
            # Create safe execution namespace
            namespace = {
                'np': np,
                'Image': Image,
                'ImageFilter': __import__('PIL.ImageFilter', fromlist=['ImageFilter']),
                'ImageDraw': __import__('PIL.ImageDraw', fromlist=['ImageDraw']),
                'VideoClip': VideoClip,
                'ImageClip': ImageClip,
                'CompositeVideoClip': CompositeVideoClip,
                'ColorClip': ColorClip,
                'math': __import__('math'),
            }

            # Execute code to define function
            exec(effect.code, namespace)

            # Find the apply_ function
            func_name = re.search(r'def\s+(apply_\w+)', effect.code).group(1)
            effect_func = namespace[func_name]

            # Inspect function signature to determine what parameters it needs
            import inspect
            sig = inspect.signature(effect_func)
            params = list(sig.parameters.keys())

            # Apply effect with appropriate parameters
            logger.info(f"[deepseek_director] Executing {func_name} for segment {effect.segment_idx}")
            logger.debug(f"[deepseek_director] Function signature: {func_name}({', '.join(params)})")

            # Build kwargs based on what the function expects
            kwargs = {}

            if 'duration' in params and clip.duration:
                kwargs['duration'] = clip.duration

            if 'bbox' in params:
                # Try to get bbox from segment or use default
                if segment and hasattr(segment, 'subject_bbox'):
                    kwargs['bbox'] = segment.subject_bbox
                else:
                    # Default centered bbox
                    kwargs['bbox'] = (0.3, 0.3, 0.4, 0.4)

            if 'w' in params and 'h' in params:
                w, h = clip.size
                kwargs['w'] = w
                kwargs['h'] = h

            if 'segment' in params:
                kwargs['segment'] = segment

            # Call with appropriate parameters
            if len(params) == 1:  # Only 'clip' parameter
                result = effect_func(clip)
            else:
                result = effect_func(clip, **kwargs)

            logger.info(f"[deepseek_director] [OK] Effect applied successfully")
            return result

        except TypeError as e:
            # Try to provide helpful error message
            logger.error(f"[deepseek_director] Parameter mismatch: {e}")
            logger.error(f"[deepseek_director] Function {func_name} expects parameters that couldn't be provided")
            logger.error(f"[deepseek_director] Try regenerating with simpler signature: def {func_name}(clip)")
            import traceback
            traceback.print_exc()
            return None

        except Exception as e:
            logger.error(f"[deepseek_director] Execution error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Call Ollama LLM"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, "--nowordwrap"],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=self.timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )

            output = (result.stdout or "").strip()
            return {"response": output}

        except subprocess.TimeoutExpired:
            logger.warning(f"[deepseek_director] LLM timeout")
            return None
        except Exception as e:
            logger.error(f"[deepseek_director] LLM error: {e}")
            return None


# Convenience function to match existing interface
def get_custom_effect_code(segment, idx: int, total: int, frame_sample: Optional[np.ndarray] = None) -> Optional[GeneratedEffect]:
    """
    Generate custom effect code for a segment using DeepSeek.

    This is the main entry point that replaces get_effect_plan() from effects_director.py

    Args:
        segment: VideoSegment with narrative context
        idx: Segment index
        total: Total segments
        frame_sample: Optional frame for YOLO analysis

    Returns:
        GeneratedEffect with validated code, or None if failed
    """
    director = DeepSeekEffectsDirector()
    return director.generate_effect_for_segment(segment, idx, total, frame_sample)

"""
Video generator responsible for final assembly and export.
Creates videos from selected images with automated cinematic effects.
"""
from typing import List
import os
import random
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from moviepy import *
from utils.logger import setup_logger
from pipeline.parser import VideoSegment
from pipeline.renderer.image_enhancer import enhance_image
from config import (
    IMAGE_SIZE,
    RENDER_SIZE,
    FPS,
    VIDEO_CODEC,
    AUDIO_CODEC,
    PRESET,
    USE_LLM_EFFECTS,
    USE_MICRO_BATCHING,
    EFFECTS_BATCH_SIZE,
    USE_DEEPSEEK_EFFECTS,
    USE_DEEPSEEK_SELECTIVE,
    DEEPSEEK_MODEL,
    DURATION_MODE,
    FINAL_DURATION_TOLERANCE,
    DEFAULT_TRANSITION_TYPE,
    DEFAULT_TRANSITION_DURATION,
    ALLOW_LLM_TRANSITIONS,
    ENABLE_AI_UPSCALE,
    ENABLE_SIMPLE_SHARPEN,
)
from pipeline.renderer.effects_director import get_effect_plan, reset_character_tracker
from pipeline.renderer.effects_batch_director import get_effect_plans_batched
from pipeline.renderer.deepseek_effects_director import get_custom_effect_code, DeepSeekEffectsDirector
from pipeline.renderer.effects_tools import TOOLS_REGISTRY
from pipeline.director_agent import choose_effects_system

# Try to import iterative effects planner (optional feature)
try:
    from pipeline.renderer.iterative_effects_director import EnhancedIterativeEffectsPlanner
    ITERATIVE_PLANNER_AVAILABLE = True
except ImportError:
    ITERATIVE_PLANNER_AVAILABLE = False
    EnhancedIterativeEffectsPlanner = None

logger = setup_logger(__name__)


class CinematicEffectsAgent:
    """Agent responsible for applying automated cinematic effects to video clips."""

    EFFECTS = [
        "ken_burns_zoom_in",
        "ken_burns_zoom_out",
        "pan_left_to_right",
        "pan_right_to_left",
        "vignette_overlay",
        "old_film_grain",
    ]

    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.render_size = RENDER_SIZE
        self.fps = FPS

    def apply_random_effect(self, clip: ImageClip, effect_name: str = None) -> ImageClip:
        """Apply a random cinematic effect to an image clip."""
        if effect_name is None:
            effect_name = random.choice(self.EFFECTS)

        logger.info(f"   Applying effect: {effect_name}")

        if effect_name == "ken_burns_zoom_in":
            return self._ken_burns_zoom_in(clip)
        elif effect_name == "ken_burns_zoom_out":
            return self._ken_burns_zoom_out(clip)
        elif effect_name == "pan_left_to_right":
            return self._pan_horizontal(clip, "left_to_right")
        elif effect_name == "pan_right_to_left":
            return self._pan_horizontal(clip, "right_to_left")
        elif effect_name == "vignette_overlay":
            return self._apply_vignette(clip)
        elif effect_name == "old_film_grain":
            return self._apply_film_grain(clip)
        else:
            return clip

    # ---------- Context-aware dispatcher ----------

    def apply_context_effect(self, clip: ImageClip, segment, idx: int, total: int) -> ImageClip:
        """Choose effects based on segment metadata (reasoning/visual_description/topic).

        AGGRESSIVE MOTION STRATEGY - Always use dynamic camera work!
        Rules of thumb:
        - Intro (idx==0): fade in + aggressive zoom in
        - Portrait images: ALWAYS use vertical pans (up/down) - more dramatic
        - Calm/mystical: zoom out + vignette (slow reveal)
        - Action/intense: pan or zoom_pan for energy
        - Complex scenes: zoom_pan (combined motion for impact)
        - Last segment: add fade out
        - Default: NEVER static - always zoom or pan
        """
        text_parts = [
            getattr(segment, "reasoning", ""),
            getattr(segment, "visual_description", ""),
            getattr(segment, "topic", ""),
            getattr(segment, "visual_query", ""),
            getattr(segment, "transcript", ""),
        ]
        ctx = (" ".join([p for p in text_parts if p]) or "").lower()

        # Detect portrait orientation
        is_portrait = False
        try:
            img = getattr(segment, "selected_image", None)
            if img and hasattr(img, "size"):
                width, height = img.size
                is_portrait = height > width
        except Exception:
            pass

        calm_words = {"calm", "soft", "quiet", "serene", "mystic", "mysterious", "ancient", "fog", "dream", "whisper"}
        action_words = {"action", "run", "storm", "battle", "explosion", "thunder", "fast", "chase", "intense", "rush"}
        complex_words = {"cityscape", "landscape", "architecture", "panorama", "vista", "complex", "detailed"}
        intro_zoom = (idx == 0)
        is_last = (idx == (total - 1))

        enhanced = clip
        # AGGRESSIVE motion choice - always dynamic!
        if intro_zoom:
            # Strong opening with aggressive zoom
            enhanced = self._ken_burns_zoom_in(enhanced)
        elif is_portrait:
            # Portrait images: ALWAYS use vertical pans (more dramatic than zoom)
            direction = "up" if (idx % 2 == 0) else "down"
            enhanced = self._pan_vertical(enhanced, direction)
        elif any(w in ctx for w in action_words):
            # Action: use ZOOM_PAN for maximum energy (50% of time), otherwise pan
            if idx % 3 == 0:
                enhanced = self._zoom_pan(enhanced)  # Combined motion for energy!
            else:
                direction = "left_to_right" if (idx % 2 == 0) else "right_to_left"
                enhanced = self._pan_horizontal(enhanced, direction)
        elif any(w in ctx for w in complex_words):
            # Complex scenes: ZOOM_PAN for impact
            enhanced = self._zoom_pan(enhanced)
        elif any(w in ctx for w in calm_words):
            # Calm: zoom out + vignette (but still moving!)
            enhanced = self._ken_burns_zoom_out(enhanced)
            enhanced = self._apply_vignette(enhanced)
        else:
            # DEFAULT: Alternate between aggressive zoom in and pans (NEVER static!)
            motion_choice = idx % 3
            if motion_choice == 0:
                enhanced = self._ken_burns_zoom_in(enhanced)  # Aggressive zoom
            elif motion_choice == 1:
                direction = "left_to_right" if (idx % 2 == 0) else "right_to_left"
                enhanced = self._pan_horizontal(enhanced, direction)
            else:
                enhanced = self._zoom_pan(enhanced)  # Combined for variety

        # Global fades
        fade_in_d = 0.6 if intro_zoom else 0.3
        fade_out_d = 0.8 if is_last else 0.0
        if fade_in_d > 0:
            enhanced = self._fade_in(enhanced, fade_in_d)
        if fade_out_d > 0:
            enhanced = self._fade_out(enhanced, fade_out_d)

        return enhanced

    def apply_plan(self, clip: ImageClip, plan: dict, segment=None) -> ImageClip:
        """Apply an LLM-provided plan to a clip.

        CRITICAL: Effects are applied in a specific order to avoid visual artifacts.
        See EFFECTS_ORDERING_REFERENCE.md for detailed explanation.

        Order: Baked Effects → Geometric Reframing → Motion → Late Overlays → Fades

        Args:
            clip: Input video clip
            plan: Effect plan dict with motion, overlays, fades, tools
            segment: Optional VideoSegment for subject detection in overlays

        Returns:
            Enhanced clip with effects applied
        """
        motion = (plan or {}).get("motion", "ken_burns_in")
        overlays = (plan or {}).get("overlays", []) or []
        fade_in = float((plan or {}).get("fade_in", 0))
        fade_out = float((plan or {}).get("fade_out", 0))
        tools = (plan or {}).get("tools", []) or []  # optional advanced tools with params

        enhanced = clip

        # ===== STAGE 2: BAKED IMAGE EFFECTS (move with camera) =====
        # These are painted onto the base image and will move during pans/zooms

        # Subject outline - MUST be before motion so it moves with the image
        if "subject_outline" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("subject_outline")
                if fn:
                    # Determine target subject from segment required_subjects
                    target_subject = None
                    if segment and hasattr(segment, 'required_subjects') and segment.required_subjects:
                        # Use first subject as primary target (e.g., "cat", "young girl")
                        target_subject = segment.required_subjects[0]
                        logger.info(f"[effects] subject_outline targeting: {target_subject}")

                    enhanced = fn(enhanced, target=target_subject)
                    logger.info("[effects] Applied subject_outline (baked, will move with motion)")
            except Exception as e:
                logger.warning(f"[effects] subject_outline failed: {e}")

        # Neon overlay - edge detection glow, baked into image
        if "neon_overlay" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("neon_overlay")
                if fn:
                    enhanced = fn(enhanced)
                    logger.info("[effects] Applied neon_overlay (baked)")
            except Exception as e:
                logger.warning(f"[effects] neon_overlay failed: {e}")

        # Film grain - texture overlay, baked into image
        if "film_grain" in overlays:
            enhanced = self._apply_film_grain(enhanced)
            logger.info("[effects] Applied film_grain (baked)")

        # ===== STAGE 3: GEOMETRIC REFRAMING (tools with params) =====
        # These change composition but preserve aspect ratio
        for tool in tools:
            try:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name", "")).strip()
                params = tool.get("params", {}) or {}
                fn = TOOLS_REGISTRY.get(name)
                if fn:
                    enhanced = fn(enhanced, **params)
                    logger.info(f"[effects] Applied tool: {name} with params {params}")
            except Exception as e:
                logger.warning(f"[effects] tool '{tool}' failed: {e}")

        # ===== STAGE 4: CAMERA MOTION (virtual camera movement) =====
        # Motion effects upscale uniformly then pan/zoom within the canvas
        if motion == "ken_burns_in":
            enhanced = self._ken_burns_zoom_in(enhanced)
        elif motion == "ken_burns_out":
            enhanced = self._ken_burns_zoom_out(enhanced)
        elif motion == "pan_ltr":
            enhanced = self._pan_horizontal(enhanced, "left_to_right")
        elif motion == "pan_rtl":
            enhanced = self._pan_horizontal(enhanced, "right_to_left")
        elif motion == "pan_up":
            enhanced = self._pan_vertical(enhanced, "up")
        elif motion == "pan_down":
            enhanced = self._pan_vertical(enhanced, "down")
        elif motion == "zoom_pan":
            enhanced = self._zoom_pan(enhanced)
        else:
            # static, keep as-is
            pass

        # ===== STAGE 6: LATE OVERLAYS (frame-locked, after motion) =====
        # These stay locked to frame corners and don't move with pans

        # Vignette - frame-locked dark corners
        if "vignette" in overlays:
            enhanced = self._apply_vignette(enhanced)
            logger.info("[effects] Applied vignette (late overlay, frame-locked)")

        # Color grading overlays (late, after all motion)
        if "warm_grade" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("warm_grade")
                if fn:
                    enhanced = fn(enhanced, intensity=0.6)
            except Exception as e:
                logger.warning(f"[effects] warm_grade failed: {e}")
        if "cool_grade" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("cool_grade")
                if fn:
                    enhanced = fn(enhanced, intensity=0.6)
            except Exception as e:
                logger.warning(f"[effects] cool_grade failed: {e}")
        if "desaturated_grade" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("desaturated_grade")
                if fn:
                    enhanced = fn(enhanced, intensity=0.5)
            except Exception as e:
                logger.warning(f"[effects] desaturated_grade failed: {e}")
        if "teal_orange_grade" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("teal_orange_grade")
                if fn:
                    enhanced = fn(enhanced)
            except Exception as e:
                logger.warning(f"[effects] teal_orange_grade failed: {e}")

        # Flash/pulse effects (late overlays)
        if "quick_flash" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("quick_flash")
                if fn:
                    enhanced = fn(enhanced, flash_time=min(1.0, enhanced.duration * 0.3))
            except Exception as e:
                logger.warning(f"[effects] quick_flash failed: {e}")
        if "flash_pulse" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("flash_pulse")
                if fn:
                    enhanced = fn(enhanced, intensity=0.6, frequency=1.5)
            except Exception as e:
                logger.warning(f"[effects] flash_pulse failed: {e}")
        if "strobe_effect" in overlays:
            try:
                fn = TOOLS_REGISTRY.get("strobe_effect")
                if fn:
                    enhanced = fn(enhanced, frequency=3.0, intensity=0.5)
            except Exception as e:
                logger.warning(f"[effects] strobe_effect failed: {e}")

        # ===== FADES (can be applied anytime, but typically last) =====
        if fade_in > 0:
            enhanced = self._fade_in(enhanced, fade_in)
        if fade_out > 0:
            enhanced = self._fade_out(enhanced, fade_out)

        return enhanced

    # ---------- Ken Burns Effects ----------

    def _ken_burns_zoom_in(self, clip: ImageClip) -> ImageClip:
        """Ken Burns effect: smooth slow zoom in with slight pan."""
        duration = clip.duration

        def transform_frame(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            
            # Smoother easing function (cubic ease-in-out)
            if progress < 0.5:
                eased_progress = 4 * progress * progress * progress
            else:
                eased_progress = 1 - pow(-2 * progress + 2, 3) / 2
            
            # Use floating-point for smoother motion
            zoom = 1.0 + (0.15 * eased_progress)
            h, w = frame.shape[:2]
            
            # Calculate crop dimensions with floating-point precision
            new_h = h / zoom
            new_w = w / zoom
            
            # Smooth pan with floating-point offsets
            pan_x = (w - new_w) / 2 * (1 + 0.1 * eased_progress)
            pan_y = (h - new_h) / 2
            
            # Convert to integer only at the end for cropping
            x1, y1 = int(pan_x), int(pan_y)
            x2, y2 = int(pan_x + new_w), int(pan_y + new_h)
            
            # Ensure bounds are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cropped = frame[y1:y2, x1:x2]
            img = Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
            return np.array(img)

        return clip.transform(transform_frame)

    def _ken_burns_zoom_out(self, clip: ImageClip) -> ImageClip:
        """Ken Burns effect: smooth slow zoom out."""
        duration = clip.duration

        def transform_frame(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            
            # Smoother easing function (cubic ease-in-out)
            if progress < 0.5:
                eased_progress = 4 * progress * progress * progress
            else:
                eased_progress = 1 - pow(-2 * progress + 2, 3) / 2
            
            # Use floating-point for smoother motion
            zoom = 1.15 - (0.15 * eased_progress)
            h, w = frame.shape[:2]
            
            # Calculate crop dimensions with floating-point precision
            new_h = h / zoom
            new_w = w / zoom
            
            # Smooth pan with floating-point offsets
            pan_x = (w - new_w) / 2
            pan_y = (h - new_h) / 2

            if new_h > h or new_w > w:
                # Zooming out beyond frame - needs padding
                target_w, target_h = int(new_w), int(new_h)
                img = Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                offset_x = (w - target_w) // 2
                offset_y = (h - target_h) // 2
                canvas.paste(img, (offset_x, offset_y))
                return np.array(canvas)

            # Convert to integer only at the end for cropping
            x1, y1 = int(pan_x), int(pan_y)
            x2, y2 = int(pan_x + new_w), int(pan_y + new_h)
            
            # Ensure bounds are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cropped = frame[y1:y2, x1:x2]
            img = Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
            return np.array(img)

        return clip.transform(transform_frame)

    # ---------- Pan Effects ----------

    def _pan_horizontal(self, clip: ImageClip, direction: str) -> ImageClip:
        """Smooth horizontal pan with uniform upscale to avoid distortion and black bars."""
        duration = clip.duration
        w, h = clip.size

        # Uniform scale factor to create pan room (15%) without changing aspect ratio
        scale = 1.15
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        max_x = max(0, new_w - w)

        def transform_frame(get_frame, t):
            progress = t / max(1e-6, duration)
            # Smooth easing function (ease in-out)
            progress = progress * progress * (3 - 2 * progress)

            frame = get_frame(t)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Uniform upscale, then crop
            img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Horizontal pan amount across the available room
            if direction == "left_to_right":
                x1 = int(round(max_x * progress))
            else:  # right_to_left
                x1 = int(round(max_x * (1.0 - progress)))
            x1 = max(0, min(max_x, x1))
            x2 = x1 + w
            # Keep vertical crop centered
            y1 = (new_h - h) // 2
            y2 = y1 + h

            cropped = np.array(img)[y1:y2, x1:x2]
            return cropped

        return clip.transform(transform_frame)

    def _pan_vertical(self, clip: ImageClip, direction: str) -> ImageClip:
        """Smooth vertical pan with uniform upscale to avoid distortion and black bars."""
        duration = clip.duration
        w, h = clip.size

        # Uniform scale factor to create pan room (15%) without changing aspect ratio
        scale = 1.15
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        max_y = max(0, new_h - h)

        def transform_frame(get_frame, t):
            progress = t / max(1e-6, duration)
            # Smooth easing function (ease in-out)
            progress = progress * progress * (3 - 2 * progress)

            frame = get_frame(t)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Uniform upscale, then crop
            img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Vertical pan amount across the available room
            if direction == "down":
                y1 = int(round(max_y * progress))
            else:  # up
                y1 = int(round(max_y * (1.0 - progress)))
            y1 = max(0, min(max_y, y1))
            y2 = y1 + h
            # Keep horizontal crop centered
            x1 = (new_w - w) // 2
            x2 = x1 + w

            cropped = np.array(img)[y1:y2, x1:x2]
            return cropped

        return clip.transform(transform_frame)

    def _zoom_pan(self, clip: ImageClip) -> ImageClip:
        """Combined zoom in with diagonal pan for dynamic energy with smooth easing."""
        duration = clip.duration

        def transform_frame(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            # Smooth easing function (ease in-out)
            eased_progress = progress * progress * (3 - 2 * progress)
            
            zoom = 1.0 + (0.20 * eased_progress)  # Smooth zoom
            h, w = frame.shape[:2]
            new_h, new_w = int(h / zoom), int(w / zoom)

            # Diagonal pan from top-left to center-right with smooth movement
            pan_x = int((w - new_w) * (0.1 + 0.4 * eased_progress))
            pan_y = int((h - new_h) * (0.1 + 0.3 * eased_progress))

            # Clamp to valid bounds
            pan_x = max(0, min(pan_x, w - new_w))
            pan_y = max(0, min(pan_y, h - new_h))

            cropped = frame[pan_y:pan_y + new_h, pan_x:pan_x + new_w]
            img = Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
            return np.array(img)

        return clip.transform(transform_frame)

    # ---------- Slide Transitions ----------

    def _slide_transition(self, clip1: ImageClip, clip2: ImageClip, direction: str, duration: float) -> VideoClip:
        """Create a slide/swipe transition between two clips."""
        # Use actual render size from clips
        w, h = clip1.size
        
        def make_frame(t):
            progress = min(1.0, t / duration)
            
            # Get frames and ensure they're the correct size
            frame1 = clip1.get_frame(min(t, clip1.duration - 0.001))
            frame2 = clip2.get_frame(0)  # Start of second clip
            
            # Resize frames to render size if they're different (e.g., from pan effects)
            if frame1.shape[:2] != (h, w):
                # Convert to PIL, resize, convert back to numpy
                img1 = Image.fromarray(frame1.astype('uint8'))
                img1 = img1.resize((w, h), Image.LANCZOS)
                frame1 = np.array(img1)
            if frame2.shape[:2] != (h, w):
                img2 = Image.fromarray(frame2.astype('uint8'))
                img2 = img2.resize((w, h), Image.LANCZOS)
                frame2 = np.array(img2)
            
            if direction == "slide_left":
                # Slide from right to left
                offset = int(w * progress)
                result = np.zeros((h, w, 3), dtype=np.uint8)
                # Outgoing clip (left side)
                result[:, :max(0, w - offset)] = frame1[:, offset:w]
                # Incoming clip (right side)
                result[:, max(0, w - offset):] = frame2[:, :min(w, offset)]
            elif direction == "slide_right":
                # Slide from left to right
                offset = int(w * progress)
                result = np.zeros((h, w, 3), dtype=np.uint8)
                result[:, :max(0, w - offset)] = frame2[:, offset:w]
                result[:, max(0, w - offset):] = frame1[:, :min(w, offset)]
            elif direction == "slide_up":
                # Slide from bottom to top
                offset = int(h * progress)
                result = np.zeros((h, w, 3), dtype=np.uint8)
                result[:max(0, h - offset), :] = frame1[offset:h, :]
                result[max(0, h - offset):, :] = frame2[:min(h, offset), :]
            elif direction == "slide_down":
                # Slide from top to bottom
                offset = int(h * progress)
                result = np.zeros((h, w, 3), dtype=np.uint8)
                result[:max(0, h - offset), :] = frame2[offset:h, :]
                result[max(0, h - offset):, :] = frame1[:min(h, offset), :]
            else:
                result = frame1
            
            return result
        
        return VideoClip(make_frame, duration=duration)

    # ---------- Static Frame Filters ----------

    def _apply_vignette(self, clip: ImageClip) -> ImageClip:
        """Darkened corners / vignette overlay."""
        def vignette_transform(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            # Increase darkening factor from 0.6 to 0.85 for stronger effect
            mask = 1 - (dist / dist.max()) * 0.85
            # Reduce minimum from 0.4 to 0.15 for darker corners
            mask = np.clip(mask, 0.15, 1.0)
            return (frame * mask[:, :, np.newaxis]).astype("uint8")

        return clip.transform(vignette_transform)

    def _apply_film_grain(self, clip: ImageClip) -> ImageClip:
        """Adds subtle film grain and muted sepia tint."""
        def film_grain_transform(get_frame, t):
            frame = get_frame(t).astype("float32")
            h, w = frame.shape[:2]

            # Add subtle Gaussian noise (grain)
            noise = np.random.normal(0, 6, (h, w, 3))  # reduced std deviation
            grain = np.clip(frame + noise, 0, 255)

            # Soft sepia tone (darker and less saturated)
            sepia = np.array([[0.45, 0.38, 0.32],
                            [0.43, 0.37, 0.31],
                            [0.36, 0.31, 0.26]])

            # Apply tint (matrix multiplication)
            tinted = np.einsum("ijk,kl->ijl", grain / 255.0, sepia) * 255.0

            # Blend original and tinted (preserve some color detail)
            blended = 0.7 * grain + 0.3 * tinted

            # Slight contrast and gamma adjustment
            blended = np.clip(blended * 0.95, 0, 255)
            return blended.astype("uint8")

        return clip.transform(film_grain_transform)

    # ---------- Fades ----------

    def _fade_in(self, clip: ImageClip, duration: float) -> ImageClip:
        duration = max(0.0, min(duration, clip.duration or duration))
        if duration == 0:
            return clip

        def fade_in_tf(get_frame, t):
            frame = get_frame(t)
            a = min(1.0, max(0.0, t / duration))
            return (frame.astype("float32") * a).clip(0, 255).astype("uint8")

        return clip.transform(fade_in_tf)

    def _fade_out(self, clip: ImageClip, duration: float) -> ImageClip:
        duration = max(0.0, min(duration, clip.duration or duration))
        if duration == 0:
            return clip

        total = clip.duration or duration

        def fade_out_tf(get_frame, t):
            frame = get_frame(t)
            a = min(1.0, max(0.0, (total - t) / duration))
            return (frame.astype("float32") * a).clip(0, 255).astype("uint8")

        return clip.transform(fade_out_tf)


# ---------- Final Video Renderer ----------

def render_final_video(segments: List[VideoSegment], audio_file: str, output_name: str = None) -> str:
    """Render final cinematic video with effects.

    Args:
        segments: List of video segments to render
        audio_file: Path to audio file
        output_name: Optional custom output filename (without extension)

    Returns:
        Path to rendered video file
    """
    logger.info("[video_generator] Starting final video rendering with cinematic effects")

    # Reset character introduction tracker for new video
    reset_character_tracker()

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    effects_agent = CinematicEffectsAgent()
    video_clips = []
    plans = []  # holds effect/transition plans per segment (may be None)
    transitions = []  # per-segment transition dicts (to next clip)

    # Use the last segment's end_time as the target duration
    total_json_duration = float(segments[-1].end_time) if segments else 0.0

    # Pre-compute all effect plans using micro-batching if enabled
    precomputed_plans = None
    if USE_LLM_EFFECTS and USE_MICRO_BATCHING and not USE_DEEPSEEK_EFFECTS:
        logger.info(f"[video_generator] Pre-computing effect plans for {len(segments)} segments using micro-batching (batch_size={EFFECTS_BATCH_SIZE})")
        precomputed_plans = get_effect_plans_batched(
            segments,
            batch_size=EFFECTS_BATCH_SIZE,
            visual_style="cinematic"  # TODO: Could extract from content analysis
        )
        logger.info(f"[video_generator] Micro-batching complete: {len([p for p in precomputed_plans if p])} valid plans generated")

    for i, seg in enumerate(segments):
        logger.info(f"[video_generator] Processing segment {i}...")
        missing_image = (not seg.selected_image or not os.path.exists(seg.selected_image))

        try:
            # Use exact JSON duration (no quantization to avoid cumulative rounding errors)
            seg_dur = float(getattr(seg, "duration", 0) or 0)
            logger.info(f"   Segment {i}: duration={seg_dur:.3f}s")
            
            if missing_image:
                logger.warning(f"   No valid image for segment {i}, using black placeholder.")
                # ColorClip size: (width, height)
                clip = ColorClip(size=(RENDER_SIZE[0], RENDER_SIZE[1]), color=(0, 0, 0), duration=seg_dur)
            else:
                # Enhance image at render size
                if ENABLE_AI_UPSCALE or ENABLE_SIMPLE_SHARPEN:
                    logger.debug(f"   Enhancing image for segment {i} at render size")
                    enhanced_img = enhance_image(seg.selected_image, RENDER_SIZE)
                    if enhanced_img:
                        # Convert PIL Image to numpy array for MoviePy
                        # Keep the large size for ken_burns effects to work properly
                        img_array = np.array(enhanced_img)
                        clip = ImageClip(img_array, duration=seg_dur)
                        logger.debug(f"   Loaded clip size: {clip.size} (will resize after effects)")
                    else:
                        # Fallback to original if enhancement fails
                        clip = ImageClip(seg.selected_image, duration=seg_dur)
                        clip = clip.resized(width=RENDER_SIZE[0], height=RENDER_SIZE[1])
                else:
                    clip = ImageClip(seg.selected_image, duration=seg_dur)
                    clip = clip.resized(width=RENDER_SIZE[0], height=RENDER_SIZE[1])
            # Effect selection: Micro-batched LLM effects with selective DeepSeek enhancement
            plan = None
            needs_custom_code = False  # Flag to trigger DeepSeek for complex segments

            # STEP 1: Get base plan from batching or per-segment LLM
            if USE_LLM_EFFECTS:
                # Use precomputed plan from micro-batching if available
                if precomputed_plans is not None and i < len(precomputed_plans):
                    plan = precomputed_plans[i]
                    if plan:
                        logger.info(f"[video_generator] Using precomputed plan from micro-batching for segment {i}")
                        # Check if plan requests custom code generation
                        needs_custom_code = plan.get("needs_custom_code", False)
                    else:
                        logger.warning(f"[video_generator] Precomputed plan was None, using fallback for segment {i}")
                else:
                    # Fallback to per-segment planning (when batching disabled or iterative planner needed)
                    # Choose effects system based on content analysis
                    effects_system = choose_effects_system(seg)

                    if effects_system == "iterative":
                        # Use iterative effects planner for fantasy/sci-fi/complex content
                        if not ITERATIVE_PLANNER_AVAILABLE:
                            logger.warning(f"[video_generator] Iterative planner not available, falling back to batch effects")
                            effects_system = "batch"  # Fallback to batch planner
                        else:
                            logger.info(f"[video_generator] Using iterative effects planner for segment {i}")
                            planner = EnhancedIterativeEffectsPlanner(
                                model="llama3",
                                max_iterations=3,
                                allow_custom_code=False  # Stick to pre-built effects for safety
                            )

                        # Convert segment to dict format expected by planner
                        seg_dict = {
                            "transcript": seg.transcript,
                            "duration": seg.duration,
                            "topic": getattr(seg, 'topic', ''),
                            "content_type": getattr(seg, 'content_type', ''),
                            "reasoning": getattr(seg, 'reasoning', ''),
                            "visual_description": getattr(seg, 'visual_description', '')
                        }

                        plan = planner.plan_effects(seg_dict, i, len(segments))
                        if plan:
                            enhanced = effects_agent.apply_plan(clip, plan, segment=seg)
                        else:
                            logger.warning(f"[video_generator] Iterative planner failed, using fallback")
                            enhanced = effects_agent.apply_context_effect(clip, seg, i, len(segments))
                    else:
                        # Use standard effects director for documentary/realistic content
                        logger.info(f"[video_generator] Using standard effects director for segment {i}")
                        plan = get_effect_plan(seg, i, len(segments))
                        if plan:
                            enhanced = effects_agent.apply_plan(clip, plan, segment=seg)
                        else:
                            enhanced = effects_agent.apply_context_effect(clip, seg, i, len(segments))
            else:
                plan = {}

            # STEP 2: Apply effects (standard or DeepSeek-enhanced)
            if (USE_DEEPSEEK_EFFECTS or (USE_DEEPSEEK_SELECTIVE and needs_custom_code)) and plan:
                # Use DeepSeek to generate custom code (either for all segments or selectively)
                try:
                    sample_frame = clip.get_frame(0)
                except:
                    sample_frame = None

                reason = "planner requested custom code" if needs_custom_code else "USE_DEEPSEEK_EFFECTS enabled"
                custom_desc = plan.get("custom_effect_description", "N/A")
                logger.info(f"[video_generator] Generating custom DeepSeek effect for segment {i + 1} ({reason})")
                if needs_custom_code:
                    logger.info(f"[video_generator] Custom effect requirement: {custom_desc}")
                generated_effect = get_custom_effect_code(seg, i, len(segments), sample_frame)

                if generated_effect and generated_effect.is_valid:
                    director = DeepSeekEffectsDirector(model=DEEPSEEK_MODEL)
                    enhanced = director.execute_effect(generated_effect, clip, segment=seg)
                    if enhanced is None:
                        logger.warning(f"[video_generator] DeepSeek effect failed, applying standard plan")
                        enhanced = effects_agent.apply_plan(clip, plan, segment=seg) if plan else effects_agent.apply_context_effect(clip, seg, i, len(segments))
                    plan["custom_effect"] = generated_effect.name
                else:
                    logger.warning(f"[video_generator] DeepSeek generation failed, applying standard plan")
                    enhanced = effects_agent.apply_plan(clip, plan, segment=seg) if plan else effects_agent.apply_context_effect(clip, seg, i, len(segments))
            elif plan:
                # Apply standard batched plan
                enhanced = effects_agent.apply_plan(clip, plan, segment=seg)
            else:
                # Fallback to heuristic effects
                enhanced = effects_agent.apply_context_effect(clip, seg, i, len(segments))

            plans.append(plan)

            # Optional: apply custom tools specified directly on the segment
            try:
                custom_tools = getattr(seg, "custom_effects", None) or []
                for tool in custom_tools:
                    if isinstance(tool, dict):
                        name = str(tool.get("name", "")).strip()
                        params = tool.get("params", {}) or {}
                        fn = TOOLS_REGISTRY.get(name)
                        if fn:
                            enhanced = fn(enhanced, **params)
            except Exception as e:
                logger.warning(f"[effects] custom_effects failed for segment {i}: {e}")

            # CRITICAL: Resize/crop to RENDER_SIZE AFTER effects (zoom-to-cover, no black bars)
            # Ken Burns needs the large canvas to pan/zoom, so we resize last
            if enhanced.size != (RENDER_SIZE[0], RENDER_SIZE[1]):
                curr_w, curr_h = enhanced.size
                target_w, target_h = RENDER_SIZE
                
                # Calculate scale to ensure BOTH width >= target AND height >= target (zoom-to-cover)
                scale_w = target_w / curr_w
                scale_h = target_h / curr_h
                scale = max(scale_w, scale_h)  # Use the larger scale to cover both dimensions
                
                # Resize maintaining aspect ratio
                new_w = int(curr_w * scale)
                new_h = int(curr_h * scale)
                logger.info(f"   Zoom-to-cover: {enhanced.size} -> ({new_w}, {new_h}) -> crop to {RENDER_SIZE}")
                enhanced = enhanced.resized(width=new_w, height=new_h)
                
                # Now crop to exact RENDER_SIZE from center
                if enhanced.size != (target_w, target_h):
                    x1 = (new_w - target_w) // 2
                    y1 = (new_h - target_h) // 2
                    x2 = x1 + target_w
                    y2 = y1 + target_h
                    enhanced = enhanced.cropped(x1=x1, y1=y1, x2=x2, y2=y2)

            # Determine transition to next clip
            if ALLOW_LLM_TRANSITIONS and plan and isinstance(plan.get("transition"), dict):
                t = plan.get("transition") or {}
                t_type = str(t.get("type", DEFAULT_TRANSITION_TYPE)).lower()
                t_dur = float(t.get("duration", DEFAULT_TRANSITION_DURATION) or 0.0)
            else:
                t_type = str(DEFAULT_TRANSITION_TYPE).lower()
                t_dur = float(DEFAULT_TRANSITION_DURATION or 0.0)
            # Last segment: no transition by default
            if i == len(segments) - 1:
                t_type, t_dur = "none", 0.0
            transitions.append({"type": t_type, "duration": max(0.0, t_dur)})
            video_clips.append(enhanced)
            logger.info(f"   [OK] Segment {i} processed {'LLM plan' if USE_LLM_EFFECTS else 'context-aware'}")
        except Exception as e:
            logger.error(f"   Error processing segment {i}: {e}")
            logger.error(traceback.format_exc())

    if not video_clips:
        logger.error("[video_generator] No video clips generated!")
        return None

    logger.info(f"[video_generator] Building timeline for {len(video_clips)} clips...")
    
    # Check if we have crossfade transitions
    has_crossfades = any((t.get("type") == "crossfade" and t.get("duration", 0) > 0) for t in transitions)
    
    if has_crossfades:
        # Apply crossfades by fading out previous clip and fading in next clip
        # WITHOUT overlapping (to preserve total duration)
        logger.info(f"[video_generator] Applying crossfades without reducing total duration")
        
        processed_clips = []
        
        for i, clip in enumerate(video_clips):
            tinfo = transitions[i]
            t_type = tinfo.get("type", "none")
            t_dur = float(tinfo.get("duration", 0.0) or 0.0)
            
            # Apply fade-out to current clip if next transition is crossfade
            if i < len(video_clips) - 1:  # Not the last clip
                next_tinfo = transitions[i + 1] if i + 1 < len(transitions) else {}
                next_t_type = next_tinfo.get("type", "none")
                next_t_dur = float(next_tinfo.get("duration", 0.0) or 0.0)
                
                if next_t_type == "crossfade" and next_t_dur > 0:
                    # Fade out at the end of this clip
                    fade_out_duration = min(next_t_dur, clip.duration)
                    
                    def make_fade_out(fade_dur, clip_dur):
                        def fade_out_transform(get_frame, t):
                            frame = get_frame(t)
                            fade_start = clip_dur - fade_dur
                            if t >= fade_start:
                                # Fade from 1 to 0
                                opacity = 1.0 - ((t - fade_start) / fade_dur)
                                return (frame * opacity).astype('uint8')
                            return frame
                        return fade_out_transform
                    
                    clip = clip.transform(make_fade_out(fade_out_duration, clip.duration))
            
            # Apply fade-in to current clip if its own transition is crossfade
            if i > 0 and t_type == "crossfade" and t_dur > 0:
                fade_in_duration = min(t_dur, clip.duration)
                
                def make_fade_in(fade_dur):
                    def fade_in_transform(get_frame, t):
                        frame = get_frame(t)
                        if t < fade_dur:
                            opacity = t / fade_dur  # 0 to 1
                            return (frame * opacity).astype('uint8')
                        return frame
                    return fade_in_transform
                
                clip = clip.transform(make_fade_in(fade_in_duration))
            
            processed_clips.append(clip)
        
        # Concatenate clips sequentially (no overlap, preserves total duration)
        final_video = concatenate_videoclips(processed_clips, method="chain")
        
        logger.info(f"[video_generator] Timeline built: duration={final_video.duration:.3f}s, target={total_json_duration:.3f}s")
    else:
        # No transitions: simple concatenation
        logger.info(f"[video_generator] No crossfades, using simple concatenation")
        final_video = concatenate_videoclips(video_clips, method="chain")

    audio_clip = None
    if os.path.exists(audio_file):
        logger.info(f"[video_generator] Adding audio: {audio_file}")
        audio_clip = AudioFileClip(audio_file)
        final_video = final_video.with_audio(audio_clip)
    else:
        logger.warning(f"[video_generator] Audio file not found: {audio_file}")

    # Always use audio duration as target if audio exists, otherwise use JSON duration
    target_duration = float(total_json_duration)
    if audio_clip is not None:
        target_duration = float(audio_clip.duration or target_duration)
        logger.info(f"[video_generator] Using audio duration as target: {target_duration:.3f}s")
    else:
        logger.info(f"[video_generator] Using JSON duration as target: {target_duration:.3f}s")

    current_duration = float(final_video.duration or 0)
    if abs(current_duration - target_duration) > (FINAL_DURATION_TOLERANCE or 0):
        logger.info(
            f"[video_generator] Adjusting final duration from {current_duration:.3f}s to {target_duration:.3f}s"
        )
        final_video = final_video.with_duration(target_duration)

    # CRITICAL: Force final video to exact IMAGE_SIZE dimensions without stretching
    current_size = final_video.size
    logger.info(f"[video_generator] Current video size: {current_size}, target: {IMAGE_SIZE}")
    if current_size != (IMAGE_SIZE[0], IMAGE_SIZE[1]):
        logger.info(f"[video_generator] Center-cropping video to {IMAGE_SIZE}")
        # Calculate crop coordinates for center crop
        x1 = (current_size[0] - IMAGE_SIZE[0]) // 2
        y1 = (current_size[1] - IMAGE_SIZE[1]) // 2
        x2 = x1 + IMAGE_SIZE[0]
        y2 = y1 + IMAGE_SIZE[1]
        final_video = final_video.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        logger.info(f"[video_generator] Video already at correct size {IMAGE_SIZE}")

    # Determine output filename
    if output_name:
        filename = f"{output_name}.mp4"
    else:
        filename = "final_video.mp4"
    
    output_path = output_dir / filename
    logger.info(f"[video_generator] Exporting to: {output_path}")
    logger.info(f"[video_generator] Using codec: {VIDEO_CODEC} (preset: {PRESET})")

    # Prepare encoding parameters based on codec
    write_params = {
        "fps": FPS,
        "codec": VIDEO_CODEC,
        "audio_codec": AUDIO_CODEC,
        "threads": 16,  # Increased from 4 to 16 for better multi-core utilization
        "verbose": False,  # Reduce console output for cleaner logs
        # Keep logger enabled to see progress (set to None to hide progress bar)
    }
    
    if VIDEO_CODEC == "h264_nvenc":
        # NVIDIA NVENC encoder - pass preset via ffmpeg_params
        write_params["ffmpeg_params"] = [
            "-preset", PRESET,  # p1-p7 quality preset
            "-rc:v", "vbr",     # Variable bitrate
            "-cq:v", "19",      # Constant quality (lower = better, 19 is high quality)
            "-b:v", "0",        # Let CQ control bitrate
            "-maxrate", "8M",   # Max bitrate cap
            "-bufsize", "16M",  # Buffer size
        ]
    elif VIDEO_CODEC == "libx264":
        # Software encoder - use preset parameter directly
        write_params["preset"] = PRESET if PRESET in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"] else "medium"
    
    final_video.write_videofile(str(output_path), **write_params)

    logger.info("[video_generator] Video rendering complete!")
    return str(output_path)

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
    DURATION_MODE,
    FINAL_DURATION_TOLERANCE,
    DEFAULT_TRANSITION_TYPE,
    DEFAULT_TRANSITION_DURATION,
    ALLOW_LLM_TRANSITIONS,
    ENABLE_AI_UPSCALE,
    ENABLE_SIMPLE_SHARPEN,
)
from pipeline.renderer.effects_director import get_effect_plan

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

    def apply_plan(self, clip: ImageClip, plan: dict) -> ImageClip:
        """Apply an LLM-provided plan to a clip."""
        motion = (plan or {}).get("motion", "ken_burns_in")
        overlays = (plan or {}).get("overlays", []) or []
        fade_in = float((plan or {}).get("fade_in", 0))
        fade_out = float((plan or {}).get("fade_out", 0))

        enhanced = clip
        # Motion
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

        # Overlays
        if "vignette" in overlays:
            enhanced = self._apply_vignette(enhanced)
        if "film_grain" in overlays:
            enhanced = self._apply_film_grain(enhanced)

        # Fades
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
            # Ensure frame is uint8 numpy array
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
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
        """Smooth horizontal pan using MoviePy's resize positioning."""
        duration = clip.duration
        # Use clip's actual size (RENDER_SIZE) instead of hardcoded IMAGE_SIZE
        w, h = clip.size
        
        def position(t):
            progress = t / duration
            # Smooth easing function (ease in-out)
            progress = progress * progress * (3 - 2 * progress)
            
            # Pan by 15% of width
            if direction == "left_to_right":
                x_offset = -int(w * 0.15 * (1 - progress))
            else:  # right_to_left
                x_offset = -int(w * 0.15 * progress)
            
            return (x_offset, 'center')
        
        # Resize clip to 115% width to allow for pan room
        resized = clip.resized(width=int(w * 1.15), height=h)
        return resized.with_position(position)

    def _pan_vertical(self, clip: ImageClip, direction: str) -> ImageClip:
        """Smooth vertical pan using MoviePy's resize positioning."""
        duration = clip.duration
        # Use clip's actual size (RENDER_SIZE) instead of hardcoded IMAGE_SIZE
        w, h = clip.size
        
        def position(t):
            progress = t / duration
            # Smooth easing function (ease in-out)
            progress = progress * progress * (3 - 2 * progress)
            
            # Pan by 15% of height
            if direction == "down":
                y_offset = -int(h * 0.15 * (1 - progress))
            else:  # up
                y_offset = -int(h * 0.15 * progress)
            
            return ('center', y_offset)
        
        # Resize clip to 115% height to allow for pan room
        resized = clip.resized(width=w, height=int(h * 1.15))
        return resized.with_position(position)

    def _zoom_pan(self, clip: ImageClip) -> ImageClip:
        """Combined zoom in with diagonal pan for dynamic energy with smooth easing."""
        duration = clip.duration

        def transform_frame(get_frame, t):
            frame = get_frame(t)
            # Ensure frame is uint8 numpy array
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
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

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    effects_agent = CinematicEffectsAgent()
    video_clips = []
    plans = []  # holds effect/transition plans per segment (may be None)
    transitions = []  # per-segment transition dicts (to next clip)
    
    # Use the last segment's end_time as the target duration
    total_json_duration = float(segments[-1].end_time) if segments else 0.0

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
            # Effect selection: LLM plan if enabled, otherwise heuristics
            plan = None
            if USE_LLM_EFFECTS:
                plan = get_effect_plan(seg, i, len(segments))
                if plan:
                    enhanced = effects_agent.apply_plan(clip, plan)
                else:
                    enhanced = effects_agent.apply_context_effect(clip, seg, i, len(segments))
            else:
                enhanced = effects_agent.apply_context_effect(clip, seg, i, len(segments))
            plans.append(plan)

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
        final_video = concatenate_videoclips(processed_clips, method="compose")
        
        logger.info(f"[video_generator] Timeline built: duration={final_video.duration:.3f}s, target={total_json_duration:.3f}s")
    else:
        # No transitions: simple concatenation
        logger.info(f"[video_generator] No crossfades, using simple concatenation")
        final_video = concatenate_videoclips(video_clips, method="compose")

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
        "threads": 4,
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

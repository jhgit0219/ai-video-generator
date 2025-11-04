# Renderer Module

The renderer is responsible for compositing video clips from selected images, applying cinematic effects, and exporting the final video.

## Key Files

- `video_generator.py`: Main compositor with `CinematicEffectsAgent` class
- `effects_director.py`: LLM-driven effects planning (motion, overlays, fades)
- `subject_detection.py`: YOLO+CLIP subject detection and face anchor finding
- `image_enhancer.py`: Real-ESRGAN upscaling and sharpening
- `effects/`: Modular effects package (see `effects/CLAUDE.md`)
- `effects_tools.py`: Compatibility shim for legacy imports

## CinematicEffectsAgent

Main class for applying effects. Key methods:

- `apply_plan(clip, plan)`: Apply LLM-generated motion/overlay/fade plan
- `_pan_horizontal(clip, direction)`: Transform-based horizontal pan (no black bars)
- `_pan_vertical(clip, direction)`: Transform-based vertical pan (no black bars)
- `_ken_burns_zoom_in/out(clip)`: Smooth zoom effects
- `_apply_vignette(clip)`: Vignette overlay
- `_apply_film_grain(clip)`: Vintage grain texture

## Pan Effects Pattern (CRITICAL)

**MUST use uniform scaling to avoid aspect ratio distortion:**

```python
def _pan_horizontal(self, clip, direction):
    w, h = clip.size
    scale = 1.15  # Create 15% pan room
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))  # UNIFORM SCALE
    max_x = max(0, new_w - w)

    def transform_frame(get_frame, t):
        progress = smooth_ease(t / duration)
        frame = get_frame(t)

        # Uniform upscale
        img = Image.fromarray(frame).resize((new_w, new_h), LANCZOS)

        # Horizontal pan within available room
        x1 = int(round(max_x * progress)) if ltr else int(round(max_x * (1 - progress)))
        x1 = max(0, min(max_x, x1))
        x2 = x1 + w

        # Vertical crop centered
        y1 = (new_h - h) // 2
        y2 = y1 + h

        return np.array(img)[y1:y2, x1:x2]

    return clip.transform(transform_frame)
```

**Why this matters:**

- Non-uniform scaling (e.g., `resize((w*1.15, h))`) stretches content → horizontal squish
- Using `with_position()` on resized clips creates black bars when composited
- Transform-based approach with uniform scaling preserves aspect ratio and fills frame

## Subject Detection

See `subject_detection.py` for implementation. Flow:

1. `detect_subject_bbox(frame, target, prefer, nth)` → normalized (x, y, w, h)

   - YOLO segments frame
   - CLIP reranks to pick matching subject (e.g., "girl")
   - Returns normalized bbox

2. `detect_subject_shape(frame, ...)` → {bbox, mask}

   - Same as above but includes segmentation mask

3. `detect_anchor_feature(frame, subject_mask, subject_bbox, feature_type)` → (x, y) pixels
   - Crops to subject region
   - Runs YOLO again to detect face (or other anchor)
   - Returns absolute pixel coordinates for precise framing

**Models:**

- YOLO: `weights/yolo11x-seg.pt` (segmentation task)
- CLIP: `openai/clip-vit-base-patch32` (semantic reranking)

## Image Enhancement

`image_enhancer.py` handles upscaling and sharpening:

- **Real-ESRGAN**: AI upscaling (2x or 4x) when `ENABLE_AI_UPSCALE=True`
- **PIL sharpening**: Simple sharpening when AI upscale disabled
- **Caching**: Enhanced images cached in `data/temp_images/enhanced_cache/` to avoid reprocessing

## Effects Director (LLM)

`effects_director.py` uses Ollama to propose per-segment effects plans:

```json
{
  "motion": "pan_up",
  "overlays": ["vignette", "film_grain"],
  "fade_in": 0.5,
  "fade_out": 0.0,
  "transition": { "type": "crossfade", "duration": 1.0 }
}
```

Supports:

- Motion: pan_up, pan_down, pan_ltr, pan_rtl, ken_burns_in, ken_burns_out, zoom_pan, static
- Overlays: vignette, film_grain, neon_overlay
- Fades: fade_in/out duration in seconds
- Transitions: crossfade or none

**When to use:**
Enable with `USE_LLM_EFFECTS=True` for automatic creative decisions based on segment context (reasoning, visual_description, topic).

## Debugging Rendering Issues

**Black bars:**

- Caused by `with_position()` on resized clips or incorrect cropping
- Solution: Use transform-based panning with uniform scaling

**Aspect ratio distortion (squish/stretch):**

- Caused by non-uniform scaling (e.g., `resize((w*1.15, h))`)
- Solution: Always scale both dimensions by same factor, then crop

**Effects cropped away:**

- Caused by applying effects before final scaling
- Solution: Follow stage ordering (see root CLAUDE.md) – apply late overlays after `zoom_to_cover`

**Static animations:**

- Caused by `resized()` or `cropped()` wrappers flattening time
- Solution: Use `transform()` for time-aware effects

**Logs to check:**

- `[effects] <effect_name>`: Effect application logs
- `[subject_detection]`: Detection and anchor finding logs
- `clip.size`: Track frame dimensions to catch unexpected resizing

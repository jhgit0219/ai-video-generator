# Effects Package

Modular effects library for video transformations. Each effect is a pure function: `VideoClip → VideoClip`.

## Registry

`registry.py` exports `TOOLS_REGISTRY` – a dict mapping effect names to functions:

```python
TOOLS_REGISTRY = {
    "zoom_on_subject": apply_variable_zoom_on_subject,
    "temporal_zoom": apply_temporal_zoom,
    "paneled_text": apply_paneled_text,
    "neon_overlay": apply_neon_overlay,
    "subject_outline": apply_subject_outline,
    "subject_pop": apply_subject_pop,
    "nightvision": apply_nightvision,
    "zoom_then_panel": apply_zoom_then_panel,
}
```

## Effect Modules

### Zoom Effects

**`zoom_variable.py` – Geometric zoom on subject**

- Face-anchored crop/zoom that reframes composition
- Detects subject via YOLO+CLIP, finds face anchor, zooms to anchor
- `anim_duration`: Zoom completes by this time, then holds
- Uses transform-based approach to preserve time animation

**`zoom_temporal.py` – Digital zoom overlay**

- Time-animated zoom that doesn't change frame composition
- Can be face-anchored via `anchor_point` parameter
- `method="transform"` (default) or `"sequence"` for explicit frame building

**`zoom_then_panel.py` – Combined effect**

- Zoom first N seconds, then show panel with text
- Convenience wrapper combining zoom_on_subject + paneled_text

### Text Overlays

**`overlay_text.py` – Paneled text with typewriter**

- Large semi-transparent panel with animated text
- `cps`: Characters per second for typewriter effect
- `start_time`/`duration`: When panel appears and how long it stays
- `animate_in`: Slide-in duration
- **MUST apply after final scaling** to prevent cropping

### Subject Effects

**`subject_outline.py` – Expanded ring outline**

- Morphology-based outline with configurable offset (gap from subject)
- `offset`: Distance in pixels between subject edge and outline inner edge
- `thickness`: Width of the outline ring
- `glow_radius`: Blur amount for glow effect
- `pulse`: Animate opacity over time

**`subject_pop.py` – Silhouette pop**

- Extracts subject silhouette and scales it with face-anchor centering
- `scale`: Pop scale factor (e.g., 1.2)
- `pop_duration`: How long the pop animation takes
- `shadow_opacity`: Drop shadow intensity

### Visual Overlays

**`overlay_neon.py` – Neon color overlay**

- Blend color overlay with configurable blend mode
- `color`: RGB tuple (e.g., [80, 220, 255])
- `opacity`: Overlay transparency (0-1)
- `blend_mode`: "screen", "multiply", "overlay", "add"
- `animate_in`: Fade-in duration

**`overlay_nightvision.py` – Night vision effect**

- Green tint, scanlines, vignette, noise
- Used for "tactical" or "surveillance" aesthetic

## Utility Modules

**`easing.py`**

- Easing functions: `ease_in_out_cubic`, `ease_out_back`, `ease_curve`
- Use for smooth animations

**`coords.py`**

- Coordinate utilities: `denorm_bbox`, `bbox_center_px`
- Convert normalized (0-1) coords to pixel coords

**`subject_import.py`**

- Dynamic import of subject detection functions
- Avoids circular dependencies

## Effect Timing Patterns

**Zoom with timed completion:**

```python
apply_variable_zoom_on_subject(
    clip,
    target="girl",
    target_scale=1.55,
    anim_duration=3.0  # Complete zoom by 3s, hold after
)
```

**Typewriter text panel:**

```python
apply_paneled_text(
    clip,
    text="Herodotus",
    cps=10,           # 10 chars/sec
    start_time=3.0,   # Appear at 3s
    duration=2.5      # Stay until 5.5s
)
```

**Face-anchored temporal zoom:**

```python
apply_temporal_zoom(
    clip,
    scale=1.3,
    anchor_point=(0.486, 0.159)  # Normalized face coords
)
```

## Common Patterns

### Transform-Based Effect (Time-Aware)

```python
def my_effect(clip: VideoClip, param: float) -> VideoClip:
    w, h = clip.size
    dur = clip.duration or 1.0

    def transform_frame(get_frame, t):
        frame = get_frame(t)
        progress = t / dur
        # Apply time-varying transformation
        transformed = do_something(frame, progress, param)
        return transformed

    return clip.transform(transform_frame)
```

### Composite Overlay Effect

```python
def my_overlay(clip: VideoClip, opacity: float) -> VideoClip:
    w, h = clip.size

    def make_overlay_frame(t):
        # Create overlay frame (PIL or numpy)
        overlay = create_overlay(w, h, t)
        return np.array(overlay)

    overlay_clip = VideoClip(make_overlay_frame, duration=clip.duration)
    overlay_clip = overlay_clip.with_opacity(opacity)

    return CompositeVideoClip([clip, overlay_clip])
```

## MoviePy Pitfalls to Avoid

❌ **Don't use `resize()` for time-varying effects:**

```python
# BAD - flattens time animation
clip.resize(lambda t: 1.0 + 0.2*t)
```

✅ **Use `transform()` instead:**

```python
# GOOD - preserves time animation
def tf(get_frame, t):
    frame = get_frame(t)
    scale = 1.0 + 0.2*t
    # ... resize frame based on t
    return resized_frame
clip.transform(tf)
```

❌ **Don't use non-uniform scaling:**

```python
# BAD - distorts aspect ratio
img.resize((w*1.15, h))
```

✅ **Use uniform scaling:**

```python
# GOOD - preserves aspect ratio
scale = 1.15
img.resize((int(w*scale), int(h*scale)))
```

## Debugging Effects

**Log frame transformations:**

```python
logger.debug(f"[effects] {effect_name} t={t:.3f}s frame_shape={frame.shape}")
```

**Track clip size:**

```python
logger.debug(f"[effects] {effect_name} input_size={clip.size}")
# ... apply effect
logger.debug(f"[effects] {effect_name} output_size={result.size}")
```

**Isolate effect:**
Comment out other effects in tools JSON to test one at a time.

**Visual inspection:**
Always render and watch output – logs don't show visual artifacts.

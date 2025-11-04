# Effects Ordering Reference - Critical for Correct Rendering

This document defines the **mandatory order** for applying effects to video clips. Incorrect ordering causes visual artifacts like static overlays during pans, aspect ratio distortion, or cropped effects.

---

## 🎯 Core Principle

**Effects that modify the base image must be applied BEFORE camera motion, and motion must be applied BEFORE final scaling.**

```
BASE IMAGE → BAKED EFFECTS → CAMERA MOTION → FINAL SCALING → LATE OVERLAYS
```

---

## 📋 The Canonical Order

### **Stage 1: Subject Detection & Metadata** (Pre-render)
Run once before any visual processing.

- `detect_subject_bbox()` - YOLO + CLIP to find subject
- `detect_anchor_feature()` - Face/eye detection for precise framing
- Store bbox/anchor for later use

**Purpose**: Gather metadata needed for zoom_on_subject and subject_outline.

---

### **Stage 2: Baked Image Effects** (Permanent modifications)
These effects **paint directly onto the base image** and move with it during camera motion.

**Order within Stage 2:**

1. **`subject_outline`** - Draw outline around detected subject
   - Uses YOLO segmentation mask
   - Permanently paints outline onto image
   - **MUST run before any motion/scaling**

2. **`neon_overlay`** - Neon glow edge detection
   - Detects edges and adds colored glow
   - Baked into image pixels

3. **`film_grain`** - Vintage texture overlay
   - Adds noise pattern to simulate film grain
   - Applied uniformly across image

**Why this order?**
- Outline/neon need clean subject detection (before noise)
- All must be baked before motion so they move with the image
- Film grain goes last to texture everything including outlines

---

### **Stage 3: Geometric Reframing** (Composition changes)
These effects change the framing but preserve aspect ratio.

**Order within Stage 3:**

1. **`zoom_on_subject`** - Face-anchored crop and zoom
   - Uses anchor from Stage 1
   - Smoothly zooms/crops to subject
   - Creates "reframed" composition
   - Uses `transform()` for time-aware animation

**Why after Stage 2?**
- Subject outline is already baked, so it zooms with the subject
- Geometric reframing preserves the painted effects

---

### **Stage 4: Camera Motion** (Virtual camera movement)
These effects simulate camera panning/zooming within the reframed image.

**Order within Stage 4:**

1. **`pan_up` / `pan_down`** - Vertical camera pan
   - Uniformly upscales image by 1.15x (both width and height)
   - Pans vertically within available room
   - Uses `transform()` for time-aware animation

2. **`pan_ltr` / `pan_rtl`** - Horizontal camera pan
   - Uniformly upscales image by 1.15x
   - Pans horizontally within available room
   - Uses `transform()` for time-aware animation

3. **`ken_burns_in` / `ken_burns_out`** - Slow zoom effect
   - Smooth zoom in or out over clip duration
   - Uses `transform()` for time-aware animation

4. **`zoom_pan`** - Combined zoom and pan
   - Zooms while panning
   - Uses `transform()` for time-aware animation

**Why after Stage 3?**
- Camera motion needs a stable, reframed composition
- Motion is virtual (within the upscaled canvas)
- Subject outline moves with the panning image

**Critical:** All motion effects **MUST use uniform scaling** (same factor for width and height) to avoid aspect ratio distortion.

---

### **Stage 5: Final Scaling** (Fit to render size)
These effects ensure the clip fits the target resolution.

**Order within Stage 5:**

1. **`zoom_to_cover`** - Aspect-ratio-preserving resize + crop
   - Resizes image to cover RENDER_SIZE (1920x1080)
   - Crops excess to exactly fit
   - Preserves aspect ratio (no squish/stretch)
   - **MUST be last resize operation**

**Why after Stage 4?**
- Motion effects create oversized canvas (1.15x upscale)
- zoom_to_cover brings it back to exact render dimensions
- Running too early would break motion effects

---

### **Stage 6: Late Overlays** (Frame-locked effects)
These effects are applied **after scaling** and remain fixed to the frame (don't move during pans).

**Order within Stage 6:**

1. **`vignette`** - Dark corner fade
   - Frame-locked overlay
   - Doesn't move with image
   - Applied after scaling to prevent cropping

2. **`temporal_zoom`** - Dynamic zoom overlay
   - Time-varying zoom effect
   - Applied as late overlay

3. **`paneled_text`** - Subtitle panels with typewriter effect
   - **MUST be absolute last**
   - Positioned at fixed screen location
   - If applied before zoom_to_cover, text gets cropped

**Why after Stage 5?**
- These effects are intentionally frame-locked
- Vignette should cover the final frame edges
- Text panels must not be cropped by zoom_to_cover

---

## ⚠️ Common Ordering Mistakes

### ❌ **Mistake 1: Applying subject_outline after camera motion**
```python
clip = pan_horizontal(clip, "left-to-right")  # Motion first
clip = apply_subject_outline(clip, ...)        # Outline after ❌
```
**Result**: Outline stays static while image pans underneath it.

**✅ Fix:**
```python
clip = apply_subject_outline(clip, ...)        # Outline first ✅
clip = pan_horizontal(clip, "left-to-right")  # Motion second
```

---

### ❌ **Mistake 2: Applying paneled_text before zoom_to_cover**
```python
clip = add_paneled_text(clip, ...)  # Text first
clip = zoom_to_cover(clip)          # Scaling second ❌
```
**Result**: Text panels get cropped off-screen.

**✅ Fix:**
```python
clip = zoom_to_cover(clip)          # Scaling first ✅
clip = add_paneled_text(clip, ...)  # Text second (absolute last)
```

---

### ❌ **Mistake 3: Non-uniform scaling during pans**
```python
# Bad: stretches aspect ratio
w, h = clip.size
new_w = int(w * 1.15)  # Only width scaled ❌
img.resize((new_w, h))
```
**Result**: Horizontal squishing during pan.

**✅ Fix:**
```python
# Good: preserves aspect ratio
w, h = clip.size
new_w = int(w * 1.15)
new_h = int(h * 1.15)  # Both scaled equally ✅
img.resize((new_w, new_h))
```

---

## 📊 Effect Classification Table

| Effect | Stage | Moves with Image? | Applied to | Uses transform()? |
|--------|-------|-------------------|------------|-------------------|
| `subject_outline` | 2 (Baked) | ✅ Yes | Base image | No (pixel painting) |
| `neon_overlay` | 2 (Baked) | ✅ Yes | Base image | No (pixel painting) |
| `film_grain` | 2 (Baked) | ✅ Yes | Base image | No (pixel painting) |
| `zoom_on_subject` | 3 (Geometric) | N/A | Composition | ✅ Yes |
| `pan_up/down` | 4 (Motion) | N/A | Virtual camera | ✅ Yes |
| `pan_ltr/rtl` | 4 (Motion) | N/A | Virtual camera | ✅ Yes |
| `ken_burns_in/out` | 4 (Motion) | N/A | Virtual camera | ✅ Yes |
| `zoom_pan` | 4 (Motion) | N/A | Virtual camera | ✅ Yes |
| `zoom_to_cover` | 5 (Scaling) | N/A | Final resize | No (one-time resize) |
| `vignette` | 6 (Late) | ❌ No | Frame overlay | No (frame-locked) |
| `temporal_zoom` | 6 (Late) | ❌ No | Frame overlay | ✅ Yes |
| `paneled_text` | 6 (Late) | ❌ No | Frame overlay | ✅ Yes (typewriter) |

---

## 🔧 Implementation in video_generator.py

The current implementation in `CinematicEffectsAgent.apply_plan()` should follow this flow:

```python
def apply_plan(self, clip, plan, segment=None):
    # Stage 1: Detection (already done during precompute)
    # bbox/anchor stored in segment metadata

    # Stage 2: Baked Image Effects
    overlays = plan.get("overlays", [])
    if "subject_outline" in overlays:
        clip = self._apply_subject_outline(clip, segment)
    if "neon_overlay" in overlays:
        clip = self._apply_neon_overlay(clip)
    if "film_grain" in overlays:
        clip = self._apply_film_grain(clip)

    # Stage 3: Geometric Reframing
    tools = plan.get("tools", [])
    for tool in tools:
        if tool["name"] == "zoom_on_subject":
            clip = self._apply_zoom_on_subject(clip, tool["params"], segment)

    # Stage 4: Camera Motion
    motion = plan.get("motion")
    if motion == "pan_up":
        clip = self._pan_vertical(clip, "up")
    elif motion == "pan_down":
        clip = self._pan_vertical(clip, "down")
    elif motion == "pan_ltr":
        clip = self._pan_horizontal(clip, "left-to-right")
    elif motion == "pan_rtl":
        clip = self._pan_horizontal(clip, "right-to-left")
    elif motion == "ken_burns_in":
        clip = self._ken_burns_zoom_in(clip)
    elif motion == "ken_burns_out":
        clip = self._ken_burns_zoom_out(clip)
    elif motion == "zoom_pan":
        clip = self._zoom_pan(clip)

    # Stage 5: Final Scaling
    clip = self._zoom_to_cover(clip)

    # Stage 6: Late Overlays (frame-locked)
    if "vignette" in overlays:
        clip = self._apply_vignette(clip)
    if "temporal_zoom" in overlays:
        clip = self._apply_temporal_zoom(clip)

    # Fades (can be applied anytime, but typically last)
    fade_in = plan.get("fade_in", 0)
    fade_out = plan.get("fade_out", 0)
    if fade_in > 0:
        clip = clip.fadein(fade_in)
    if fade_out > 0:
        clip = clip.fadeout(fade_out)

    # Stage 6 (absolute last): Paneled Text
    # Applied in main loop after apply_plan() returns

    return clip
```

---

## 🎬 For LLM Effects Director

When planning effects, remember:

1. **Subject outline** is a "baked" effect - it will move with the image during pans
2. **Vignette** is a "late" effect - it stays locked to frame corners
3. **Pan motion + subject outline = correct** (outline moves with pan)
4. **Pan motion + vignette = correct** (vignette stays at frame edges)
5. Never suggest applying paneled_text via overlays (it's handled separately)

---

## 🔍 Debugging Checklist

If you see visual artifacts, check this order:

1. ✅ Is `subject_outline` in Stage 2 (before motion)?
2. ✅ Is `zoom_to_cover` in Stage 5 (after motion, before late overlays)?
3. ✅ Is `paneled_text` absolute last (after zoom_to_cover)?
4. ✅ Do all motion effects use uniform scaling (w*scale, h*scale)?
5. ✅ Do all time-based effects use `transform()` instead of `resize()`?

---

**Last Updated**: 2025-11-04
**Status**: Reference document for effects ordering

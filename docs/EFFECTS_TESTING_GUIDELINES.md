# Effects Testing Guidelines

**Purpose**: Ensure effect tests accurately emulate the production pipeline to catch integration issues early.

## Core Principle

**Effect tests must replicate how frames enter the effects application stage in production.**

## Pipeline Frame Preparation Flow

### 1. Load Image
```python
img = Image.open(image_path).convert("RGB")
```

### 2. Enhancement (Optional - based on config)
```python
if ENABLE_AI_UPSCALE or ENABLE_SIMPLE_SHARPEN:
    # Apply Real-ESRGAN or PIL sharpening
    img = enhance_image(img)
```

### 3. Create Clip (at original or enhanced size)
```python
# Create clip from numpy array
img_array = np.array(img)
clip = ImageClip(img_array, duration=seg_dur)

# Note: Clip keeps original/enhanced dimensions at this stage
```

### 4. Apply Effects
```python
# Effects operate on clip at original/enhanced size
# This allows content-aware effects to access full resolution for detection
clip = apply_some_effect(clip, **params)
```

### 5. Final Resize to RENDER_SIZE (zoom-to-cover, no black bars)
```python
# CRITICAL: This happens AFTER effects application
if clip.size != (RENDER_SIZE[0], RENDER_SIZE[1]):
    curr_w, curr_h = clip.size
    target_w, target_h = RENDER_SIZE

    # Calculate scale to cover both dimensions (zoom-to-cover)
    scale_w = target_w / curr_w
    scale_h = target_h / curr_h
    scale = max(scale_w, scale_h)  # Use larger scale

    # Resize maintaining aspect ratio
    new_w = int(curr_w * scale)
    new_h = int(curr_h * scale)
    clip = clip.resized(width=new_w, height=new_h)

    # Crop to exact RENDER_SIZE from center
    if clip.size != (target_w, target_h):
        x1 = (new_w - target_w) // 2
        y1 = (new_h - target_h) // 2
        x2 = x1 + target_w
        y2 = y1 + target_h
        clip = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
```

### 6. Export
```python
clip.write_videofile(output_path, fps=FPS, codec=VIDEO_CODEC, audio=False)
```

## Test Script Template

```python
import sys
from pathlib import Path
from moviepy import ImageClip
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from pipeline.renderer.effects.some_effect import apply_some_effect

def zoom_to_cover(clip, target_size):
    """Apply zoom-to-cover resize to match RENDER_SIZE (production pattern).

    This replicates video_generator.py:1066-1089.
    """
    if clip.size == target_size:
        return clip

    curr_w, curr_h = clip.size
    target_w, target_h = target_size

    # Calculate scale to cover both dimensions
    scale_w = target_w / curr_w
    scale_h = target_h / curr_h
    scale = max(scale_w, scale_h)

    # Resize maintaining aspect ratio
    new_w = int(curr_w * scale)
    new_h = int(curr_h * scale)
    clip = clip.resized(width=new_w, height=new_h)

    # Crop to exact target size from center
    if clip.size != target_size:
        x1 = (new_w - target_w) // 2
        y1 = (new_h - target_h) // 2
        x2 = x1 + target_w
        y2 = y1 + target_h
        clip = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)

    return clip

def test_my_effect():
    """Test my_effect with production-like setup."""
    # Load test image at original size
    img = Image.open("data/test_webp_input/test.webp").convert("RGB")
    img_array = np.array(img)
    clip = ImageClip(img_array, duration=6.0)

    # Apply effect (at original size)
    result = apply_some_effect(clip, param1="value")

    # Final resize to RENDER_SIZE (zoom-to-cover)
    result = zoom_to_cover(result, config.RENDER_SIZE)

    # Export
    result.write_videofile(
        "data/output/test_results/my_effect.mp4",
        fps=config.FPS,
        codec="libx264",  # CPU codec for test scripts
        audio=False,
        preset="ultrafast"
    )

    print(f"✓ Test complete: {result.size} at {result.duration}s")
```

## Common Mistakes to Avoid

### ❌ Wrong: Pre-resize to RENDER_SIZE before effects
```python
# This breaks content-aware effects that need full resolution for detection
img = Image.open("path/to/image.jpg")
img_resized, _ = resize_and_crop(img, config.RENDER_SIZE[0], config.RENDER_SIZE[1])
frame = np.array(img_resized)
clip = ImageClip(frame, duration=6.0)
clip = apply_effect(clip)  # Effect gets pre-cropped input!
```

### ❌ Wrong: Skip final resize to RENDER_SIZE
```python
img = Image.open("path/to/image.jpg")
frame = np.array(img)
clip = ImageClip(frame, duration=6.0)
clip = apply_effect(clip)
# Missing zoom_to_cover step - output will be wrong size!
```

### ✅ Correct: Apply effect first, then resize to RENDER_SIZE
```python
# Load at original size
img = Image.open("path/to/image.jpg")
frame = np.array(img)
clip = ImageClip(frame, duration=6.0)

# Apply effect at original size
clip = apply_effect(clip)

# Final resize to RENDER_SIZE (zoom-to-cover)
clip = zoom_to_cover(clip, config.RENDER_SIZE)
```

## Codec Selection for Tests

**Use CPU codecs in test scripts to avoid CUDA multiprocessing issues:**

```python
# Test scripts (always works)
clip.write_videofile(path, codec="libx264", preset="ultrafast")

# Production (can use GPU if available)
clip.write_videofile(path, codec=config.VIDEO_CODEC)  # May be "h264_nvenc"
```

## Effect-Specific Considerations

### Subject-Based Effects
Effects that use YOLO detection (`subject_glow`, `subject_pop`, `character_highlight`):

```python
# Ensure test image contains detectable subject
# Test with images that have:
# - Clear subject (person, object)
# - Good lighting
# - Subject not too small

# Effects should work without bbox/mask params (auto-detect)
result = apply_subject_glow(clip)  # Detects automatically

# Or provide precomputed bbox/mask for consistent testing
result = apply_subject_glow(clip, bbox=(0.2, 0.1, 0.6, 0.8))
```

### Overlay Effects
Effects that add overlays (`slime_splatter`, `news_overlay`, `newspaper_frame`):

```python
# Verify overlay visibility at RENDER_SIZE
# - Text should be readable
# - Shapes should be visible
# - Positions should be within frame bounds

# Debug by checking overlay dimensions
logger.debug(f"Overlay size: {overlay.size} vs clip size: {clip.size}")
```

### Transition Effects
Effects that transition between clips (`branded_transition`):

```python
# Test with two clips to show actual transition
clip1 = load_test_clip(img1_path, duration=3.0)
clip2 = load_test_clip(img2_path, duration=3.0)

# Apply transition
transitioned = apply_branded_transition(clip1, transition_type="purple_fade")

# Concatenate to show effect
from moviepy import concatenate_videoclips
final = concatenate_videoclips([transitioned, clip2])
```

## Visual Validation Checklist

After running tests, always verify:

- [ ] Output resolution matches `RENDER_SIZE` (check video properties)
- [ ] No black bars or letterboxing
- [ ] No aspect ratio distortion (squish/stretch)
- [ ] Effects are visible and positioned correctly
- [ ] Text is readable (if applicable)
- [ ] Animations complete within expected duration
- [ ] No visual artifacts or glitches

## Reference

- **Pipeline implementation**: `pipeline/renderer/video_generator.py:935-958`
- **Image utilities**: `pipeline/renderer/image_utils.py:resize_and_crop()`
- **Config constants**: `config.py` (`RENDER_SIZE`, `FPS`, `VIDEO_CODEC`)

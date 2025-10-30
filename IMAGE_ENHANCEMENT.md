# Image Enhancement - AI Upscaling & Sharpening

This document explains how to enable AI-powered image enhancement for better video quality.

## Features

### 1. Simple PIL Sharpening (Default: Enabled)

- **No installation required** - uses built-in PIL library
- Fast processing with minimal overhead
- Good for slightly soft images
- Configure with `ENABLE_SIMPLE_SHARPEN = True` in `config.py`
- Adjust strength with `SHARPEN_STRENGTH` (1.0-2.0)

### 2. AI Upscaling with Real-ESRGAN (Optional)

- **Best quality** - Uses deep learning to enhance resolution
- Can upscale 2x or 4x
- Particularly good for low-resolution scraped images
- Requires additional installation (see below)

## Quick Start

### Enable Simple Sharpening (Already Enabled)

In `config.py`:

```python
ENABLE_SIMPLE_SHARPEN = True   # Enable PIL sharpening
SHARPEN_STRENGTH = 1.5          # 1.0 = normal, 1.5 = moderate, 2.0 = strong
```

### Enable AI Upscaling (Optional - Best Quality)

#### Step 1: Install Dependencies

```powershell
# Install Real-ESRGAN and dependencies
pip install realesrgan
pip install basicsr
pip install facexlib
pip install gfpgan
```

#### Step 2: Download Model Weights

Create a `weights` folder in your project root:

```powershell
mkdir weights
cd weights
```

Download the appropriate model:

**For 2x upscaling (Faster, 17MB)**

```powershell
# Download RealESRGAN_x2plus.pth
# From: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

**For 4x upscaling (Best Quality, 67MB)**

```powershell
# Download RealESRGAN_x4plus.pth
# From: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

**PowerShell download commands:**

```powershell
# For 2x model
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" -OutFile "RealESRGAN_x2plus.pth"

# For 4x model
Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -OutFile "RealESRGAN_x4plus.pth"
```

#### Step 3: Enable in Config

In `config.py`:

```python
# AI Image Enhancement
ENABLE_AI_UPSCALE = True           # Enable Real-ESRGAN
UPSCALE_FACTOR = 2                 # 2x or 4x
UPSCALE_MODEL = "RealESRGAN_x2plus"  # or "RealESRGAN_x4plus"
ENABLE_SIMPLE_SHARPEN = False      # Disable simple sharpen (AI handles it)
```

## Configuration Options

### In `config.py`:

```python
# AI Image Enhancement (Upscaling & Sharpening)
ENABLE_AI_UPSCALE = False     # Enable Real-ESRGAN AI upscaling
UPSCALE_FACTOR = 2            # 2x or 4x upscaling
UPSCALE_MODEL = "RealESRGAN_x2plus"  # Model name
ENABLE_SIMPLE_SHARPEN = True  # Fast PIL sharpening (when AI disabled)
SHARPEN_STRENGTH = 1.5        # 1.0-2.0 (higher = sharper)
```

## Performance Comparison

| Method             | Speed     | Quality   | Installation   |
| ------------------ | --------- | --------- | -------------- |
| **Simple Sharpen** | Very Fast | Good      | None needed    |
| **Real-ESRGAN 2x** | Moderate  | Excellent | Requires setup |
| **Real-ESRGAN 4x** | Slow      | Best      | Requires setup |

## Recommendations

### For Quick Setup (Default)

- Use `ENABLE_SIMPLE_SHARPEN = True`
- Set `SHARPEN_STRENGTH = 1.5`
- No additional installation needed
- Works well for slightly soft images

### For Best Quality

- Install Real-ESRGAN (see Step 1-3 above)
- Use `ENABLE_AI_UPSCALE = True`
- Start with 2x upscaling for speed
- Use 4x for maximum quality on low-res images

### For Maximum Speed

- Set both to `False` to disable enhancement
- Images will be used as-is

## Troubleshooting

### "Import realesrgan could not be resolved"

This is expected if Real-ESRGAN is not installed. The code handles this gracefully - it will skip AI upscaling and use simple sharpening instead.

To fix: Follow installation steps above.

### "Model not found at weights/RealESRGAN_x2plus.pth"

Download the model weights (see Step 2 above).

### "CUDA out of memory"

Reduce the tile size in `image_enhancer.py`:

```python
tile=200,  # Reduce from 400 if GPU memory is limited
```

Or use CPU:

```python
half=False,  # Keep this False for CPU mode
```

### Slow processing

- Use 2x instead of 4x upscaling
- Or disable AI upscaling and use simple sharpening
- AI upscaling processes each image once, then caches

## Technical Details

### Image Enhancement Pipeline

1. **Load Image** - Opens selected image
2. **AI Upscale** (if enabled) - Real-ESRGAN enhances resolution
3. **Simple Sharpen** (if AI disabled) - PIL UnsharpMask filter
4. **Resize to Target** - Final resize to 1920x1080
5. **MoviePy Integration** - Converted to video clip

### When to Use Each Method

**Simple Sharpening:**

- Images are already high resolution
- Speed is critical
- Slight softness correction needed

**AI Upscaling:**

- Scraped images are low quality
- Want maximum visual quality
- Have GPU available for faster processing
- Processing time is acceptable

## Example Results

### Before Enhancement

- Slightly blurry scraped images
- Soft edges and details
- 720p or lower resolution sources

### After Simple Sharpen

- Crisper edges
- Better detail visibility
- No resolution increase

### After AI Upscale 2x

- Significantly improved detail
- Natural-looking enhancement
- Better for 1080p output

### After AI Upscale 4x

- Maximum detail recovery
- Best for very low-res sources
- Slower but highest quality

## Integration with Video Pipeline

Enhancement happens during video generation:

1. Parser reads JSON script
2. Scraper collects images
3. CLIP ranker selects best images (with sharpness scoring)
4. **Enhancement applied here** (upscale/sharpen)
5. Effects director applies motion
6. Video renderer exports final video

The enhanced images are generated on-the-fly during rendering and not saved to disk (to save space).

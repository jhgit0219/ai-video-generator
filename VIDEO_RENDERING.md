# Video Rendering Setup

## Install MoviePy and Dependencies

```powershell
# Install moviepy (includes imageio, numpy)
pip install moviepy

# Install Pillow if not already installed
pip install Pillow

# Verify installation
python test_video_rendering.py
```

## Cinematic Effects Available

The video generator includes 6 automated cinematic effects:

1. **Ken Burns Zoom In** - Slow zoom into the image with slight pan
2. **Ken Burns Zoom Out** - Slow zoom out from the image
3. **Pan Left to Right** - Horizontal camera movement across image
4. **Pan Right to Left** - Reverse horizontal pan
5. **Vignette Overlay** - Darkened corners for cinematic look
6. **Old Film Grain** - Vintage film texture with sepia tint

Each segment automatically gets a random effect applied.

## How It Works

1. **Load Image**: Each segment's `selected_image` is loaded
2. **Create Clip**: Image is converted to video clip with `duration` from JSON
3. **Apply Effect**: Random cinematic effect applied via `CinematicEffectsAgent`
4. **Concatenate**: All clips are joined in sequence
5. **Add Audio**: Original audio file is synchronized
6. **Export**: Final video saved to `data/output/final_video.mp4`

## Run Full Pipeline

```powershell
# Make sure SKIP_FLAG = True in config.py
python main.py
```

## Expected Output

```
[video_generator] Starting final video rendering with cinematic effects
[video_generator] Processing segment 0...
   Image: data/temp_images/segment_0/image_1.jpg
   Duration: 12.5s
   Applying effect: ken_burns_zoom_in
   [OK] Segment 0 processed with ken_burns_zoom_in
[video_generator] Processing segment 1...
   ...
[video_generator] Concatenating 3 clips...
[video_generator] Adding audio: data/input/audio.mp3
[video_generator] Exporting to: data/output/final_video.mp4
Moviepy - Building video...
[video_generator] Video rendering complete!
```

## Troubleshooting

### MoviePy Import Error

```powershell
pip install --upgrade moviepy imageio imageio-ffmpeg
```

### FFmpeg Not Found

MoviePy should install `imageio-ffmpeg` automatically, but if issues persist:

```powershell
# Windows (via Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### Memory Issues

If rendering large videos, reduce resolution in `config.py`:

```python
IMAGE_SIZE = (1280, 720)  # Use 720p instead of 1080p
```

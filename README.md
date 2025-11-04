# AI Video Generator

An AI-powered video generation pipeline that transforms text scripts into cinematic videos with automated visual effects, subject detection, and intelligent image selection.

## 🎯 Overview

This tool automates professional video creation by:

1. **Intelligent Image Selection** – YOLO segmentation + CLIP semantic ranking finds the perfect visuals
2. **Cinematic Effects** – Face-anchored zooms, temporal animations, stylized overlays, and text panels
3. **LLM-Driven Direction** – Ollama suggests camera movements and effects based on narrative context
4. **Automated Composition** – Transforms portrait images to landscape, applies transitions, syncs with audio
5. **Production-Ready Output** – Hardware-accelerated encoding (NVIDIA/AMD/Intel) for fast 1080p exports

## ✨ Key Features

### AI-Powered Visual Intelligence

- **YOLO + CLIP Detection**: Precisely identifies and segments subjects in images
- **Face-Anchored Framing**: Automatically centers compositions on detected faces
- **Semantic Ranking**: CLIP ensures images semantically match your narrative
- **Smart Requerying**: Automatically refines search when initial results don't match

### Cinematic Effects Suite

- **Face-Anchored Zoom**: Smooth geometric zoom that keeps subjects centered
- **Temporal Zoom**: Digital zoom overlays with precise timing control
- **Typewriter Text Panels**: Large, styled panels with letter-by-letter animation
- **Subject Outline**: Expanded ring outlines with glow and pulse effects
- **Neon Overlays**: Color washes with blend modes (screen, multiply, overlay)
- **Camera Motion**: Pan (up/down/left/right), Ken Burns zoom, combined zoom-pan

### Production Features

- **Hardware Encoding**: NVIDIA (h264_nvenc), AMD (h264_amf), Intel (h264_qsv) support
- **AI Upscaling**: Real-ESRGAN 4x upscaling for enhanced image quality
- **Proxy Support**: Rotating proxy system for large-scale image scraping
- **CAPTCHA Handling**: Manual, retry, or skip strategies for web scraping
- **Effect Director**: LLM suggests optimal effects per segment based on content

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (in system PATH)
- CUDA-compatible GPU (optional, but recommended)
- Ollama (optional, for LLM-driven effects)

### Installation

1. **Clone and setup virtual environment:**

```bash
git clone <repository-url>
cd ai-video-generator
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Install PyTorch with CUDA support:**

```bash
# For CUDA 12.1+:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure the application:**

```bash
# Copy example config
cp config.example.py config.py

# Create .env for proxy credentials (if using proxies)
echo "PROXY_USERNAME=your-username" > .env
echo "PROXY_PASSWORD=your-password" >> .env
echo "PROXY_HOST=your-proxy-host" >> .env
echo "PROXY_PORT=80" >> .env
echo "USE_PROXIES=False" >> .env  # Set to True if using proxies
```

5. **Download AI models (optional):**

```bash
# Real-ESRGAN 4x upscaling model (64MB)
# Download from: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
# Place in: weights/RealESRGAN_x4plus.pth

# YOLO segmentation model (auto-downloads on first run)
# Will be saved to: weights/yolo11x-seg.pt
```

### First Video

1. **Create a script** in `data/input/my_story.json`:

```json
{
  "audio_file": "my_audio.mp3",
  "total_duration": 10.0,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.0,
      "duration": 5.0,
      "transcript": "A young explorer discovers ancient secrets.",
      "visual_query": "young explorer ancient ruins adventure",
      "visual_description": "Portrait of a young adventurer in front of ancient ruins"
    },
    {
      "start_time": 5.0,
      "end_time": 10.0,
      "duration": 5.0,
      "transcript": "Hidden treasures await those brave enough to seek them.",
      "visual_query": "treasure chest gold artifacts cave",
      "visual_description": "Close-up of an ancient treasure chest filled with gold"
    }
  ]
}
```

2. **Place audio file** at `data/input/my_audio.mp3`

3. **Generate video:**

```bash
python main.py my_story
```

4. **Find output** at `data/output/my_story.mp4`

## 📖 Usage

### Command Line

```bash
# Basic usage
python main.py <script_name>

# Clear cache before generating
python main.py my_story --clear-cache

# Test effects in isolation
python test_effects.py --image-dir data/test_webp_input \
                       --duration 6 \
                       --output data/output/test.mp4 \
                       --tools-file tools_herodotus_panel.json
```

### Script Format

Required fields in your JSON script:

```json
{
  "audio_file": "audio.mp3", // Audio filename (in data/input/)
  "total_duration": 15.5, // Total video duration in seconds
  "segments": [
    {
      "start_time": 0.0, // Segment start (seconds)
      "end_time": 5.0, // Segment end (seconds)
      "duration": 5.0, // Segment duration (seconds)
      "transcript": "...", // Narration text
      "visual_query": "...", // Image search query
      "visual_description": "...", // Desired visual composition
      "topic": "...", // Optional: segment topic
      "content_type": "...", // Optional: narrative_hook, event, etc.
      "reasoning": "..." // Optional: creative direction notes
    }
  ]
}
```

### Tools Configuration

Create a `tools.json` file to define parametric effects:

```json
[
  {
    "name": "zoom_on_subject",
    "params": {
      "target": "girl", // Subject to detect
      "target_scale": 1.55, // Zoom scale
      "anim_duration": 3.0 // Complete zoom by 3s, then hold
    }
  },
  {
    "name": "temporal_zoom",
    "params": {
      "scale": 1.3, // Digital zoom scale
      "ease": "easeInOut" // Easing curve
    }
  },
  {
    "name": "paneled_text",
    "params": {
      "text": "Herodotus",
      "side": "right",
      "fontsize": 128,
      "cps": 10, // Characters per second (typewriter)
      "start_time": 3.0, // Appear at 3s
      "duration": 2.5 // Stay visible 2.5s
    }
  },
  {
    "name": "neon_overlay",
    "params": {
      "color": [80, 220, 255], // RGB cyan
      "opacity": 0.06,
      "blend_mode": "screen"
    }
  }
]
```

## 🎬 Effects Reference

### Zoom Effects

**`zoom_on_subject`** - Geometric zoom with face anchoring

```json
{
  "name": "zoom_on_subject",
  "params": {
    "target": "person", // YOLO class: person, girl, boy, etc.
    "prefer": "center", // If multiple: center, left, right
    "target_scale": 1.55, // Zoom factor (1.0 = no zoom)
    "speed_curve": "easeInOut", // linear, easeInOut
    "anim_duration": 3.0, // Complete by 3s, then hold
    "anchor_feature": "auto" // auto (face), center, or custom
  }
}
```

**`temporal_zoom`** - Digital zoom overlay

```json
{
  "name": "temporal_zoom",
  "params": {
    "scale": 1.3, // Target zoom scale
    "ease": "easeInOut", // Easing curve
    "start_scale": 1.0, // Starting scale
    "method": "transform" // transform or sequence
  }
}
```

### Text Overlays

**`paneled_text`** - Styled text panel with typewriter effect

```json
{
  "name": "paneled_text",
  "params": {
    "text": "Title Text",
    "side": "right", // left, right
    "fontsize": 128,
    "panel_opacity": 0.55, // 0-1
    "border_color": [80, 220, 255], // RGB
    "animate_in": 0.35, // Slide-in duration (seconds)
    "cps": 10, // Characters per second (typewriter)
    "start_time": 3.0, // When to appear
    "duration": 2.5, // How long to stay visible
    "min_panel_width_px": 640,
    "min_panel_height_px": 220
  }
}
```

### Subject Effects

**`subject_outline`** - Expanded ring outline with glow

```json
{
  "name": "subject_outline",
  "params": {
    "color": [57, 255, 20], // RGB
    "thickness": 8, // Outline width (pixels)
    "offset": 12, // Gap from subject (pixels)
    "glow_radius": 15, // Blur amount for glow
    "opacity": 0.85, // 0-1
    "pulse": true // Animate opacity over time
  }
}
```

**`subject_pop`** - Silhouette pop with face centering

```json
{
  "name": "subject_pop",
  "params": {
    "scale": 1.2, // Pop scale factor
    "pop_duration": 0.4, // Animation duration (seconds)
    "shadow_opacity": 0.35 // Drop shadow intensity
  }
}
```

### Visual Overlays

**`neon_overlay`** - Color wash with blend modes

```json
{
  "name": "neon_overlay",
  "params": {
    "color": [80, 220, 255], // RGB
    "opacity": 0.06, // 0-1
    "blend_mode": "screen", // screen, multiply, overlay, add
    "animate_in": 0.2 // Fade-in duration
  }
}
```

## ⚙️ Configuration

### Key Settings in `config.py`

**Video Quality:**

```python
RENDER_SIZE = (1920, 1080)        # Output resolution
FPS = 30                          # Frames per second (5 for tests, 30 for production)
VIDEO_CODEC = "h264_nvenc"        # Hardware encoder (NVIDIA)
VIDEO_BITRATE = "4000k"
PRESET = "p4"                     # NVENC preset (p1=fastest, p7=best quality)
```

**AI Models:**

```python
USE_LLM_EFFECTS = True            # Enable Ollama effects director
EFFECTS_LLM_MODEL = "llama3"      # Ollama model name
ENABLE_AI_UPSCALE = True          # Enable Real-ESRGAN upscaling
UPSCALE_FACTOR = 4                # 2x or 4x upscaling
```

**Image Ranking:**

```python
CLIP_WEIGHT = 0.9                 # Semantic similarity weight
RES_WEIGHT = 0.05                 # Resolution quality weight
SHARPNESS_WEIGHT = 0.05           # Blur detection weight
RANK_MIN_CLIP_SIM = 0.18          # Minimum similarity threshold
```

**Scraping:**

```python
MAX_SCRAPED_IMAGES = 3            # Images to download per segment
SCROLL_PAUSES = 3                 # Scroll actions per search
ENABLE_CLIP_FILTER = True         # Filter during scraping
CLIP_RELEVANCE_THRESHOLD = 0.5    # Min similarity during scraping
```

## 🏗️ Architecture

### Pipeline Stages

1. **Subject Detection** (YOLO + CLIP)

   - Segment subjects in images
   - Rerank via semantic similarity
   - Detect face anchors for precise framing

2. **Geometric Reframing** (zoom_on_subject)

   - Face-anchored crop and zoom
   - Preserves aspect ratio

3. **Stylistic Effects** (neon, outline, temporal zoom)

   - Visual enhancements
   - Time-aware animations

4. **Camera Motion** (LLM-directed panning)

   - Transform-based panning (no black bars)
   - Ken Burns effects

5. **Final Scaling** (zoom_to_cover)

   - Ensures correct output dimensions
   - Face-aware cropping

6. **Late Overlays** (paneled_text)
   - Applied after scaling to prevent cropping

### Directory Structure

```
ai-video-generator/
├── CLAUDE.md                    # Agent development guide
├── config.py                    # Configuration settings
├── main.py                      # Main orchestrator
├── test_effects.py              # Effects testing harness
├── requirements.txt             # Python dependencies
├── pipeline/
│   ├── CLAUDE.md               # Pipeline architecture docs
│   ├── director_agent.py       # Segment orchestration & retries
│   ├── parser.py               # JSON script parsing
│   ├── scraper/                # Web scraping with proxies
│   │   ├── CLAUDE.md
│   │   ├── google_scraper.py
│   │   └── proxy.py
│   ├── ai_filter/              # CLIP ranking
│   │   ├── CLAUDE.md
│   │   ├── clip_ranker.py
│   │   └── semantic_filter.py
│   └── renderer/               # Video composition
│       ├── CLAUDE.md
│       ├── video_generator.py
│       ├── effects_director.py
│       ├── subject_detection.py
│       └── effects/            # Modular effects package
│           ├── CLAUDE.md
│           ├── registry.py
│           ├── zoom_variable.py
│           ├── zoom_temporal.py
│           ├── overlay_text.py
│           ├── subject_outline.py
│           └── ...
├── data/
│   ├── input/                  # Scripts and audio
│   ├── output/                 # Generated videos
│   ├── temp_images/            # Downloaded images
│   └── cache/                  # CLIP embeddings
└── weights/
    ├── yolo11x-seg.pt          # YOLO segmentation model
    └── RealESRGAN_x4plus.pth   # Upscaling model (optional)
```

## 🐛 Troubleshooting

### Common Issues

**Black bars during panning:**

- Pan effects must use uniform scaling (same factor for width and height)
- See `pipeline/renderer/video_generator.py` for correct pattern

**Horizontal/vertical squish:**

- Non-uniform resize (e.g., `resize((w*1.15, h))`) distorts aspect ratio
- Always scale both dimensions equally, then crop

**Text panels cropped away:**

- Apply `paneled_text` at Stage 6 (after final scaling)
- Check tools configuration `start_time` is within clip duration

**No images passing filter:**

- Lower `RANK_MIN_CLIP_SIM` in config.py
- Check if `visual_query` is too specific
- Review CLIP similarity scores in logs

**CUDA out of memory:**

- Reduce `RENDER_SIZE` to (1280, 720)
- Lower `MAX_SCRAPED_IMAGES`
- Close other GPU applications

### Debug Mode

Enable detailed logging:

```python
# In config.py
LOG_LEVEL = "DEBUG"
```

Check logs for:

- `[effects]` - Effect application details
- `[subject_detection]` - YOLO/CLIP detection logs
- `[director_agent]` - Segment processing and retries
- `[ai_filter]` - Image ranking scores

## 📚 Documentation

- **CLAUDE.md** - Main development guide
- **pipeline/CLAUDE.md** - Pipeline architecture
- **pipeline/renderer/CLAUDE.md** - Rendering and effects
- **pipeline/renderer/effects/CLAUDE.md** - Effects library details
- **context_state.json** - Session state and recent fixes

## 🤝 Contributing

1. Read CLAUDE.md for development guidelines
2. Follow Python docstring conventions (Args/Returns sections)
3. Test changes with `test_effects.py`
4. Visual validation required – always check rendered output

## 📄 License

MIT License

## 🙏 Acknowledgments

- [MoviePy](https://zulko.github.io/moviepy/) - Video composition
- [Ultralytics YOLO](https://ultralytics.com/) - Object detection
- [OpenAI CLIP](https://github.com/openai/CLIP) - Semantic ranking
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI upscaling
- [Ollama](https://ollama.ai/) - Local LLM inference

Create a JSON script file in `data/input/` with the following structure:

```json
{
  "audio_file": "your_audio_file.mp3",
  "total_duration": 13.692,
  "generation_method": "narrative pacing with visual-emotional pairing",
  "analysis_date": "2025-10-29",

  "segments": [
    {
      "start_time": 0.0,
      "end_time": 4.092,
      "duration": 4.092,
      "transcript": "In a quiet village where the sky brushes the fields in hues of gold,",
      "topic": "Setting and tone establishment",
      "content_type": "narrative_hook",
      "visual_query": "golden sunset village countryside fields cinematic",
      "visual_description": "Warm aerial shot of a village at sunset, golden fields swaying under soft light",
      "reasoning": "4.1s - Opening imagery designed for slow, gentle introduction; emphasis on warmth and tranquility"
    },
    {
      "start_time": 4.093,
      "end_time": 7.702,
      "duration": 3.609,
      "transcript": "young Mia discovered a map leading to forgotten treasures.",
      "topic": "Inciting discovery",
      "content_type": "narrative_event",
      "visual_query": "child holding old map candlelight treasure adventure",
      "visual_description": "Close-up of a child's hands unfolding an aged treasure map under candlelight",
      "reasoning": "3.6s - Quick transition to establish action; tighter pacing to move from calm setup to curiosity"
    }
  ],

  "summary": {
    "total_segments": 2,
    "average_duration": 3.9,
    "min_duration": 2.3,
    "max_duration": 4.1,
    "segments_under_4s": 1,
    "segments_4s_or_over": 1,
    "topic_shifts_detected": 2,
    "engagement_notes": "Short-form pacing with clear visual motifs and narrative beats."
  }
}
```

**Required Fields:**

- `audio_file`: Name of the audio file (must be in `data/input/` directory)
- `segments`: Array of video segments with:
  - `start_time`, `end_time`, `duration`: Timing information
  - `transcript`: Narration text for this segment
  - `visual_query`: Search query for finding relevant images
  - `visual_description`: Description of the desired visual
  - Additional metadata: `topic`, `content_type`, `reasoning`

### Running the Generator

Place your files in `data/input/`:

- JSON script file (e.g., `mia_story.json`)
- Audio file referenced in the JSON (e.g., `your_audio.mp3`)

Run the generator with:

```bash
# Generate video from a specific script
python main.py mia_story

# Or specify both script and audio explicitly
python main.py mia_story your_audio
```

The script name should match your JSON file (without `.json` extension).
The audio file should match the `audio_file` field in your JSON.

### Output

Generated videos will be saved to `data/output/` with the naming format:
`{script_name}.mp4`

Example: `mia_story.mp4`

### Pipeline Stages

1. **Parser** (`pipeline/parser.py`)

   - Reads input JSON
   - Creates video segment objects
   - Validates durations and timing

2. **Scraper** (`pipeline/scraper.py`)

   - Downloads candidate images for each segment
   - Filters by aspect ratio and quality
   - Stores in temp directory

3. **AI Filter** (`pipeline/ai_filter.py`)

   - Uses CLIP to rank images
   - Matches images to text content
   - Selects best-fitting images

4. **Post-processing** (`pipeline/postprocessing.py`)

   - Applies Ken Burns effect
   - Adds transitions
   - Applies visual enhancements

5. **Text Overlay** (`pipeline/text_overlay.py`)

   - Generates subtitles from audio
   - Applies stylized text overlays
   - Synchronizes with audio

6. **Renderer** (`pipeline/renderer.py`)
   - Concatenates segments
   - Adds audio track
   - Exports final video

## Agent-controllable Effect Tools

You can instruct the pipeline to apply parametric effects per segment, either via the LLM effects plan (`plan.tools`) or programmatically by setting `segment.custom_effects`.

Available tools (in `pipeline/renderer/effects_tools.py`):

- `neon_overlay(color=(57,255,20), opacity=0.18)`

  - Adds a neon color wash over the frame

- `subject_pop(bbox, scale=1.15, pop_duration=0.4, shadow_opacity=0.35)`

  - Crops the subject box and animates a bounce-in “pop” with optional shadow
  - `bbox` is normalized (x, y, w, h) in [0,1]

- `zoom_on_subject(bbox, target_scale=1.5, speed_curve='easeInOut')`

  - Variable-speed zoom toward subject center while keeping it centered

- `paneled_text(text, bbox=None, side='right', panel_color=(0,0,0), panel_opacity=0.6, border_color=(57,255,20), fontsize=48)`

  - Renders a caption panel near the subject with a neon border

- `zoom_then_panel(text, bbox=None, zoom_duration=3.0, zoom_scale=1.5, panel_side='right', typing_speed=12, ...)`
  - Combines temporal zoom with delayed panel text overlay
  - Zooms into subject over specified duration, then displays panel with typewriter effect
  - Panel is fixed to video frame (unaffected by zoom)
  - Highly customizable with 20+ parameters for zoom and panel appearance

Usage (programmatic):

```python
# For a given segment, define custom effects (applied after motion effects)
segment.custom_effects = [
  {"name": "neon_overlay", "params": {"opacity": 0.22}},
  {"name": "zoom_on_subject", "params": {"bbox": (0.35, 0.2, 0.25, 0.45), "target_scale": 1.6}},
  {"name": "subject_pop", "params": {"bbox": (0.35, 0.2, 0.25, 0.45), "scale": 1.2, "pop_duration": 0.3}},
  {"name": "paneled_text", "params": {"text": "Herodotus, c. 484–425 BC", "bbox": (0.35,0.2,0.25,0.45), "side": "right"}},
  {"name": "zoom_then_panel", "params": {
    "text": "Marcus Aurelius",
    "bbox": (0.3, 0.15, 0.4, 0.6),
    "zoom_duration": 3.0,
    "zoom_scale": 1.5,
    "panel_side": "right",
    "border_color": (255, 215, 0),
    "typing_speed": 15
  }}
]
```

Usage (from LLM plan):

```json
{
  "motion": "ken_burns_in",
  "overlays": ["neon_overlay"],
  "tools": [
    {
      "name": "zoom_on_subject",
      "params": { "bbox": [0.35, 0.2, 0.25, 0.45], "target_scale": 1.6 }
    },
    {
      "name": "paneled_text",
      "params": {
        "text": "Herodotus",
        "bbox": [0.35, 0.2, 0.25, 0.45],
        "side": "left"
      }
    }
  ]
}
```

## Configuration

Edit `config.py` to customize:

- Image/video quality settings
- AI model parameters
- Post-processing effects
- Output formats

Key settings:

```python
# Video settings
FPS = 30
VIDEO_CODEC = "libx264"
VIDEO_BITRATE = "4000k"

# Image settings
IMAGE_SIZE = (1920, 1080)
IMAGE_QUALITY = 95

# Effects
TRANSITION_DURATION = 1.0
DEFAULT_ZOOM_FACTOR = 1.1
```

## Directory Structure

```
ai_video_generator/
├── main.py              # Main orchestration
├── config.py            # Configuration settings
├── requirements.txt     # Dependencies
├── utils/
│   ├── logger.py       # Logging utilities
│   └── helpers.py      # Helper functions
├── pipeline/
│   ├── parser.py       # Input parsing
│   ├── scraper.py      # Image scraping
│   ├── ai_filter.py    # CLIP-based filtering
│   ├── postprocessing.py # Visual effects
│   ├── text_overlay.py  # Subtitle generation
│   └── renderer.py     # Final assembly
└── data/
    ├── input/          # Input files
    ├── temp_images/    # Temporary files
    └── output/         # Generated videos
```

## Logging

Logs are written to `video_generator.log` with detailed information about each stage of the pipeline. Set log level in `config.py`:

```python
LOG_LEVEL = "INFO"  # or "DEBUG" for more details
```

## Error Handling

The pipeline includes comprehensive error handling:

- Input validation
- Resource availability checks
- Process monitoring
- Cleanup of temporary files

## GPU Acceleration

The pipeline automatically uses CUDA if available:

- CLIP model inference
- Image processing
- Video encoding

## Future Enhancements

Planned features:

- [ ] Multiple aspect ratio support
- [ ] Custom effect templates
- [ ] Batch processing
- [ ] Web UI for configuration
- [ ] Cloud storage integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[MIT License](LICENSE)

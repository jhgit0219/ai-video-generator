# AI Video Generator

An AI-powered video generation pipeline that transforms text scripts into cinematic videos with automated visual effects, subject detection, and intelligent image selection.

---

## 🚀 Quick Start

### Setup (All Platforms)

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ai-video-generator
   ```

2. **Install dependencies:**

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt

   # Mac/Linux
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Download required models:**

   - **YOLO11x-seg**: [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) → Save to `weights/yolo11x-seg.pt`
   - **Real-ESRGAN** (optional): [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) → Save to `weights/RealESRGAN_x4plus.pth`
   - **Ollama LLMs**: Install [Ollama](https://ollama.com/download), then run:
     ```bash
     ollama pull llama3
     ollama pull deepseek-coder:6.7b
     ```

4. **Create .env file:**

   ```bash
   # Copy template and edit
   cp .env.example .env
   # Set USE_PROXIES=False in .env
   ```

5. **Launch the web interface:**

   ```bash
   # Windows
   .\venv\Scripts\python.exe gradio_interface.py

   # Mac/Linux
   python gradio_interface.py
   ```

6. **Open your browser** at: http://localhost:7860
   - Upload your script (text or JSON) and optional audio file
   - Click "Generate Video"
   - Download your completed video!

### Manual Setup (Advanced)

If you prefer manual setup or encounter issues with the web interface:

1. **Place your files in `data/input/`:**

   - `my_script.json` (your video script)
   - `my_script.mp3` (your narration audio - must match script name)

2. **Run the generator:**

   ```bash
   # Windows
   .\venv\Scripts\python.exe main.py my_script

   # Mac/Linux
   ./venv/bin/python3 main.py my_script
   ```

3. **Find your video** in `data/output/my_script.mp4`

### Command-Line Options

```bash
# Auto-detect matching audio file (my_story.json + my_story.mp3)
python main.py my_story

# Specify audio file explicitly (different name)
python main.py my_story custom_audio

# Clear LLM cache before generating (prevents cross-story contamination)
python main.py my_story --clear-cache
```

**File Naming Rules:**

- If you only provide the script name: `python main.py my_story`
  - Looks for `my_story.json` and `my_story.mp3` with matching names
- If you provide both: `python main.py my_story custom_audio`
  - Uses `my_story.json` with `custom_audio.mp3`
- Extensions (.json, .mp3) are optional in commands

---

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

## 🛠️ Advanced Installation (Developers)

For developers who want manual control over the installation:

### Prerequisites

- Python 3.11+
- FFmpeg (in system PATH)
- CUDA-compatible GPU (optional, but recommended)
- Ollama (optional, for LLM-driven effects)

### Manual Installation

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
# For CUDA 12.8+:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Configure the application (optional):**

```bash
# Create .env for proxy credentials (if using proxies)
echo "USE_PROXIES=False" > .env
echo "PROXY_USERNAME=your-username" >> .env
echo "PROXY_PASSWORD=your-password" >> .env
echo "PROXY_HOST=your-proxy-host" >> .env
echo "PROXY_PORT=80" >> .env
```

5. **Install Ollama (optional, for LLM effects):**

```bash
# Download from: https://ollama.com/download
# Then pull models:
ollama pull llama3
ollama pull deepseek-coder:6.7b
```

## 📖 Script Format

Your JSON script should follow this structure:

```json
{
  "audio_file": "my_audio.mp3",
  "total_duration": 10.0,
  "generation_method": "narrative pacing with visual-emotional pairing",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.0,
      "duration": 5.0,
      "transcript": "A young explorer discovers ancient secrets.",
      "visual_query": "young explorer ancient ruins adventure",
      "visual_description": "Portrait of a young adventurer in front of ancient ruins",
      "topic": "Setting and introduction",
      "content_type": "narrative_hook",
      "reasoning": "Opening scene to establish character and setting"
    },
    {
      "start_time": 5.0,
      "end_time": 10.0,
      "duration": 5.0,
      "transcript": "Hidden treasures await those brave enough to seek them.",
      "visual_query": "treasure chest gold artifacts cave",
      "visual_description": "Close-up of an ancient treasure chest filled with gold",
      "topic": "Rising action",
      "content_type": "narrative_event",
      "reasoning": "Build excitement and anticipation"
    }
  ],
  "summary": {
    "total_segments": 2,
    "average_duration": 5.0,
    "min_duration": 5.0,
    "max_duration": 5.0,
    "segments_under_7s": 2,
    "segments_7s_or_over": 0,
    "topic_shifts_detected": 2,
    "engagement_notes": "Consistent 5-second pacing for clear narrative flow."
  }
}
```

**Required Fields:**

- `audio_file`: Audio filename (must be in `data/input/`)
- `total_duration`: Total video duration in seconds
- `generation_method`: Description of pacing/generation approach
- `segments`: Array of video segments with:
  - `start_time`, `end_time`, `duration`: Timing information (in seconds)
  - `transcript`: Narration text for this segment
  - `visual_query`: Search query for finding images
  - `visual_description`: Desired visual composition
  - `topic`: Segment topic/theme
  - `content_type`: Type of content (e.g., "narrative_hook", "narrative_event")
  - `reasoning`: Creative direction notes
- `summary`: Statistics and metadata about the script:
  - `total_segments`: Number of segments
  - `average_duration`, `min_duration`, `max_duration`: Duration stats (seconds)
  - `segments_under_7s`, `segments_7s_or_over`: Pacing distribution counts
  - `topic_shifts_detected`: Number of topic transitions
  - `engagement_notes`: Notes about pacing and timing strategy

### Tools Configuration (Optional - Advanced)

For advanced users who want manual control over effects, you can create a `tools.json` file to define parametric effects per segment. **This is optional** - the LLM effects director (`USE_LLM_EFFECTS=True` in config) automatically plans effects based on your script content.

Example `tools.json` for manual effects:

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

## 🎬 Available Effects

The system includes 24+ built-in effects that are automatically applied by the LLM effects director. For advanced users, these can be manually configured via `tools.json`.

### Motion & Zoom (3 effects)

- **`zoom_on_subject`** - Geometric zoom with face anchoring
- **`temporal_zoom`** - Digital zoom overlay with smooth easing
- **`zoom_then_panel`** - Combined zoom + text panel effect

### Text Overlays (1 effect)

- **`paneled_text`** - Large styled text panels with typewriter animation

### Subject Effects (2 effects)

- **`subject_outline`** - Expanded ring outline with glow and pulse
- **`subject_pop`** - Silhouette pop effect with face-centered framing

### Visual Overlays (2 effects)

- **`neon_overlay`** - Color wash with blend modes (screen, multiply, overlay, add)
- **`nightvision`** - Green tint night vision effect with scanlines

### Flash & Pulse Effects (3 effects)

- **`flash_pulse`** - Pulsing white flash effect
- **`quick_flash`** - Quick camera flash
- **`strobe_effect`** - Strobe light effect

### Color Grading (5 effects)

- **`color_grade`** - Custom color grading
- **`warm_grade`** - Warm color tone
- **`cool_grade`** - Cool color tone
- **`desaturated_grade`** - Black and white / desaturated look
- **`teal_orange_grade`** - Cinematic teal and orange color grade

### Branding Effects (8 effects)

**Atomic Effects:**
- **`slime_splatter`** - Animated slime splatter overlay
- **`full_frame_tint`** - Full-frame color tint
- **`animated_location_text`** - Animated location/place name text
- **`subject_glow`** - Glow effect around detected subjects

**Composite Effects:**
- **`map_highlight`** - Highlight locations on maps with markers
- **`character_highlight`** - Highlight character introductions
- **`news_overlay`** - News broadcast style overlay
- **`newspaper_frame`** - Vintage newspaper frame effect
- **`branded_transition`** - Branded swipe/zoom transitions

## ⚙️ Configuration

### Key Settings in `config.py`

> **IMPORTANT:** Set `SKIP_FLAG = False` in `config.py` for normal operation. When `SKIP_FLAG = True`, the pipeline skips image scraping and uses cached images only (for testing).

**Image Processing:**

```python
SKIP_FLAG = False                 # MUST be False for normal operation (True = testing only)
IMAGE_SIZE = (1920, 1080)         # Image processing size
RENDER_SIZE = (1920, 1080)        # Final video output resolution
IMAGE_QUALITY = 95                # JPEG quality (1-100)
```

**Video Encoding:**

```python
FPS = 30                          # Frames per second
VIDEO_CODEC = "h264_nvenc"        # Hardware encoder (NVIDIA/AMD/Intel/CPU)
AUDIO_CODEC = "aac"               # Audio codec
VIDEO_BITRATE = "4000k"           # Video bitrate
AUDIO_BITRATE = "192k"            # Audio bitrate
PRESET = "p4"                     # NVENC preset (p1=fastest, p7=best quality)
```

**AI Models:**

```python
# CLIP (for semantic image ranking)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda"                   # GPU device (auto-fallback to "cpu")

# LLM Effects Director (Ollama)
USE_LLM_EFFECTS = True            # Enable Ollama effects director
EFFECTS_LLM_MODEL = "llama3"      # Primary Ollama model
DEEPSEEK_MODEL = "deepseek-coder:6.7b"  # For complex code generation
USE_MICRO_BATCHING = True         # Batch segments (10x fewer API calls)
USE_DEEPSEEK_SELECTIVE = True     # Use DeepSeek only for complex segments

# Real-ESRGAN (AI upscaling)
ENABLE_AI_UPSCALE = True          # Enable Real-ESRGAN upscaling
UPSCALE_FACTOR = 4                # 2x or 4x upscaling
UPSCALE_MODEL = "RealESRGAN_x4plus.pth"
```

**Content-Aware Effects:**

```python
USE_CONTENT_AWARE_EFFECTS = True  # Enable branded effects kit
```

**Image Ranking:**

```python
CLIP_WEIGHT = 0.9                 # Semantic similarity weight
RES_WEIGHT = 0.05                 # Resolution quality weight
SHARPNESS_WEIGHT = 0.05           # Blur detection weight
ASPECT_WEIGHT = 0.03              # Landscape aspect ratio bonus
RANK_MIN_CLIP_SIM = 0.25          # Minimum CLIP similarity to accept image
MAX_RES_MP = 2.0                  # Resolution megapixels for max quality score

# Director Agent requery behavior
DIRECTOR_MAX_RETRIES = 1          # LLM requery attempts per segment (0 = disabled)
SCRAPER_REQUERY_MAX_RETRIES = 1   # In-scraper requery attempts (0 = disabled)
```

**Image Scraping:**

```python
MAX_SCRAPED_IMAGES = 3            # Images to download per segment
SCROLL_PAUSES = 3                 # Scroll actions per search
SCROLL_SLEEP = 1.5                # Seconds between scrolls
PLAYWRIGHT_HEADFUL = False        # Show browser (set True for debugging)
ENABLE_CLIP_FILTER = False        # Disable in-scraper filtering (trust Google + post-scrape CLIP)
CLIP_RELEVANCE_THRESHOLD = 0.5    # Min similarity during scraping (if ENABLE_CLIP_FILTER=True)

# CAPTCHA handling: "manual", "retry", or "skip"
CAPTCHA_HANDLING = "retry"        # Retry with fresh context on CAPTCHA
CAPTCHA_MAX_RETRIES = 2           # Max retry attempts
```

**Performance:**

```python
PARALLEL_RENDER_METHOD = "parallel_v2"  # Parallel chunk rendering (4-8x speedup)
MAX_WORKERS = 4                   # Worker processes (2 for 8GB RAM, 4 for 16GB, 6 for 32GB)
CHUNK_DURATION = 20               # Seconds per batch
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
- [Gradio](https://gradio.app/) - Web interface framework

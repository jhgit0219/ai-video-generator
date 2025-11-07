# AI Video Generator

An AI-powered video generation pipeline that transforms text scripts into cinematic videos with automated visual effects, subject detection, and intelligent image selection.

---

## 🚀 Quick Start for Users

### One-Click Setup (Windows)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-video-generator
   ```

2. **Run the installer:**
   ```bash
   INSTALL.bat
   ```
   This will:
   - Create a Python virtual environment
   - Install all dependencies
   - Download Ollama and AI models (llama3, deepseek-coder)
   - Create a desktop shortcut

3. **Launch the web interface:**
   ```bash
   RUN.bat
   ```
   Or double-click the desktop shortcut created during installation.

4. **Open your browser** at: http://localhost:7860
   - Paste your script text or upload a JSON file
   - (Optional) Upload an audio file
   - Click "Generate Video"
   - Download your completed video!

### Setup (Mac/Linux)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-video-generator
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Install Ollama (optional, for LLM effects):**
   ```bash
   # Download from: https://ollama.com/download
   ollama pull llama3
   ollama pull deepseek-coder:6.7b
   ```

4. **Create .env file:**
   ```bash
   cp .env.example .env
   # Edit .env and set USE_PROXIES=False
   ```

5. **Launch the web interface:**
   ```bash
   source venv/bin/activate
   python gradio_interface.py
   ```

6. **Open your browser** at: http://localhost:7860

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
  "total_duration": 15.5,
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
      "visual_description": "Close-up of an ancient treasure chest filled with gold"
    }
  ]
}
```

**Required Fields:**
- `audio_file`: Audio filename (must be in `data/input/`)
- `total_duration`: Total video duration in seconds
- `segments`: Array of video segments with:
  - `start_time`, `end_time`, `duration`: Timing information
  - `transcript`: Narration text
  - `visual_query`: Search query for images
  - `visual_description`: Desired visual composition

**Optional Fields:**
- `topic`, `content_type`, `reasoning`: Metadata for LLM effects planning

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
- [Gradio](https://gradio.app/) - Web interface framework

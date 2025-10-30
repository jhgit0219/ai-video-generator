# AI Video Generator

An AI-powered video generation pipeline that creates engaging video content from text scripts and audio input.

## ⚠️ Important Setup Required

**This repository does NOT include:**

- ❌ `config.py` - Contains configuration settings (use `config.example.py` as template)
- ❌ `.env` - Contains sensitive credentials (create manually, see setup instructions)
- ❌ Model weights - 64MB AI upscaling model (download separately)
- ❌ Data/cache - Generated images and video outputs

**After cloning, you MUST:**

1. ✅ Copy `config.example.py` → `config.py`
2. ✅ Create `.env` file with your proxy credentials (if using proxies)
3. ✅ Download Real-ESRGAN model weights (optional, for AI upscaling)
4. ✅ Install dependencies: `pip install -r requirements.txt`

See [Setup](#setup) section below for detailed instructions.

## Overview

This tool automates the creation of video content by:

1. Converting text scripts into video segments
2. Finding and filtering relevant imagery
3. Applying professional video effects
4. Adding synchronized subtitles
5. Producing a final landscape-format video

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (optional but recommended)
- FFmpeg installed and in system PATH

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd ai-video-generator
```

2. Create a virtual environment:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install PyTorch with CUDA support (if available):

```bash
# For CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. Install other dependencies:

```bash
pip install -r requirements.txt
```

5. Set up configuration:

```bash
# Copy the example config
cp config.example.py config.py

# Create .env file for sensitive credentials
# Add your proxy settings (if using proxies):
echo "PROXY_USERNAME=your-username" > .env
echo "PROXY_PASSWORD=your-password" >> .env
echo "PROXY_HOST=your-proxy-host" >> .env
echo "PROXY_PORT=80" >> .env
echo "USE_PROXIES=True" >> .env
```

6. Download AI model weights (optional, for upscaling):

```bash
# Download Real-ESRGAN model (64MB)
# Place in: weights/RealESRGAN_x4plus.pth
# Download from: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

## Usage

### Input Format

Create a JSON script file in `data/input/` with the following structure:

```json
{
  "title": "My Video",
  "segments": [
    {
      "text": "First segment text",
      "duration": 5.0,
      "keywords": ["nature", "mountains"]
    },
    {
      "text": "Second segment text",
      "duration": 4.0,
      "keywords": ["city", "night"]
    }
  ]
}
```

### Running the Generator

1. Place your input JSON file in `data/input/`
2. Place your audio file in `data/input/`
3. Run the generator:

```python
from main import generate_video

output_path = generate_video(
    "data/input/script.json",
    "data/input/audio.mp3"
)
```

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

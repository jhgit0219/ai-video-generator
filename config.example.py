"""
Configuration settings for the AI Video Generator.
Contains paths, model settings, and processing parameters.

SETUP INSTRUCTIONS:
1. Copy this file to 'config.py': cp config.example.py config.py
2. Create a .env file with your proxy credentials (see .env.example)
3. Edit config.py if you need to customize any settings
4. Never commit config.py or .env to version control (already in .gitignore)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------
# File paths (directories, not individual files)
# ------------------------------------------------------------
INPUT_DIR = "data/input"
TEMP_IMAGES_DIR = "data/temp_images"
OUTPUT_DIR = "data/output"

# Scraper-specific paths
SCRAPER_CACHE_DIR = "data/scraper_cache"
SCRAPER_LOGS_DIR = "data/logs/scraper"

# ------------------------------------------------------------
# Image processing settings
# ------------------------------------------------------------
MAX_IMAGES_PER_SEGMENT = 5
IMAGE_SIZE = (1920, 1080)  # 720p landscape output (faster processing)
RENDER_SIZE = (1920, 1080)  # Use cached 1080p images, downscale to 720p at end
IMAGE_QUALITY = 95
SKIP_FLAG = False  # IMPORTANT: Set to False for normal operation (True skips scraping, uses cache only)

# AI Image Enhancement (Upscaling & Sharpening)
ENABLE_AI_UPSCALE = True      # Enable Real-ESRGAN AI upscaling (requires realesrgan package)
UPSCALE_FACTOR = 4            # 2x or 4x upscaling (2 is faster, 4 is higher quality)
UPSCALE_MODEL = "RealESRGAN_x4plus.pth"  # Use official Real-ESRGAN 4x model (download with: python download_realesrgan.py)
# Set to True to apply simple PIL sharpening if AI upscale is disabled
ENABLE_SIMPLE_SHARPEN = False  # Disable PIL sharpening when using AI upscale
SHARPEN_STRENGTH = 1.5        # 1.0 = normal, 1.5 = moderate, 2.0 = strong
# Cache enhanced images to avoid re-processing (saved in data/temp_images/enhanced_cache/)
CACHE_ENHANCED_IMAGES = True  # Disable only if you want to force re-enhancement

# Parallel Frame Rendering (for faster video encoding)
# Options: "parallel_v2" (recommended), "disk", "pipe", "chunk", or "disabled"
# - parallel_v2: TRUE parallel chunk rendering (4-8x speedup, no pickling issues)
# - disk: Render frames to disk in parallel, then encode
# - pipe: Pipe frames directly to FFMPEG (saves disk I/O)
# - chunk: Render time chunks in parallel (old implementation, has pickling issues)
# - disabled: Use single-threaded MoviePy rendering
PARALLEL_RENDER_METHOD = "parallel_v2"
FRAME_RENDER_WORKERS = None  # Number of workers (None = auto-detect CPU count, capped at MAX_WORKERS)

# Memory and CPU limits to prevent system crashes
# CRITICAL: Each worker loads YOLO (500MB) + CLIP (350MB) + video frames (varies)
# Estimate: ~1-2GB RAM per worker depending on video complexity
#
# IMPORTANT: More workers = faster rendering BUT more memory!
# With 6 batches:
#   2 workers = 3 rounds of rendering (batch 0-1, then 2-3, then 4-5)
#   4 workers = 2 rounds of rendering (batch 0-3, then 4-5)
#   6 workers = 1 round (all batches in parallel) - FASTEST but needs ~12GB RAM!
#
# Recommended settings:
#   8GB RAM:  MAX_WORKERS=2, CHUNK_DURATION=30
#   16GB RAM: MAX_WORKERS=4, CHUNK_DURATION=20
#   32GB RAM: MAX_WORKERS=6, CHUNK_DURATION=15
MAX_WORKERS = 4  # Maximum worker processes (increase if you have 16GB+ RAM)
THREADS_PER_WORKER = 2  # MoviePy threads per worker (total threads = MAX_WORKERS * THREADS_PER_WORKER)
FRAME_SEQUENCE_QUALITY = 95  # JPEG quality for disk/pipe methods (1-100)
CHUNK_DURATION = 20  # Seconds per batch (smaller = more batches = more parallelism, but more overhead)


# ------------------------------------------------------------
# Scraper settings
# ------------------------------------------------------------
MAX_CONCURRENT_SCRAPER = 2        # concurrent scraper tasks
MAX_SCRAPED_IMAGES = 3       # total images per query (successful downloads)
SCROLL_PAUSES = 3                # number of scroll actions per search
SCROLL_SLEEP = 1.5               # time between scrolls
GOOGLE_IMAGE_BATCH_SIZE = 20     # images per scroll batch
PLAYWRIGHT_HEADFUL = False        # debug mode for scraping
SEARCH_ENGINE_URL = "https://www.google.com/search?tbm=isch&q="
THUMBNAIL_ATTEMPT_LIMIT = 10     # max thumbnails to attempt per query before stopping
MAX_SCROLL_ROUNDS = 3            # safety cap for scroll/pagination rounds

# CAPTCHA handling strategy for scraping
# Options:
#  - "manual": try auto-checkbox; if challenge, pause and wait for human up to CAPTCHA_WAIT_SECONDS
#  - "retry": on detection, discard context and retry with a fresh context (randomized UA) up to CAPTCHA_MAX_RETRIES
#  - "skip": on detection, skip this query immediately
CAPTCHA_HANDLING = "retry"  # "manual" | "retry" | "skip"
CAPTCHA_MAX_RETRIES = 2
CAPTCHA_WAIT_SECONDS = 300     # used only for manual mode
CAPTCHA_RANDOMIZE_UA = True

# ------------------------------------------------------------
# Proxy settings (for all image requests)
# ------------------------------------------------------------
# Proxy credentials are loaded from .env file for security
# Example .env format:
#   PROXY_USERNAME=your-username
#   PROXY_PASSWORD=your-password
#   PROXY_HOST=p.webshare.io
#   PROXY_PORT=80

# Build proxy URL from environment variables
PROXY_USERNAME = os.getenv("PROXY_USERNAME", "")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD", "")
PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = os.getenv("PROXY_PORT", "80")

# Construct the full proxy endpoint URL
if PROXY_USERNAME and PROXY_PASSWORD and PROXY_HOST:
    ROTATING_PROXY_ENDPOINT = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
else:
    ROTATING_PROXY_ENDPOINT = ""

# Load other proxy settings from .env with fallback defaults
USE_PROXIES = os.getenv("USE_PROXIES", "True").lower() == "true"
PROXY_LIST = []  # Add additional proxies here if needed
PROXY_ROTATION_MODE = os.getenv("PROXY_ROTATION_MODE", "round_robin")
PROXY_APPLY_TO_PLAYWRIGHT = os.getenv("PROXY_APPLY_TO_PLAYWRIGHT", "True").lower() == "true"
PROXY_SCRAPER_BROWSING = os.getenv("PROXY_SCRAPER_BROWSING", "True").lower() == "true"
PROXY_LOG_EXTERNAL_IP = os.getenv("PROXY_LOG_EXTERNAL_IP", "True").lower() == "true"
PROXY_HEALTHCHECK_URL = os.getenv("PROXY_HEALTHCHECK_URL", "https://ipv4.webshare.io/")

# CLIP-based AI filtering
# NOTE: Text-to-text filtering disabled - trust Google's ranking + CLIP image-to-text ranking
ENABLE_CLIP_FILTER = False
# Minimum text-vs-text relevance (query vs alt/title) to accept a scraped image
# Increase this to demand stricter semantic matches during scraping
CLIP_RELEVANCE_THRESHOLD = 0.5


# ------------------------------------------------------------
# Video settings
# ------------------------------------------------------------
FPS = 30
# Hardware-accelerated encoding (much faster than libx264)
# Options: "h264_nvenc" (NVIDIA), "h264_amf" (AMD), "h264_qsv" (Intel), "libx264" (CPU fallback)
VIDEO_CODEC = "h264_nvenc"  # NVIDIA hardware encoder
AUDIO_CODEC = "aac"
VIDEO_BITRATE = "4000k"
AUDIO_BITRATE = "192k"
PRESET = "p4"  # NVENC preset: p1 (fastest) to p7 (slowest/best quality), p4 is balanced

# ------------------------------------------------------------
# AI model settings
# ------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda"  # Automatically falls back to "cpu" if GPU unavailable

# Effects director (LLM-driven effects selection)
USE_LLM_EFFECTS = True      # When True, use offline LLM to choose per-segment effects
USE_MICRO_BATCHING = True   # When True, batch segments together (10x fewer API calls, 78% token savings)
EFFECTS_BATCH_SIZE = 10     # Number of segments per batch (default 10 = ~3.5K tokens per batch)
USE_DEEPSEEK_EFFECTS = False  # When True, use DeepSeek for ALL segments (expensive, disables batching)
USE_DEEPSEEK_SELECTIVE = True  # When True, use DeepSeek ONLY for complex segments that need custom code
EFFECTS_LLM_MODEL = "llama3"  # Ollama model name
DEEPSEEK_MODEL = "deepseek-coder:6.7b"  # DeepSeek model for code generation

# Custom instructions for effects director (optional)
# Use this to guide the effects director with specific creative requirements
# Example: "Make sure there's subject focus edits displaying the subject's name"
# Leave empty string "" for default behavior
EFFECTS_CUSTOM_INSTRUCTIONS = ""

# Ranking weights and behavior
# Increase CLIP weight to prioritize semantic relevance over resolution
CLIP_WEIGHT = 0.9           # Weight for semantic (CLIP) similarity
RES_WEIGHT = 0.05            # Weight for resolution quality
SHARPNESS_WEIGHT = 0.05      # Weight for image sharpness (blur detection)
ASPECT_WEIGHT = 0.03         # Weight for aspect ratio preference (landscape bonus)
MAX_RES_MP = 2.0             # 2 megapixels and above gets full quality score
HTTP_TIMEOUT = 10            # Timeout (s) for image downloads in ranking
CACHE_DIR = f"{TEMP_IMAGES_DIR}/.cache"  # Cache folder for CLIP/size data

# Additional ranking constraints/tuning
# Discard candidates whose image<->text CLIP similarity falls below this floor
# Lowered from 0.18 to 0.12 to be more permissive (visual inspection shows good images being rejected)
RANK_MIN_CLIP_SIM = 0.25  # Minimum CLIP similarity for image selection (raised from 0.12 to improve quality)
# Minimum confidence threshold for including composition/quality labels in CLIP captions
# Labels with confidence below this won't be added to image captions
CLIP_CAPTION_CONFIDENCE_THRESHOLD = 0.25
# Prompt construction parameters for ranking
PROMPT_USE_EXACT_PHRASE = False   # Wrap visual_query in quotes to bias exact concept
PROMPT_TRANSCRIPT_MAX_WORDS = 12 # If transcript is used, cap words to avoid diluting prompt

# Director Agent and scraper requery retries
# Number of refine-and-retry cycles the Director Agent will perform per segment
DIRECTOR_MAX_RETRIES = 1   # set 0 to disable LLM requery retries
# Number of requery attempts the scraper will try when no images pass semantic filter
SCRAPER_REQUERY_MAX_RETRIES = 1  # set 0 to disable in-scraper requery

# ------------------------------------------------------------
# Post-processing effects
# ------------------------------------------------------------
TRANSITION_DURATION = 1.0  # seconds
DEFAULT_ZOOM_FACTOR = 1.1
VIGNETTE_INTENSITY = 0.3

# Duration alignment
# Set to "audio" to clamp the final video to the audio track length, or "json" to honor the sum of segment durations
DURATION_MODE = "json"  # "audio" | "json"
# Allow a small tolerance (in seconds) before applying a duration correction
FINAL_DURATION_TOLERANCE = 0.03

# Transitions
# Global default transition if LLM doesn't specify one
DEFAULT_TRANSITION_TYPE = "crossfade"   # "none" | "crossfade"
DEFAULT_TRANSITION_DURATION = 0.5   # seconds
# Allow the LLM effects director to control transitions per segment
ALLOW_LLM_TRANSITIONS = True

# ------------------------------------------------------------
# Content-Aware Branding Effects
# ------------------------------------------------------------
# Automatically detect locations/characters and apply branded effects
USE_CONTENT_AWARE_EFFECTS = bool(os.getenv("USE_CONTENT_AWARE_EFFECTS", "True").lower() in ("true", "1"))

# spaCy model for Named Entity Recognition
# Options: "en_core_web_sm" (fast), "en_core_web_md" (accurate), "en_core_web_lg" (best)
SPACY_MODEL = "en_core_web_sm"

# Minimum confidence for entity detection (0.0-1.0)
ENTITY_CONFIDENCE_THRESHOLD = 0.5

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
LOG_LEVEL = "DEBUG"
LOG_FILE = "video_generator.log"

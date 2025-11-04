"""
Configuration settings for the AI Video Generator.
Contains paths, model settings, and processing parameters.

SETUP INSTRUCTIONS:
1. Copy this file to 'config.py': cp config.example.py config.py
2. Edit config.py and add your sensitive credentials (proxy, API keys, etc.)
3. Never commit config.py to version control (already in .gitignore)
"""

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
SKIP_FLAG = False

# AI Image Enhancement (Upscaling & Sharpening)
ENABLE_AI_UPSCALE = True      # Enable Real-ESRGAN AI upscaling (requires realesrgan package)
UPSCALE_FACTOR = 4            # 2x or 4x upscaling (2 is faster, 4 is higher quality)
UPSCALE_MODEL = "RealESRGAN_x4plus.pth"  # Use official Real-ESRGAN 4x model (download with: python download_realesrgan.py)
# Set to True to apply simple PIL sharpening if AI upscale is disabled
ENABLE_SIMPLE_SHARPEN = False  # Disable PIL sharpening when using AI upscale
SHARPEN_STRENGTH = 1.5        # 1.0 = normal, 1.5 = moderate, 2.0 = strong
# Cache enhanced images to avoid re-processing (saved in data/temp_images/enhanced_cache/)
CACHE_ENHANCED_IMAGES = True  # Disable only if you want to force re-enhancement


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
# If you have a rotating proxy provider, set ROTATING_PROXY_ENDPOINT to the provider URL.
# If you manage your own pool, list them in PROXY_LIST and choose rotation mode.
# 
# SECURITY: Replace these placeholder values with your actual credentials.
# Format for proxy URL: "http://username:password@proxy-host:port"
USE_PROXIES = False  # Set to True when you configure your proxy
ROTATING_PROXY_ENDPOINT = ""  # Example: "http://user:pass@proxy.example.com:80"
PROXY_LIST = [
	# "http://username:password@proxy1.example.com:8080",
	# "http://username:password@proxy2.example.com:8080",
]
PROXY_ROTATION_MODE = "round_robin"  # "round_robin" | "random"
# Apply proxy also to Playwright fallback downloads (and optionally to Playwright browsing if enabled separately)
PROXY_APPLY_TO_PLAYWRIGHT = True
# Apply proxy to the main Google Images browsing as well (scraper sessions)
PROXY_SCRAPER_BROWSING = True
# Log external IP via a healthcheck endpoint when proxies are enabled
PROXY_LOG_EXTERNAL_IP = True
PROXY_HEALTHCHECK_URL = "https://ipv4.webshare.io/"  # Or your proxy provider's healthcheck URL

# CLIP-based AI filtering
ENABLE_CLIP_FILTER = True
# Minimum text-vs-text relevance (query vs alt/title) to accept a scraped image
# Increase this to demand stricter semantic matches during scraping
CLIP_RELEVANCE_THRESHOLD = 0.6


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
USE_DEEPSEEK_SELECTIVE = False  # When True, use DeepSeek ONLY for complex segments flagged by batch planner
EFFECTS_LLM_MODEL = "llama3"  # Ollama model name
DEEPSEEK_MODEL = "deepseek-coder:6.7b"  # DeepSeek model for code generation

# Ranking weights and behavior
# Increase CLIP weight to prioritize semantic relevance over resolution
CLIP_WEIGHT = 0.9           # Weight for semantic (CLIP) similarity
RES_WEIGHT = 0.05            # Weight for resolution quality
SHARPNESS_WEIGHT = 0.05      # Weight for image sharpness (blur detection)
MAX_RES_MP = 2.0             # 2 megapixels and above gets full quality score
HTTP_TIMEOUT = 10            # Timeout (s) for image downloads in ranking
CACHE_DIR = f"{TEMP_IMAGES_DIR}/.cache"  # Cache folder for CLIP/size data

# Additional ranking constraints/tuning
# Discard candidates whose image<->text CLIP similarity falls below this floor
RANK_MIN_CLIP_SIM = 0.18
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
# Logging
# ------------------------------------------------------------
LOG_LEVEL = "DEBUG"
LOG_FILE = "video_generator.log"

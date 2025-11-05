"""Add segment 15 image to enhanced cache with proper hash."""
import hashlib
import shutil
from pathlib import Path
from PIL import Image

# Source image
source_image = Path("data/temp_images/lost_labyrinth_script/segment_15/afp_68d26c9c2785-1758620828.jpg")
enhanced_cache_dir = Path("data/temp_images/enhanced_cache")
enhanced_cache_dir.mkdir(parents=True, exist_ok=True)

# Calculate MD5 hash of the source path (same as image_enhancer.py does)
image_path_str = str(source_image.absolute())
cache_key = hashlib.md5(image_path_str.encode()).hexdigest()
cache_file = enhanced_cache_dir / f"{cache_key}.jpg"

print(f"[cache] Source: {source_image}")
print(f"[cache] Cache key: {cache_key}")
print(f"[cache] Cache file: {cache_file}")

# Load, resize to render size, and save to cache
img = Image.open(source_image).convert('RGB')
print(f"[cache] Original size: {img.size}")

# Resize to render size (1920x1080)
from config import RENDER_SIZE
img_resized = img.resize(RENDER_SIZE, Image.Resampling.LANCZOS)
print(f"[cache] Resized to: {img_resized.size}")

# Save to cache
img_resized.save(cache_file, format='JPEG', quality=95)
print(f"[cache] Saved to cache: {cache_file}")
print(f"[cache] File exists: {cache_file.exists()}")

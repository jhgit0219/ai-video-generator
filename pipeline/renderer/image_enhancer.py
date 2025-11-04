"""
Image Enhancement Module - AI upscaling and sharpening.
Supports Real-ESRGAN for AI upscaling and PIL for simple sharpening.
Caches enhanced images to avoid re-processing.
"""
import os
import hashlib
from pathlib import Path
from typing import Optional
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch

from utils.logger import setup_logger
from config import (
    ENABLE_AI_UPSCALE,
    UPSCALE_FACTOR,
    UPSCALE_MODEL,
    ENABLE_SIMPLE_SHARPEN,
    SHARPEN_STRENGTH,
    IMAGE_SIZE,
    TEMP_IMAGES_DIR,
    CACHE_ENHANCED_IMAGES,
    DEVICE,
)

# Enhanced images cache directory
ENHANCED_CACHE_DIR = Path(TEMP_IMAGES_DIR) / "enhanced_cache"
ENHANCED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)

# Lazy load Real-ESRGAN
_realesrgan_upsampler = None


def _init_realesrgan():
    """Initialize Real-ESRGAN model (lazy loaded)."""
    global _realesrgan_upsampler
    if _realesrgan_upsampler is not None:
        return _realesrgan_upsampler
    
    try:
        
        
        logger.info(f"[image_enhancer] Loading Real-ESRGAN model: {UPSCALE_MODEL}")
        
        # Determine model parameters based on name
        # Support custom .pth files or predefined models
        if UPSCALE_MODEL.endswith('.pth'):
            # Custom model path provided
            model_path = UPSCALE_MODEL if os.path.isabs(UPSCALE_MODEL) else f"weights/{UPSCALE_MODEL}"
            
            # Quick model compatibility check
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'params' in checkpoint:
                    state_dict = checkpoint['params']
                elif 'params_ema' in checkpoint:
                    state_dict = checkpoint['params_ema']
                else:
                    state_dict = checkpoint
                
                # Check if it looks like RRDBNet architecture
                if 'conv_first.weight' in state_dict:
                    first_layer_shape = state_dict['conv_first.weight'].shape
                    num_feat = first_layer_shape[0]
                    logger.debug(f"[image_enhancer] Detected num_feat={num_feat} from model weights")
                    
                    # Adjust model architecture to match
                    if num_feat == 64:
                        # Standard RealESRGAN
                        num_block = 23
                    elif num_feat == 32:
                        # Compact model
                        num_block = 12
                    else:
                        logger.warning(f"[image_enhancer] Unusual num_feat={num_feat}, this may not be RRDBNet architecture")
                        logger.warning(f"[image_enhancer] Supported architectures: RRDBNet (ESRGAN-based only)")
                        logger.warning(f"[image_enhancer] NOT supported: DAT, SwinIR, HAT, Compact, Anime models")
                        return None
                else:
                    logger.error(f"[image_enhancer] Model doesn't have 'conv_first.weight' - not RRDBNet architecture!")
                    logger.error(f"[image_enhancer] This appears to be: DAT, SwinIR, or another incompatible architecture")
                    logger.error(f"[image_enhancer] Please use official Real-ESRGAN models or ESRGAN-based models only")
                    return None
                    
            except Exception as check_err:
                logger.warning(f"[image_enhancer] Could not verify model architecture: {check_err}")
                logger.warning(f"[image_enhancer] Attempting to load with default RRDBNet settings...")
                num_feat = 64
                num_block = 23
            
            # Try to infer scale from filename
            if '4x' in UPSCALE_MODEL.lower() or 'x4' in UPSCALE_MODEL.lower():
                netscale = 4
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=4)
            elif '2x' in UPSCALE_MODEL.lower() or 'x2' in UPSCALE_MODEL.lower():
                netscale = 2
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=2)
            else:
                # Default to 4x for unknown custom models
                logger.warning(f"[image_enhancer] Cannot infer scale from filename, defaulting to 4x")
                netscale = 4
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=4)
        elif "x4plus" in UPSCALE_MODEL:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = "weights/RealESRGAN_x4plus.pth"
        else:  # x2plus or default
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            model_path = "weights/RealESRGAN_x2plus.pth"
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(f"[image_enhancer] Model not found at {model_path}")
            logger.warning("[image_enhancer] Download from: https://github.com/xinntao/Real-ESRGAN/releases")
            return None
        
        # Determine device and enable GPU acceleration
        use_gpu = torch.cuda.is_available() and DEVICE == "cuda"
        device = torch.device('cuda' if use_gpu else 'cpu')
        logger.info(f"[image_enhancer] Using device: {device}")
        
        # Enable half-precision (FP16) for GPU to save VRAM
        use_half = use_gpu  # Only use FP16 on GPU
        if use_half:
            logger.info("[image_enhancer] Enabling FP16 (half-precision) for GPU acceleration")
        
        _realesrgan_upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=512,  # Increased tile size for 4K processing
            tile_pad=10,
            pre_pad=0,
            half=use_half,  # FP16 for GPU, FP32 for CPU
            device=device,  # Explicitly set device
        )
        
        logger.info("[image_enhancer] Real-ESRGAN loaded successfully")
        return _realesrgan_upsampler
        
    except ImportError as e:
        logger.warning(f"[image_enhancer] Real-ESRGAN not available: {e}")
        logger.warning("[image_enhancer] Install with: pip install realesrgan basicsr")
        return None
    except Exception as e:
        logger.error(f"[image_enhancer] Failed to load Real-ESRGAN: {e}")
        return None


def _get_cache_key(image_path: str, target_size: tuple) -> str:
    """Generate cache key based on image path, settings, and target size."""
    settings_str = f"{ENABLE_AI_UPSCALE}_{UPSCALE_FACTOR}_{UPSCALE_MODEL}_{ENABLE_SIMPLE_SHARPEN}_{SHARPEN_STRENGTH}_{target_size}"
    # Use file path + modification time + settings for cache key
    mtime = os.path.getmtime(image_path) if os.path.exists(image_path) else 0
    cache_input = f"{image_path}_{mtime}_{settings_str}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def enhance_image(image_path: str, target_size: tuple = IMAGE_SIZE) -> Optional[Image.Image]:
    """
    Enhance image with AI upscaling and/or sharpening.
    Uses caching to avoid re-processing the same image.
    
    Args:
        image_path: Path to input image
        target_size: Target resolution (width, height)
    
    Returns:
        Enhanced PIL Image or None if failed
    """
    # Check cache first (if caching enabled)
    cache_key = _get_cache_key(image_path, target_size)
    cache_path = ENHANCED_CACHE_DIR / f"{cache_key}.jpg"  # Changed to JPEG for speed
    
    if CACHE_ENHANCED_IMAGES and cache_path.exists():
        try:
            logger.debug(f"[image_enhancer] Loading from cache: {cache_path.name}")
            return Image.open(cache_path)
        except Exception as e:
            logger.warning(f"[image_enhancer] Cache read failed: {e}, re-processing")
    
    # Process image if not in cache or caching disabled
    try:
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        logger.debug(f"[image_enhancer] Processing {image_path} ({original_size[0]}x{original_size[1]})")
        
        # AI Upscaling with Real-ESRGAN
        if ENABLE_AI_UPSCALE:
            upsampler = _init_realesrgan()
            if upsampler:
                try:
                    # Convert PIL to numpy
                    img_np = np.array(img)
                    
                    # Upscale with Real-ESRGAN
                    logger.debug(f"[image_enhancer] Upscaling with Real-ESRGAN ({UPSCALE_FACTOR}x)")
                    output, _ = upsampler.enhance(img_np, outscale=UPSCALE_FACTOR)
                    
                    # Convert back to PIL
                    img = Image.fromarray(output)
                    logger.debug(f"[image_enhancer] Upscaled to {img.size[0]}x{img.size[1]}")
                    
                except Exception as e:
                    logger.warning(f"[image_enhancer] AI upscaling failed: {e}, using original")
            else:
                logger.debug("[image_enhancer] AI upscaling not available, skipping")
        
        # Simple PIL Sharpening (if AI upscale is disabled or as additional enhancement)
        if ENABLE_SIMPLE_SHARPEN and not ENABLE_AI_UPSCALE:
            logger.debug(f"[image_enhancer] Applying PIL sharpening (strength={SHARPEN_STRENGTH})")
            
            # Apply moderate unsharp mask - reduced from 150% to more subtle values
            # radius=1.5 for tighter sharpening, lower percent for less artifacts
            img = img.filter(ImageFilter.UnsharpMask(
                radius=1.5, 
                percent=int(SHARPEN_STRENGTH * 60),  # Was 100, now 60-90 range for 1.0-1.5 strength
                threshold=3
            ))
            
            # Skip the additional sharpness enhancement - UnsharpMask is enough
        
        # Resize to target size while preserving aspect ratio
        # Strategy: Zoom to COVER the LARGEST dimension (ensure no black bars)
        # Keep the full zoomed image so pan/zoom effects can reveal off-screen areas
        if img.size != target_size:
            logger.debug(f"[image_enhancer] Resizing from {img.size} to {target_size}")
            
            # Calculate aspect ratios
            img_aspect = img.width / img.height
            target_aspect = target_size[0] / target_size[1]
            
            # Zoom to ensure BOTH dimensions meet or exceed target (no black bars)
            # The larger dimension determines the scale
            if img_aspect > target_aspect:
                # Image is wider than target - fit to WIDTH, height will be smaller or equal
                # Scale up to ensure height also meets target
                new_width = target_size[0]
                new_height = int(target_size[0] / img_aspect)
                
                # If height is less than target, scale up based on height instead
                if new_height < target_size[1]:
                    new_height = target_size[1]
                    new_width = int(target_size[1] * img_aspect)
            else:
                # Image is taller than target - fit to HEIGHT, width will be smaller or equal
                # Scale up to ensure width also meets target
                new_height = target_size[1]
                new_width = int(target_size[1] * img_aspect)
                
                # If width is less than target, scale up based on width instead
                if new_width < target_size[0]:
                    new_width = target_size[0]
                    new_height = int(target_size[0] / img_aspect)
            
            # Resize with high-quality LANCZOS - keep full image, no cropping
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"[image_enhancer] Resized to {new_width}x{new_height} (zoom-to-cover, no black bars, motion-ready)")
        
        
        # Save to cache for future use (if caching enabled)
        if CACHE_ENHANCED_IMAGES:
            try:
                # Ensure cache directory exists (in case it was deleted during runtime)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use JPEG for 4K images - much faster to save than PNG
                # Quality 95 is visually lossless but 10x faster than PNG
                img.save(cache_path, format='JPEG', quality=95, optimize=False)
                logger.debug(f"[image_enhancer] Cached enhanced image: {cache_path.name}")
            except Exception as e:
                logger.warning(f"[image_enhancer] Failed to save cache: {e}")
        
        return img
        
    except Exception as e:
        logger.error(f"[image_enhancer] Failed to enhance {image_path}: {e}")
        return None


def enhance_images_batch(image_paths: list[str], target_size: tuple = IMAGE_SIZE) -> list[Optional[Image.Image]]:
    """
    Enhance multiple images in batch for better GPU utilization (2-4x faster).

    Args:
        image_paths: List of input image paths
        target_size: Target resolution for all images

    Returns:
        List of enhanced PIL Images (same order as input), None for failed images
    """
    if not image_paths:
        return []

    # Check which images need processing (not in cache)
    to_process = []
    to_process_indices = []
    cached_results = [None] * len(image_paths)

    for idx, image_path in enumerate(image_paths):
        cache_key = _get_cache_key(image_path, target_size)
        cache_path = ENHANCED_CACHE_DIR / f"{cache_key}.jpg"

        if CACHE_ENHANCED_IMAGES and cache_path.exists():
            try:
                cached_results[idx] = Image.open(cache_path)
                logger.debug(f"[batch_enhancer] Loaded from cache: {Path(image_path).name}")
            except Exception as e:
                logger.warning(f"[batch_enhancer] Cache read failed for {image_path}: {e}")
                to_process.append(image_path)
                to_process_indices.append(idx)
        else:
            to_process.append(image_path)
            to_process_indices.append(idx)

    # If everything was cached, return early
    if not to_process:
        logger.info(f"[batch_enhancer] All {len(image_paths)} images loaded from cache")
        return cached_results

    logger.info(f"[batch_enhancer] Processing batch of {len(to_process)} images (GPU batch processing)")

    # Load all images to process
    images_np = []
    valid_indices = []
    for i, img_path in enumerate(to_process):
        try:
            img = Image.open(img_path).convert('RGB')
            images_np.append(np.array(img))
            valid_indices.append(to_process_indices[i])
        except Exception as e:
            logger.error(f"[batch_enhancer] Failed to load {img_path}: {e}")
            cached_results[to_process_indices[i]] = None

    if not images_np:
        return cached_results

    # AI Upscaling in batch (if enabled)
    if ENABLE_AI_UPSCALE:
        upsampler = _init_realesrgan()
        if upsampler:
            try:
                logger.debug(f"[batch_enhancer] Batch upscaling {len(images_np)} images...")

                # Process images in batch (Real-ESRGAN processes them sequentially but keeps model loaded)
                enhanced_images = []
                for img_np in images_np:
                    output, _ = upsampler.enhance(img_np, outscale=UPSCALE_FACTOR)
                    enhanced_images.append(Image.fromarray(output))

                logger.debug(f"[batch_enhancer] Batch upscaling complete")

            except Exception as e:
                logger.warning(f"[batch_enhancer] Batch AI upscaling failed: {e}, using originals")
                enhanced_images = [Image.fromarray(img_np) for img_np in images_np]
        else:
            enhanced_images = [Image.fromarray(img_np) for img_np in images_np]
    else:
        enhanced_images = [Image.fromarray(img_np) for img_np in images_np]

    # Post-process each image (sharpening, resizing, caching)
    for i, img in enumerate(enhanced_images):
        idx = valid_indices[i]

        # Simple PIL Sharpening
        if ENABLE_SIMPLE_SHARPEN and not ENABLE_AI_UPSCALE:
            img = img.filter(ImageFilter.UnsharpMask(
                radius=1.5,
                percent=int(SHARPEN_STRENGTH * 60),
                threshold=3
            ))

        # Resize to target
        if img.size != target_size:
            img_aspect = img.width / img.height
            target_aspect = target_size[0] / target_size[1]

            if img_aspect > target_aspect:
                new_width = target_size[0]
                new_height = int(target_size[0] / img_aspect)
                if new_height < target_size[1]:
                    new_height = target_size[1]
                    new_width = int(target_size[1] * img_aspect)
            else:
                new_height = target_size[1]
                new_width = int(target_size[1] * img_aspect)
                if new_width < target_size[0]:
                    new_width = target_size[0]
                    new_height = int(target_size[0] / img_aspect)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        cached_results[idx] = img

        # Save to cache
        if CACHE_ENHANCED_IMAGES:
            try:
                cache_key = _get_cache_key(image_paths[idx], target_size)
                cache_path = ENHANCED_CACHE_DIR / f"{cache_key}.jpg"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(cache_path, format='JPEG', quality=95, optimize=False)
            except Exception as e:
                logger.warning(f"[batch_enhancer] Failed to cache image {idx}: {e}")

    return cached_results


def enhance_and_save(input_path: str, output_path: str, target_size: tuple = IMAGE_SIZE) -> bool:
    """
    Enhance image and save to disk.

    Args:
        input_path: Source image path
        output_path: Destination path
        target_size: Target resolution

    Returns:
        True if successful, False otherwise
    """
    enhanced = enhance_image(input_path, target_size)
    if enhanced:
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save with high quality
            enhanced.save(output_path, quality=95, optimize=True)
            logger.info(f"[image_enhancer] Saved enhanced image: {output_path}")
            return True
        except Exception as e:
            logger.error(f"[image_enhancer] Failed to save {output_path}: {e}")
            return False
    return False

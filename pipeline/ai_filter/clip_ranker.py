"""
CLIP-based image ranking module.
Ranks images per segment by semantic similarity (CLIP) and resolution quality.
Includes simple on-disk caching to speed up repeated runs.
"""

import os
import json
import time
import hashlib
import re
import subprocess
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

from utils.logger import setup_logger
from pipeline.parser import VideoSegment  # assumes your dataclass or similar
from config import (
    CLIP_MODEL_NAME,
    DEVICE,
    CLIP_WEIGHT,
    RES_WEIGHT,
    MAX_RES_MP,
    HTTP_TIMEOUT,
    CACHE_DIR,
    TEMP_IMAGES_DIR,
    RANK_MIN_CLIP_SIM,
    CLIP_CAPTION_CONFIDENCE_THRESHOLD,
    PROMPT_USE_EXACT_PHRASE,
    PROMPT_TRANSCRIPT_MAX_WORDS,
    EFFECTS_LLM_MODEL,
    SHARPNESS_WEIGHT,
)
from prompts import CLIP_LABEL_GENERATION_TEMPLATE

logger = setup_logger(__name__)

# Paths
os.makedirs(CACHE_DIR, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
CLIP_CACHE_PATH = os.path.join(CACHE_DIR, "clip_cache.json")
SIZE_CACHE_PATH = os.path.join(CACHE_DIR, "size_cache.json")
RANKED_MANIFEST_PATH = os.path.join(TEMP_IMAGES_DIR, "ranked_manifest.json")
LABEL_CACHE_PATH = os.path.join(CACHE_DIR, "clip_labels_cache.json")

# Lazy caches
_clip_cache: Dict[str, float] = {}
_size_cache: Dict[str, Tuple[int, int]] = {}
_label_cache: Optional[Dict[str, List[str]]] = None  # Cache for LLM-generated labels

def _convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def _load_json_cache(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"[ai_filter] Could not load cache {path}: {e}")
    return {}

def _save_json_cache(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # <— new line
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"[ai_filter] Could not save cache {path}: {e}")

# Load caches once
_clip_cache = _load_json_cache(CLIP_CACHE_PATH)
_size_cache = _load_json_cache(SIZE_CACHE_PATH)

# Initialize CLIP once
logger.info(f"[ai_filter] Loading CLIP model: {CLIP_MODEL_NAME} on {DEVICE}")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

def _cache_key(url: str, prompt: Optional[str] = None) -> str:
    m = hashlib.sha256()
    m.update(url.encode("utf-8"))
    if prompt:
        m.update(b"|")
        m.update(prompt.encode("utf-8"))
    return m.hexdigest()

def _open_image(url_or_path: str) -> Image.Image:
    # Local file
    if os.path.exists(url_or_path):
        return Image.open(url_or_path).convert("RGB")

    # Remote URL
    r = requests.get(url_or_path, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def _get_image_size(url_or_path: str) -> Tuple[int, int]:
    # Cached
    key = _cache_key(url_or_path)
    if key in _size_cache:
        w, h = _size_cache[key]
        return int(w), int(h)

    try:
        img = _open_image(url_or_path)
        w, h = img.size
        _size_cache[key] = [int(w), int(h)]
        return w, h
    except Exception as e:
        logger.warning(f"[ai_filter] Failed to get size for {url_or_path[:80]}...: {e}")
        _size_cache[key] = [0, 0]
        return 0, 0

def _resolution_score(url_or_path: str) -> float:
    w, h = _get_image_size(url_or_path)
    if w <= 0 or h <= 0:
        return 0.0
    mp = (w * h) / 1_000_000
    # Normalize to 0..1 using MAX_RES_MP as saturation
    return min(mp / MAX_RES_MP, 1.0)


def _sharpness_score(url_or_path: str) -> float:
    """Calculate image sharpness using simple edge detection. Returns 0-1 score."""
    try:
        img = _open_image(url_or_path)
        # Convert to grayscale numpy array
        gray = np.array(img.convert('L'), dtype=np.float32)
        
        # Simple gradient-based sharpness (approximates Laplacian)
        # Calculate horizontal and vertical gradients
        gy, gx = np.gradient(gray)
        # Sharpness is the sum of gradient magnitudes
        sharpness = np.sqrt(gx**2 + gy**2).mean()
        
        # Normalize: typical sharp images have mean gradient > 10, blurry < 5
        # Map 0-20 range to 0-1 score
        score = min(sharpness / 20.0, 1.0)
        logger.debug(f"[ai_filter] Sharpness for {url_or_path[:60]}: gradient={sharpness:.1f}, score={score:.3f}")
        return score
    except Exception as e:
        logger.debug(f"[ai_filter] Sharpness check failed for {url_or_path[:60]}: {e}")
        return 0.5  # Neutral score if we can't check


def _generate_contextual_labels(segments: List[VideoSegment], cache_path: str = LABEL_CACHE_PATH) -> Dict[str, List[str]]:
    """
    Use LLM to generate contextual CLIP labels based on video topic and segments.
    
    Analyzes the video's overall theme and segment content to create relevant
    scene types, composition styles, and visual qualities for CLIP classification.
    
    Args:
        segments: List of video segments to analyze
        cache_path: Path to cache file (allows separate test cache)
    
    Returns dict with keys: 'scene_labels', 'composition_labels', 'quality_labels'
    """
    global _label_cache
    
    # Check in-memory cache first (only if using default cache path)
    if cache_path == LABEL_CACHE_PATH and _label_cache is not None:
        logger.info("[ai_filter] Using cached contextual labels")
        return _label_cache
    
    # Try loading from disk
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
                logger.info(f"[ai_filter] Loaded contextual labels from cache: {cache_path}")
                # Only update global cache if using default path
                if cache_path == LABEL_CACHE_PATH:
                    _label_cache = labels
                return labels
        except Exception as e:
            logger.warning(f"[ai_filter] Failed to load label cache: {e}")
    
    # Gather context from segments
    topics = []
    content_types = []
    visual_queries = []
    descriptions = []
    transcripts = []
    
    for seg in segments:  # Use ALL segments for comprehensive context
        topic = getattr(seg, "topic", None)
        if topic:
            topics.append(str(topic))
        
        content_type = getattr(seg, "content_type", None)
        if content_type:
            content_types.append(str(content_type))
        
        vq = getattr(seg, "visual_query", None)
        if vq:
            visual_queries.append(str(vq))
        
        desc = getattr(seg, "visual_description", None)
        if desc:
            descriptions.append(str(desc))
        
        trans = getattr(seg, "transcript", None)
        if trans:
            transcripts.append(str(trans))
    
    # Build comprehensive context summary
    context = {
        "topics": list(set(topics)),  # All unique topics
        "content_types": list(set(content_types)),  # All unique content types
        "visual_queries": visual_queries,  # All queries
        "descriptions": descriptions,  # All descriptions
        "transcripts": transcripts[:10]  # First 10 transcripts for narrative understanding
    }
    
    logger.info(f"[ai_filter] Generating contextual CLIP labels for video content...")
    
    # Create focused summary - concatenate transcripts to capture character names
    sample_transcript = " ".join(context['transcripts'][:5])
    
    # Build visual queries string
    visual_queries_text = "\n".join(f"- {vq}" for vq in context['visual_queries'][:20])
    
    prompt = CLIP_LABEL_GENERATION_TEMPLATE.format(
        sample_transcript=sample_transcript[:700],
        visual_query_count=len(context['visual_queries']),
        visual_queries=visual_queries_text
    )

    try:
        result = subprocess.run(
            ["ollama", "run", EFFECTS_LLM_MODEL, "--nowordwrap"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        
        if result.returncode != 0:
            error_msg = f"LLM label generation failed with return code {result.returncode}: {result.stderr}"
            logger.error(f"[ai_filter] {error_msg}")
            raise RuntimeError(error_msg)
        
        output = result.stdout.strip()
        
        # Extract JSON from response
        parsed = _extract_json_labels(output)
        if not parsed:
            error_msg = f"LLM returned invalid JSON. Output: {output[:200]}"
            logger.error(f"[ai_filter] {error_msg}")
            raise ValueError(error_msg)
        
        if not all(k in parsed for k in ['scene_labels', 'composition_labels', 'quality_labels']):
            error_msg = f"LLM returned incomplete label structure. Got keys: {list(parsed.keys())}"
            logger.error(f"[ai_filter] {error_msg}")
            raise ValueError(error_msg)
        
        # Only update global cache if using default path
        if cache_path == LABEL_CACHE_PATH:
            _label_cache = parsed
        
        # Save to disk cache
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            logger.info(f"[ai_filter] Saved contextual labels to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"[ai_filter] Failed to save label cache: {e}")
        
        logger.info(f"[ai_filter] Generated {len(parsed['scene_labels'])} scene labels, "
                   f"{len(parsed['composition_labels'])} composition labels, "
                   f"{len(parsed['quality_labels'])} quality labels")
        return parsed
            
    except subprocess.TimeoutExpired:
        error_msg = "LLM label generation timed out after 60s"
        logger.error(f"[ai_filter] {error_msg}")
        raise TimeoutError(error_msg)
    except (RuntimeError, ValueError, TimeoutError):
        raise  # Re-raise our custom errors
    except Exception as e:
        error_msg = f"LLM label generation failed unexpectedly: {str(e)}"
        logger.error(f"[ai_filter] {error_msg}")
        raise RuntimeError(error_msg) from e


def _extract_json_labels(text: str) -> Optional[Dict[str, List[str]]]:
    """Extract JSON object from LLM response."""
    # Remove markdown code fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()
    
    # Find JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try parsing entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def generate_clip_caption(url_or_path: str, labels: Dict[str, List[str]]) -> str:
    """
    Generate a caption for an image using CLIP zero-shot classification.
    
    Uses CLIP to classify the image against descriptive labels generated from
    LLM analysis of video content.
    
    Args:
        url_or_path: Path to image file or URL
        labels: Dict with 'scene_labels', 'composition_labels', 'quality_labels'
    
    Returns:
        Concise caption combining top-scoring labels
    """
    try:
        image = _open_image(url_or_path)
        
        # Validate labels
        scene_labels = labels.get("scene_labels", [])
        composition_labels = labels.get("composition_labels", [])
        quality_labels = labels.get("quality_labels", [])
        
        if not scene_labels:
            raise ValueError("No scene labels provided for CLIP caption generation")
        
        # Process all labels at once for efficiency
        all_labels = scene_labels + composition_labels + quality_labels
        
        inputs = clip_processor(
            text=all_labels,
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        outputs = clip_model(**inputs)
        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds
        
        # Normalize embeddings
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (img_embeds @ txt_embeds.T).squeeze(0)
        
        # Get top scene type (highest similarity from scene_labels)
        scene_sims = similarities[:len(scene_labels)]
        top_scene_idx = scene_sims.argmax().item()
        top_scene = scene_labels[top_scene_idx]
        top_scene_score = scene_sims[top_scene_idx].item()
        
        # Get top composition (from composition_labels)
        comp_start = len(scene_labels)
        comp_end = comp_start + len(composition_labels)
        comp_sims = similarities[comp_start:comp_end]
        top_comp_idx = comp_sims.argmax().item()
        top_comp = composition_labels[top_comp_idx]
        top_comp_score = comp_sims[top_comp_idx].item()
        
        # Get top quality (from quality_labels)
        qual_start = comp_end
        qual_sims = similarities[qual_start:]
        top_qual_idx = qual_sims.argmax().item()
        top_qual = quality_labels[top_qual_idx]
        top_qual_score = qual_sims[top_qual_idx].item()
        
        # Build caption from high-confidence labels
        caption_parts = [top_scene]
        if top_comp_score > CLIP_CAPTION_CONFIDENCE_THRESHOLD:
            caption_parts.append(top_comp)
        if top_qual_score > CLIP_CAPTION_CONFIDENCE_THRESHOLD:
            caption_parts.append(top_qual)
        
        caption = ", ".join(caption_parts)
        logger.debug(f"[ai_filter] Generated caption for {url_or_path[:60]}: '{caption}' "
                    f"(scene={top_scene_score:.3f}, comp={top_comp_score:.3f}, qual={top_qual_score:.3f})")
        return caption
        
    except Exception as e:
        logger.warning(f"[ai_filter] Caption generation failed for {url_or_path[:80]}...: {e}")
        return "image"

@torch.no_grad()
def _clip_similarity(url_or_path: str, prompt: str) -> float:
    key = _cache_key(url_or_path, prompt)
    if key in _clip_cache:
        return float(_clip_cache[key])

    try:
        image = _open_image(url_or_path)
        inputs = clip_processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        # Move to device
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = clip_model(**inputs)
        img = outputs.image_embeds
        txt = outputs.text_embeds

        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        sim = float((img @ txt.T).item())

        _clip_cache[key] = sim
        return sim
    except Exception as e:
        logger.warning(f"[ai_filter] CLIP similarity failed for {url_or_path[:80]}...: {e}")
        _clip_cache[key] = 0.0
        return 0.0

def _build_prompt(segment: Any) -> str:
    """Construct a concise, high-signal prompt for CLIP ranking.

    Strategy:
    - Emphasize the visual_query (already refined by director agent into cinematic phrases).
    - Add topic/content_type if present.
    - Optionally include a short slice of transcript for context but cap words
      to avoid diluting the signal.
    """
    parts: list[str] = []

    vq = getattr(segment, "visual_query", None)
    if vq:
        vq = str(vq).strip()
        if PROMPT_USE_EXACT_PHRASE and len(vq.split()) <= 8:
            parts.append(f'"{vq}"')
        else:
            parts.append(vq)

    topic = getattr(segment, "topic", None)
    if topic:
        parts.append(str(topic))

    content_type = getattr(segment, "content_type", None)
    if content_type:
        parts.append(str(content_type))

    # Light/optional transcript tail for context
    t = getattr(segment, "transcript", None)
    if t and len(parts) < 2:  # only if we have little signal so far
        words = str(t).split()
        if len(words) > PROMPT_TRANSCRIPT_MAX_WORDS:
            words = words[:PROMPT_TRANSCRIPT_MAX_WORDS]
        parts.append(" ".join(words))

    prompt = ". ".join(p for p in parts if p)
    logger.debug(f"[clip_ranker] Built ranking prompt: '{prompt}'")
    return prompt

def _rank_for_segment(segment: Any, labels: Dict[str, List[str]]) -> Dict[str, Any]:
    """Rank images for a segment and generate CLIP caption using provided labels."""
    prompt = _build_prompt(segment)
    images = getattr(segment, "images", []) or []

    ranked: List[Dict[str, Any]] = []
    all_ranked: List[Dict[str, Any]] = []
    for i, img in enumerate(images):
        # Handle both structured and flat image lists
        if isinstance(img, dict):
            path = img.get("path") or img.get("url")
            url = img.get("url")
        else:
            path = img
            url = img

        clip_s = _clip_similarity(path, prompt)
        # Discard extremely off-topic images early
        if clip_s < RANK_MIN_CLIP_SIM:
            logger.debug(f"[ai_filter] Drop low-sim candidate (sim={clip_s:.3f} < {RANK_MIN_CLIP_SIM}) - {path}")
            continue
        res_s = _resolution_score(path)
        sharp_s = _sharpness_score(path) if SHARPNESS_WEIGHT > 0 else 0.5
        final = CLIP_WEIGHT * clip_s + RES_WEIGHT * res_s + SHARPNESS_WEIGHT * sharp_s

        entry = {
            "id": str(i),
            "url": url,
            "path": path,
            "clip_score": round(clip_s, 6),
            "resolution_score": round(res_s, 6),
            "sharpness_score": round(sharp_s, 6) if SHARPNESS_WEIGHT > 0 else 0.5,
            "final_score": round(final, 6),
        }
        all_ranked.append(entry)
        # Apply semantic floor here
        if clip_s >= RANK_MIN_CLIP_SIM:
            ranked.append(entry)

    # Sort best to worst
    if ranked:
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
    else:
        # Fallback: no items passed the floor, use top-5 from all candidates
        if all_ranked:
            logger.info("[ai_filter] No candidates passed semantic floor; falling back to raw top-5")
            all_ranked.sort(key=lambda x: x["final_score"], reverse=True)
            ranked = all_ranked[:5]
        else:
            ranked = []

    best_entry = ranked[0] if ranked else {}

    # Persist best on the segment object for downstream modules
    segment.selected_image = best_entry.get("path") or best_entry.get("url")
    segment.selected_url = best_entry.get("url")

    # Generate CLIP caption for the selected image using contextual labels
    if segment.selected_image:
        try:
            caption = generate_clip_caption(segment.selected_image, labels)
            segment.clip_caption = caption
            best_entry["clip_caption"] = caption  # Add to ranked manifest
            logger.info(f"[ai_filter] Generated caption for segment: '{caption}'")
        except Exception as e:
            logger.warning(f"[ai_filter] Failed to generate caption: {e}")
            segment.clip_caption = None

    # Debug info for director agent or postprocessing
    setattr(segment, "ranking_debug", {
        "prompt": prompt,
        "top3": ranked[:3],
        "top5": ranked[:5],
        "count": len(ranked),
        "clip_caption": getattr(segment, "clip_caption", None),
    })

    return {
        "prompt": prompt,
        "ranked": ranked,
        "selected": best_entry,
    }


def _write_ranked_manifest(all_rankings: Dict[str, Any], temp_dir: str | None = None) -> None:
    """
    Merge incoming rankings into the on-disk ranked manifest instead of overwriting it.
    This prevents incremental re-ranking of a subset of segments from wiping other entries.
    
    Args:
        all_rankings: Dictionary of rankings to write
        temp_dir: Story-specific temp directory. If None, uses TEMP_IMAGES_DIR from config.
    """
    # Use story-specific path if provided, otherwise fall back to global config
    working_temp_dir = temp_dir if temp_dir is not None else TEMP_IMAGES_DIR
    ranked_manifest_path = os.path.join(working_temp_dir, "ranked_manifest.json")
    
    try:
        existing = {}
        if os.path.exists(ranked_manifest_path):
            try:
                with open(ranked_manifest_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception as e:
                logger.warning(f"[ai_filter] Failed to load existing ranked manifest for merge: {e}")

        # Merge/replace only the provided segment keys
        existing.update(all_rankings)
        
        # Convert numpy types to Python native types for JSON serialization
        existing = _convert_to_json_serializable(existing)

        tmp = ranked_manifest_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            # Atomic replace: only rename if write succeeded completely
            os.replace(tmp, ranked_manifest_path)
            logger.info(f"[ai_filter] Wrote ranked manifest: {ranked_manifest_path}")
        except Exception as write_err:
            logger.error(f"[ai_filter] Failed to write ranked manifest JSON: {write_err}")
            # Clean up corrupted tmp file
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                    logger.debug("[ai_filter] Cleaned up corrupted .tmp file")
                except:
                    pass
            raise  # Re-raise to signal failure
    except Exception as e:
        logger.warning(f"[ai_filter] Failed to write ranked manifest: {e}")

def _flush_caches() -> None:
    _save_json_cache(CLIP_CACHE_PATH, _clip_cache)
    _save_json_cache(SIZE_CACHE_PATH, _size_cache)

def rank_images(segments: List[VideoSegment], base_index: int | None = None, temp_dir: str | None = None) -> List[VideoSegment]:
    """
    Rank candidate images for each segment using CLIP and resolution score.
    Mutates segments by setting `selected_image` and `ranking_debug`.
    Persists ranked results to ranked_manifest.json and caches to disk.
    
    Args:
        segments: List of VideoSegment objects to rank
        base_index: Base index for segment naming
        temp_dir: Story-specific temp directory. If None, uses TEMP_IMAGES_DIR from config.
    """
    start = time.time()
    logger.info("[ai_filter] Starting ranking - CLIP + resolution")

    # Generate contextual labels once for all segments
    logger.info("[ai_filter] Generating contextual CLIP labels from video content...")
    contextual_labels = _generate_contextual_labels(segments)
    logger.info(f"[ai_filter] Using {len(contextual_labels['scene_labels'])} scene types, "
               f"{len(contextual_labels['composition_labels'])} compositions, "
               f"{len(contextual_labels['quality_labels'])} qualities")

    rankings: Dict[str, Any] = {}
    for idx, seg in enumerate(segments):
        r = _rank_for_segment(seg, contextual_labels)
        # compute stable key: prefer explicit seg.id, otherwise map to segment_{base_index+idx} when provided
        seg_index = (base_index + idx) if base_index is not None else idx
        key = getattr(seg, "id", None) or f"segment_{seg_index}"
        rankings[key] = r

    _write_ranked_manifest(rankings, temp_dir=temp_dir)
    _flush_caches()

    elapsed = time.time() - start
    logger.info(f"[ai_filter] Completed ranking for {len(segments)} segments in {elapsed:.2f}s")
    return segments

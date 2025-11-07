# AI Filter Module

CLIP-based semantic filtering and ranking of images for video segments.

## Key Files

- `clip_ranker.py`: CLIP model loader and similarity computation
- `semantic_filter.py`: High-level filtering and ranking logic

## Ranking Pipeline

Images are ranked by weighted score:

```
score = (CLIP_weight × clip_sim) + (RES_weight × res_score) + (SHARP_weight × sharp_score) + (ASPECT_weight × aspect_score)
```

**CLIP similarity (semantic match):**

- Computes cosine similarity between image and text query embeddings
- Weighted by `CLIP_WEIGHT` (default 0.9)
- Most important factor for relevance

**Resolution score:**

- Based on megapixels: `min(1.0, megapixels / MAX_RES_MP)`
- Weighted by `RES_WEIGHT` (default 0.05)
- Prefers higher resolution images

**Sharpness score:**

- Laplacian variance to detect blur
- Weighted by `SHARPNESS_WEIGHT` (default 0.05)
- Penalizes blurry images

**Aspect ratio score:**

- Prefers landscape images (width > height)
- Weighted by `ASPECT_WEIGHT` (default 0.03)
- Score 0.0 (very portrait) to 1.0 (very landscape)

## Configuration

See `config.py`:

**Ranking Weights:**

- `CLIP_WEIGHT`: Semantic similarity weight (0.9 recommended)
- `RES_WEIGHT`: Resolution quality weight
- `SHARPNESS_WEIGHT`: Blur detection weight
- `ASPECT_WEIGHT`: Landscape image preference weight

**Filtering Thresholds:**

- `RANK_MIN_CLIP_SIM`: Floor for inclusion in ranking (default 0.18)
- `CLIP_RELEVANCE_THRESHOLD`: Min similarity during scraping (default 0.5)

**Caption Enhancement:**

- `CLIP_CAPTION_CONFIDENCE_THRESHOLD`: Min confidence for adding composition labels
- Adds labels like "close-up", "wide shot", "high quality" to improve matching

**Prompt Construction:**

- `PROMPT_USE_EXACT_PHRASE`: Wrap query in quotes for exact matching
- `PROMPT_TRANSCRIPT_MAX_WORDS`: Cap transcript words to avoid diluting prompt

## CLIP Model

Loaded via Hugging Face Transformers:

- Model: `openai/clip-vit-base-patch32`
- Device: CUDA (GPU) if available, falls back to CPU
- Cached in `~/.cache/huggingface/`

**Performance:**

- First load takes ~2s to download/load model
- Subsequent loads instant (cached)
- Inference: ~10-50ms per image on GPU, ~200-500ms on CPU

## Semantic Filtering Flow

1. **Scraper-time filtering** (optional via `ENABLE_CLIP_FILTER`):

   - Each scraped image checked against query
   - If similarity < `CLIP_RELEVANCE_THRESHOLD`, rejected immediately
   - Reduces downloads of irrelevant images

2. **Post-scraping ranking**:
   - All downloaded images ranked by combined score
   - Images below `RANK_MIN_CLIP_SIM` excluded
   - Top-ranked image selected for segment

## Caption Enhancement

`semantic_filter.py` can augment image captions with composition labels:

- Detected labels: close-up, wide shot, portrait, landscape, high quality, bokeh
- Only added if confidence > `CLIP_CAPTION_CONFIDENCE_THRESHOLD`
- Helps CLIP better match images to queries mentioning composition

**Example:**

```
Original query: "portrait of a girl"
Image detected as: close-up, portrait, high quality
Enhanced caption: "portrait of a girl, close-up, high quality"
→ Better CLIP match
```

## Debugging Ranking Issues

**No images passing filter:**

- Lower `RANK_MIN_CLIP_SIM` (try 0.1 or 0.05)
- Check if visual_query is too specific
- Disable scraper-time filtering (`ENABLE_CLIP_FILTER=False`) to see more candidates

**Wrong image selected:**

- Check `[ai_filter]` logs for similarity scores
- Increase `CLIP_WEIGHT` to prioritize semantic match over resolution
- Review visual_query quality (may be too generic or misaligned)

**Slow ranking:**

- CLIP inference on CPU is slow (~500ms per image)
- Use GPU if available (automatic with `DEVICE="cuda"`)
- Consider reducing `MAX_SCRAPED_IMAGES` to rank fewer candidates

**Caption enhancement not helping:**

- Adjust `CLIP_CAPTION_CONFIDENCE_THRESHOLD` (lower = more labels, higher = only confident labels)
- Check logs for detected labels: `[ai_filter] enhanced caption: ...`

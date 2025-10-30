# CLIP Caption Integration

## Overview

Added CLIP-based image captioning to automatically analyze downloaded images and generate descriptions. This enhances both ranking relevance and effects director decision-making by providing AI-powered understanding of actual image content.

## Implementation

### 1. Caption Generation (`pipeline/ai_filter/clip_ranker.py`)

**New Function: `generate_clip_caption(url_or_path: str)`**

- Uses CLIP zero-shot classification to analyze images
- Classifies against 3 categories of labels:
  - **Scene Types** (15 labels): cityscape, aerial view, landscape, portrait, architectural detail, etc.
  - **Composition** (6 labels): vertical/horizontal composition, centered framing, symmetrical, dramatic perspective, wide angle
  - **Visual Qualities** (4 labels): highly detailed, minimalist, dramatic lighting, soft/subtle
- Returns concise caption combining top-scoring labels (threshold: 0.25)
- Examples:
  - `"cityscape with tall buildings, vertical composition, dramatic lighting"`
  - `"aerial view from above, wide angle view, highly detailed"`
  - `"landscape with natural scenery, horizontal composition"`

**Integration Points:**

- Called automatically during image ranking after selecting best image
- Caption stored in `segment.clip_caption` attribute
- Added to ranked manifest for persistence
- Included in ranking debug info

### 2. Effects Director Enhancement (`pipeline/renderer/effects_director.py`)

**Updated System Prompt:**

- Added "INPUT INFORMATION" section explaining:
  - Description = Human-written context
  - CLIP Analysis = AI-generated analysis of ACTUAL image
  - Priority: Use CLIP Analysis to understand what's actually in the image

**Updated User Prompt:**

- Dynamically includes CLIP caption when available:
  ```
  IMAGE CONTENT:
  Visual Query: {vq}
  Description: {visdesc}
  CLIP Analysis: {clip_caption}  ← NEW
  Reasoning: {reasoning}
  ```
- Emphasizes: "The CLIP Analysis tells you what the image actually contains"
- Prompts LLM to use CLIP analysis for informed motion decisions

**Updated Example Decisions:**

- Now reference composition types: "Cityscape with vertical composition"
- Align with CLIP's classification categories

## Benefits

### 1. Improved Ranking Relevance

- **Problem**: Ranking relied solely on text queries, couldn't verify if image matched
- **Solution**: CLIP caption describes actual image content
- **Result**: Can validate semantic match between query and downloaded image

### 2. Better Motion Decisions

- **Problem**: Effects director only knew human description, not actual image
- **Solution**: CLIP analysis reveals true composition and scene type
- **Result**: Motion choices match actual image structure
  - Vertical cityscapes → pan up
  - Horizontal landscapes → pan left-to-right
  - Close-ups → zoom in
  - Wide shots → zoom out

### 3. Automatic Pipeline

- **Problem**: Required manual description of image composition
- **Solution**: CLIP auto-generates scene/composition analysis
- **Result**: Less manual work, more accurate to downloaded images

### 4. Zero-Shot Flexibility

- No additional model downloads or training
- Reuses existing CLIP model already loaded for ranking
- Works on any image type without fine-tuning

## Example Flow

```
1. Image downloaded: desert_dunes.jpg
   ↓
2. CLIP Ranking: Scores similarity to query "Egyptian desert aerial shot"
   ↓
3. CLIP Caption: Analyzes best image
   → "aerial view from above, wide angle view, highly detailed"
   ↓
4. Effects Director receives:
   - Description: "Aerial shot of Egyptian desert with sand dunes"
   - CLIP Analysis: "aerial view from above, wide angle view, highly detailed"
   ↓
5. Intelligent decision:
   - Recognizes "aerial view" + "wide angle"
   - Chooses: pan_down (descending) or zoom_out (show scale)
   - Reasoning: Aerial views benefit from descent or revealing context
```

## Configuration

No new config needed! Uses existing:

- `CLIP_MODEL_NAME` (already configured for ranking)
- `DEVICE` (GPU/CPU selection)

## Testing

Run `test_clip_caption.py` to verify caption generation:

```bash
python test_clip_caption.py
```

Tests:

- Loads images from `data/temp_images/segment_*/`
- Generates captions for multiple images
- Shows caption generation working

## Technical Details

**Label Categories:**

- 25 total labels across 3 categories
- Processed in single CLIP inference (efficient)
- Top label from each category selected
- Only includes labels with confidence > 0.25

**Performance:**

- No additional model loading (reuses existing CLIP)
- One forward pass per selected image (minimal overhead)
- Cached implicitly through CLIP's efficiency

**Error Handling:**

- Falls back to "image" if caption generation fails
- Logs warnings for debugging
- Effects director still works with human descriptions only

## Next Steps

1. **Clear Cache**: Delete `enhanced_cache/*` to regenerate with new captions
2. **Test Pipeline**: Run full pipeline with SKIP_FLAG=True
3. **Verify Motion**: Check effects director logs for CLIP-informed decisions
4. **Tune Labels**: Adjust label categories if needed for your content type
5. **Monitor Quality**: Compare motion decisions before/after CLIP integration

## Notes

- CLIP captions stored in segment object and ranked manifest
- Both human descriptions and CLIP analysis available to effects director
- Effects director prioritizes CLIP analysis as "ground truth" of image content
- Fallback: If CLIP caption unavailable, uses human description only

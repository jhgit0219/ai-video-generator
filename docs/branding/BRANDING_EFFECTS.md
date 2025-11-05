# Branding Effects Documentation

Comprehensive guide to the on-brand visual effects available in the AI Video Generator.

## Overview

The branding effects package provides four categories of effects designed for specific visual storytelling patterns common in historical, documentary, and educational content:

1. **Map Highlights** - Location introductions with slime green markers
2. **Character Highlights** - Person/subject emphasis with cyan glow
3. **News Overlays** - Vintage newspaper-style text presentations
4. **Branded Transitions** - Purple/green color-coded transitions

All text effects use **Century font family** for brand consistency.

## Effect Registry

All branding effects are registered in `pipeline/renderer/effects/registry.py`:

```python
from .branding import (
    apply_map_highlight,
    apply_character_highlight,
    apply_news_overlay,
    apply_branded_transition,
)

TOOLS_REGISTRY = {
    # ... existing effects
    "map_highlight": apply_map_highlight,
    "character_highlight": apply_character_highlight,
    "news_overlay": apply_news_overlay,
    "branded_transition": apply_branded_transition,
}
```

## 1. Map Highlight Effect

**Use Case**: First introduction of a geographic location in the narrative.

**Visual Style**: Bright slime green (#00FF00) rectangular highlight with location name appearing letter-by-letter, then fading to realistic sepia tone.

**Function**: `apply_map_highlight(clip, location_name, box_position, ...)`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | VideoClip | required | Input video clip (map image) |
| `location_name` | str | required | Name to display (e.g., "EGYPT") |
| `box_position` | tuple | required | Normalized (x, y, w, h) coords for highlight box |
| `cps` | int | 8 | Characters per second for typing animation |
| `highlight_duration` | float | 2.0 | Duration to hold bright green (seconds) |
| `fade_to_normal_duration` | float | 1.5 | Duration to fade to sepia (seconds) |
| `box_thickness` | int | 8 | Border thickness in pixels |
| `fontsize` | int | 72 | Font size for location name |
| `font_path` | str | None | Path to Century font (auto-detected if None) |
| `text_position` | str | "center" | Text placement: "center", "top", "bottom" |
| `glow_intensity` | float | 0.6 | Glow effect intensity (0-1) |

### Example Usage

```json
{
  "name": "map_highlight",
  "params": {
    "location_name": "EGYPT",
    "box_position": [0.35, 0.25, 0.3, 0.2],
    "cps": 8,
    "highlight_duration": 2.5,
    "fade_to_normal_duration": 1.5,
    "text_position": "top"
  }
}
```

### Animation Timeline

```
0.0s ─────> 2.0s ─────> 3.5s ─────> END
  │            │           │
  │            │           └─ Realistic sepia tone
  │            └─ Start fade to sepia
  └─ Bright slime green with letter-by-letter text
```

### Future Enhancement (Content-Aware)

**Note**: A future enhancement will make this effect content-aware:
- When a location is first introduced in the script (e.g., "Egypt"), the system will:
  1. Scrape a 3D globe image pointing to that location
  2. Automatically apply the map_highlight effect with the slime green marker
  3. Animate the location name letter-by-letter using Century font

This will require extending the effects director to detect location mentions and trigger appropriate visual queries.

## 2. Character Highlight Effect

**Use Case**: Introduce or emphasize a key character/person in the narrative.

**Visual Style**: Cyan/green (#00FFFF) glow around subject with name label appearing letter-by-letter. Multi-stage intensity: bright → medium → subtle.

**Function**: `apply_character_highlight(clip, character_name, bbox, mask, ...)`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | VideoClip | required | Input video clip |
| `character_name` | str | required | Name to display (e.g., "HERODOTUS") |
| `bbox` | tuple | None | Normalized (x, y, w, h) subject bbox (auto-injected) |
| `mask` | ndarray | None | Subject segmentation mask (auto-injected) |
| `cps` | int | 10 | Characters per second for typing animation |
| `glow_color` | tuple | (0,255,255) | RGB color for glow effect |
| `glow_stages` | tuple | (1.0,0.6,0.3) | Intensity levels for bright/medium/subtle |
| `stage_durations` | tuple | (0.5,1.0,1.5) | Duration for each stage in seconds |
| `fontsize` | int | 64 | Font size for character name |
| `font_path` | str | None | Path to Century font (auto-detected if None) |
| `label_position` | str | "top" | Label placement: "top", "bottom" |
| `label_offset` | int | 20 | Pixel offset between subject and label |

### Example Usage

```json
{
  "name": "character_highlight",
  "params": {
    "character_name": "HERODOTUS",
    "glow_color": [0, 255, 200],
    "glow_stages": [1.0, 0.6, 0.3],
    "stage_durations": [0.5, 1.2, 1.8],
    "label_position": "top"
  }
}
```

**Note**: `bbox` and `mask` are automatically injected from pre-computed subject detection data (YOLO+CLIP) if available. See `video_generator.py:259` for injection logic.

### Animation Stages

```
Stage 1 (0.0-0.5s): Bright cyan glow (100% intensity)
  ├─ Full brightness glow around subject
  └─ Name appears letter-by-letter

Stage 2 (0.5-1.7s): Medium glow (60% intensity)
  ├─ Smooth transition to medium brightness
  └─ Name fully visible

Stage 3 (1.7-3.5s): Subtle glow (30% intensity)
  └─ Gentle persistent glow
```

### Subject Detection Integration

Character highlight leverages the parallel rendering optimization:
- Subject detection (YOLO+CLIP) runs once in main process
- Bbox and mask are serialized and passed to workers
- No redundant model loading in parallel workers

## 3. News Overlay Effect

**Use Case**: Present historical quotes, newspaper headlines, or documentary-style text.

**Visual Style**: Vintage newspaper layout with aged paper texture, green-highlighted key phrases, and classic typography.

**Function**: `apply_news_overlay(clip, headline, subheadline, body_text, ...)`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | VideoClip | required | Input video clip |
| `headline` | str | required | Main headline text |
| `subheadline` | str | None | Optional subheadline |
| `body_text` | str | None | Optional body content |
| `highlighted_phrases` | list | None | List of phrases to highlight in green |
| `paper_opacity` | float | 0.9 | Background paper opacity (0-1) |
| `position` | tuple | (0.1, 0.1) | Normalized (x, y) position |
| `size` | tuple | (0.8, 0.3) | Normalized (width, height) size |
| `headline_fontsize` | int | 56 | Font size for headline |
| `body_fontsize` | int | 28 | Font size for body text |
| `font_path` | str | None | Path to Century font |
| `fade_in_duration` | float | 0.5 | Fade in animation duration |
| `fade_out_duration` | float | 0.5 | Fade out animation duration |
| `start_time` | float | 0.0 | When overlay appears |
| `duration` | float | None | How long overlay stays visible |

### Example Usage

```json
{
  "name": "news_overlay",
  "params": {
    "headline": "Underground City? Giza Pyramids Yield New Secrets",
    "subheadline": "Italian Researchers Claim of the Shaft",
    "body_text": "New evidence from ground-penetrating radar suggests unexplored chambers beneath the Giza plateau.",
    "highlighted_phrases": ["Underground City", "New Secrets"],
    "position": [0.05, 0.6],
    "size": [0.9, 0.35],
    "start_time": 1.0,
    "duration": 4.0
  }
}
```

### Visual Elements

**Paper Texture**:
- Sepia tone base (#F0E4C8)
- Random noise for aged appearance
- Vintage stains and grain
- Subtle Gaussian blur for softness

**Text Layout**:
- Centered headline with underline separator
- Word-wrapped body text with line spacing
- Green highlights (#00FF64) for key phrases
- Black border frame (#1E1E1E)

**Timing**:
```
start_time ─> fade_in ─> hold ─> fade_out ─> end
  1.0s         0.5s      3.0s      0.5s      5.0s
```

## 4. Branded Transitions

**Use Case**: Transitions between segments using brand color palette (purple/violet and neon green).

**Visual Style**: Multiple transition presets with different effects: wipes, flashes, leaks, noise, graffiti, luminance.

**Function**: `apply_branded_transition(clip, transition_type, direction, duration, intensity)`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | VideoClip | required | Input video clip |
| `transition_type` | str | "purple_wipe" | Transition style (see types below) |
| `direction` | str | "in" | "in" (fade in), "out" (fade out), "both" |
| `duration` | float | 1.0 | Transition duration in seconds |
| `intensity` | float | 1.0 | Effect intensity (0-1) |

### Transition Types

| Type | Brand Color | Description |
|------|-------------|-------------|
| `purple_wipe` | Purple (#8B00FF) | Simple vertical wipe transition |
| `green_flash` | Neon Green (#00FF00) | Quick bright flash |
| `leak_1` | Purple (#8B00FF) | Liquid drip effect, variant 1 |
| `leak_2` | Violet (#9400D3) | Liquid drip effect, variant 2 |
| `neon_green` | Neon Green (#00FF00) | Neon glow fade |
| `dashed_graffiti` | Purple (#8B00FF) | Dashed line pattern |
| `noise_2` | Random | Static/noise effect |
| `old` | Sepia (#704214) | Vintage film fade |
| `luminance` | White (#FFFFFF) | Brightness-based fade |

### Example Usage

```json
{
  "name": "branded_transition",
  "params": {
    "transition_type": "leak_1",
    "direction": "both",
    "duration": 1.2,
    "intensity": 0.8
  }
}
```

### Brand Colors Reference

```python
PURPLE_BRAND = (139, 0, 255)   # #8B00FF - Primary purple
VIOLET_BRAND = (148, 0, 211)   # #9400D3 - Secondary violet
NEON_GREEN = (0, 255, 0)       # #00FF00 - Accent green
CYAN_BRAND = (0, 255, 255)     # #00FFFF - Character highlights
SEPIA_TONE = (112, 66, 20)     # #704214 - Vintage effects
```

## Performance Characteristics

All branding effects are designed for compatibility with parallel rendering:

**Memory Footprint**:
- Map highlight: ~2-5MB per segment (PIL operations)
- Character highlight: ~5-10MB per segment (glow layers + blur)
- News overlay: ~3-8MB per segment (texture generation)
- Branded transitions: ~1-3MB per segment (simple overlays)

**Render Time** (per 6-second segment, 1920x1080):
- Map highlight: ~0.8-1.2s
- Character highlight: ~1.0-1.5s (with mask)
- News overlay: ~0.5-1.0s
- Branded transitions: ~0.3-0.8s

**Parallel Rendering Impact**:
- All effects work in multiprocessing workers (no CUDA dependencies)
- Century font auto-detection works across worker processes
- Effects add <1.5s overhead per segment on average

## Font Requirements

All text effects use **Century font family**. The system attempts to load fonts in this order:

1. `font_path` parameter (if provided)
2. `CENTURY.TTF` (system fonts)
3. `century.ttf` (system fonts)
4. `Century.ttf` (system fonts)
5. Fallback to Arial or default font with warning

**Installation** (Windows):
Century fonts are typically pre-installed. If missing:
1. Download Century font family
2. Install to `C:\Windows\Fonts\`
3. Or specify absolute path in `font_path` parameter

**Installation** (Linux):
```bash
sudo apt-get install fonts-liberation
# Or place Century.ttf in ~/.fonts/
fc-cache -f -v
```

## Integration with LLM Effects Director

The branding effects are registered in `TOOLS_REGISTRY` and can be used by:

1. **Manual tools.json**: Specify effects explicitly per segment
2. **LLM Effects Director**: Add tool descriptions to director prompt

To enable LLM-driven branding effects, the effects director would need:
- Tool descriptions explaining when to use each effect
- Parameters guidance for common use cases
- Examples of proper usage patterns

## Testing

Test configuration for validating branding effects:

```bash
# Test all branding effects with sample images
python test_effects.py \
  --image-dir data/test_webp_input \
  --duration 6 \
  --output data/output/test_branding.mp4 \
  --tools-file docs/branding/tools_branding_test.json
```

See `tools_branding_test.json` for example configuration.

## Known Limitations

1. **Century Font Availability**: If Century font is not installed, effects fall back to Arial
2. **Content-Aware Map Effect**: Not yet implemented (requires location detection and globe scraping)
3. **Character Name Extraction**: Requires manual specification or future NER integration
4. **Green Highlight Detection**: News overlay requires manual phrase specification

## Future Enhancements

1. **Content-Aware Location Detection**:
   - NLP-based location entity extraction from transcript
   - Automatic 3D globe image scraping for detected locations
   - Automated map_highlight application on first location mention

2. **Character Name Extraction**:
   - Named Entity Recognition (NER) for person detection
   - Automatic character_highlight on first character introduction

3. **Smart Highlight Detection**:
   - Keyword extraction for news_overlay green highlights
   - Sentiment analysis to determine highlight worthiness

4. **Adaptive Styling**:
   - Theme-based color palette selection
   - Dynamic font sizing based on text length
   - Responsive layout for different aspect ratios

## Code References

**Implementation Files**:
- `pipeline/renderer/effects/branding/map_highlight.py` - Map highlight effect
- `pipeline/renderer/effects/branding/character_highlight.py` - Character glow effect
- `pipeline/renderer/effects/branding/news_overlay.py` - Newspaper overlay
- `pipeline/renderer/effects/branding/branded_transitions.py` - Brand color transitions
- `pipeline/renderer/effects/registry.py:58-62` - Registry integration
- `pipeline/renderer/video_generator.py:259` - Subject data injection

**Related Documentation**:
- `CLAUDE.md` - Main project documentation
- `pipeline/renderer/CLAUDE.md` - Renderer architecture
- `pipeline/renderer/effects/CLAUDE.md` - Effects system overview

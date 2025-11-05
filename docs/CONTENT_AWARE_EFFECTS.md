# Content-Aware Branding Effects

Automatic detection and application of branding effects based on video content analysis.

## Overview

The content-aware effects system automatically:
- Detects location mentions → applies **map_highlight** effect
- Detects character names → applies **character_highlight** effect
- Detects news-worthy content → applies **news_overlay** effect

No manual configuration required. Effects are intelligently applied on first mentions.

## How It Works

### 1. Entity Extraction (spaCy NER)

```
Transcript: "In Egypt, the pyramids were built thousands of years ago."
           ↓ spaCy NER
Entities: locations=["Egypt"], persons=[]
```

### 2. First Mention Tracking

```
Segment 0: "Welcome to our story"
  → No entities

Segment 1: "In Egypt, the pyramids..."
  → First mention of "Egypt" ✓
  → Mark segment.is_first_location_mention = True

Segment 2: "The Egyptian pharaohs..."
  → "Egypt" already seen, not first mention
```

### 3. Visual Query Modification

For first location mentions, visual query is modified to scrape map/globe imagery:

```
Original: "pyramids of giza"
Modified: "3D globe showing Egypt highlighted, world map with Egypt marked"
```

### 4. Effect Injection

Effects are automatically added to the effect plan:

```json
{
  "motion": "pan_up",
  "overlays": [],
  "tools": [
    {
      "name": "map_highlight",
      "params": {
        "location_name": "EGYPT",
        "box_position": [0.35, 0.25, 0.3, 0.2]
      }
    }
  ]
}
```

## Configuration

### Enable/Disable Feature

`config.py`:
```python
USE_CONTENT_AWARE_EFFECTS = True  # Set to False to disable
```

### spaCy Model Selection

Default: `en_core_web_sm` (small, fast, ~40MB)

For better accuracy, use larger model:
```bash
python -m spacy download en_core_web_md  # Medium, ~90MB
```

Update `pipeline/nlp/entity_extractor.py`:
```python
EntityExtractor(model_name="en_core_web_md")
```

## Entity Types Detected

### Locations (GPE, LOC, FAC)
- Countries: "Egypt", "Greece", "China"
- Cities: "Rome", "Athens", "Cairo"
- Landmarks: "Colosseum", "Parthenon"

### Persons (PERSON)
- Historical figures: "Herodotus", "Cleopatra"
- Mythological: "Zeus", "Athena"
- Modern names: "Einstein", "Curie"

### News-Worthy Content
- Discovery announcements: "researchers discover..."
- Quotes: "As Herodotus wrote: '...'"
- Questions: "Underground City?"

## Character Inference

Bare character names are enhanced with context:

```
Input: "Herodotus wrote about the Persian Wars."
       ↓ Character Inference
Output: "Greek historian Herodotus"
```

Inference sources:
- **Gender**: Name database (Cleopatra → female)
- **Occupation**: Keywords (wrote → historian)
- **Nationality**: Context mentions (Persian Wars → Greek)

## Effect Customization

### Map Highlight Defaults

```python
{
  "location_name": "EGYPT",           # Auto-capitalized
  "box_position": [0.35, 0.25, 0.3, 0.2],  # Center region
  "cps": 8,                           # Characters per second
  "highlight_duration": 2.5,          # Bright green duration
  "fade_to_normal_duration": 1.5,     # Fade to sepia
  "text_position": "center"
}
```

To customize, modify `ContentAwareEffectsDirector.inject_branding_effects()`.

### Character Highlight Defaults

```python
{
  "character_name": "HERODOTUS",      # Auto-capitalized
  "glow_color": [0, 255, 200],        # Cyan-green
  "glow_stages": [1.0, 0.6, 0.3],     # Bright → medium → subtle
  "stage_durations": [0.5, 1.2, 1.8], # Stage timings
  "label_position": "top",
  "cps": 10
}
```

### News Overlay Defaults

```python
{
  "headline": "Auto-extracted from first sentence",
  "body_text": "Second and third sentences",
  "highlighted_phrases": ["Key", "Phrases"],  # Auto-detected
  "position": [0.05, 0.6],           # Bottom of frame
  "size": [0.9, 0.35],               # Full width
  "start_time": 0.5,
  "duration": 4.0
}
```

## Performance Impact

| Operation | Time (per script) | Memory |
|-----------|------------------|--------|
| spaCy NER | ~0.1-0.3s | ~50MB |
| Character inference | ~0.05s | Minimal |
| Keyword extraction | ~0.02s | Minimal |
| **Total overhead** | **~0.2-0.4s** | **~50MB** |

Negligible impact on overall render time (6 minutes for full video).

## Debugging

### Enable Debug Logging

```python
# In utils/logger.py, set level
logger.setLevel(logging.DEBUG)
```

Look for:
```
[entity_extractor] Found location: Egypt
[content_aware_effects] Segment 1: First mention of location 'Egypt'
[video_generator] Segment 1: Modified query for location 'Egypt'
[content_aware_effects] Injected map_highlight for EGYPT
```

### Common Issues

**Issue**: No entities detected
- **Cause**: spaCy model not installed
- **Fix**: `python -m spacy download en_core_web_sm`

**Issue**: Wrong entity type (e.g., "Egypt" detected as PERSON)
- **Cause**: Ambiguous context
- **Fix**: Add more context to transcript or use larger spaCy model

**Issue**: Effects not appearing
- **Cause**: `USE_CONTENT_AWARE_EFFECTS=False`
- **Fix**: Set to `True` in `config.py`

## Testing

### Unit Tests

```bash
pytest tests/test_entity_extractor.py -v
pytest tests/test_character_inference.py -v
pytest tests/test_keyword_extractor.py -v
pytest tests/test_content_aware_effects_director.py -v
```

### Integration Test

```bash
pytest tests/test_video_generator_content_aware.py -v
```

### Manual Test with Sample Script

```bash
python main.py --script data/input/test_content_aware.json
```

Expected: Map highlight on segment 1 (Egypt), character highlight on segment 2 (Herodotus).

## Architecture Diagram

```
Transcript
    ↓
EntityExtractor (spaCy NER)
    ↓
  locations, persons
    ↓
ContentAwareEffectsDirector
    ├─→ First mention detection
    ├─→ Character inference
    ├─→ Keyword extraction
    └─→ Effect injection
         ↓
    Effect Plan (tools list)
         ↓
  CinematicEffectsAgent
         ↓
    Rendered Video
```

## Future Enhancements

1. **Geocoding Integration**: Use geocoding API to calculate actual box_position based on lat/lon
2. **Sentiment Analysis**: Adjust effect intensity based on narrative tone
3. **Coreference Resolution**: Track "he", "she", "it" references to entities
4. **Multi-language Support**: Detect language and load appropriate spaCy model
5. **Custom Entity Training**: Train spaCy on domain-specific entities (ancient civilizations, etc.)

## Code References

- Entity extraction: `pipeline/nlp/entity_extractor.py`
- Character inference: `pipeline/nlp/character_inference.py`
- Keyword extraction: `pipeline/nlp/keyword_extractor.py`
- Director integration: `pipeline/renderer/content_aware_effects_director.py`
- Video generator: `pipeline/renderer/video_generator.py:724-760`

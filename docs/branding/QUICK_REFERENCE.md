# Branding Effects - Quick Reference

Fast lookup for branding effect parameters and usage patterns.

## Map Highlight

**Use**: Location introductions with slime green markers

```json
{
  "name": "map_highlight",
  "params": {
    "location_name": "EGYPT",
    "box_position": [0.35, 0.25, 0.3, 0.2],
    "cps": 8,
    "highlight_duration": 2.0,
    "fade_to_normal_duration": 1.5,
    "text_position": "center"
  }
}
```

**Required**: `location_name`, `box_position`
**Color**: Slime green (#00FF00) → sepia

## Character Highlight

**Use**: Character/person emphasis with cyan glow

```json
{
  "name": "character_highlight",
  "params": {
    "character_name": "HERODOTUS",
    "glow_color": [0, 255, 200],
    "glow_stages": [1.0, 0.6, 0.3],
    "stage_durations": [0.5, 1.0, 1.5],
    "label_position": "top"
  }
}
```

**Required**: `character_name`
**Color**: Cyan (#00FFFF)
**Auto-injected**: `bbox`, `mask` (from YOLO+CLIP)

## News Overlay

**Use**: Vintage newspaper-style text

```json
{
  "name": "news_overlay",
  "params": {
    "headline": "Underground City Found",
    "subheadline": "Giza Pyramids Yield Secrets",
    "body_text": "Researchers claim discovery...",
    "highlighted_phrases": ["Underground City", "Secrets"],
    "position": [0.1, 0.6],
    "size": [0.8, 0.3],
    "start_time": 1.0,
    "duration": 4.0
  }
}
```

**Required**: `headline`
**Colors**: Sepia paper (#F0E4C8), green highlights (#00FF64)

## Branded Transitions

**Use**: Transitions with brand colors

```json
{
  "name": "branded_transition",
  "params": {
    "transition_type": "purple_wipe",
    "direction": "in",
    "duration": 1.0,
    "intensity": 0.8
  }
}
```

**Types**: `purple_wipe`, `green_flash`, `leak_1`, `leak_2`, `neon_green`, `dashed_graffiti`, `noise_2`, `old`, `luminance`
**Directions**: `in`, `out`, `both`

## Brand Colors

```python
SLIME_GREEN = (0, 255, 0)        # Map highlights
CYAN_GLOW = (0, 255, 255)        # Character glow
PURPLE = (139, 0, 255)           # Transitions
VIOLET = (148, 0, 211)           # Transitions alt
NEWS_GREEN = (0, 255, 100)       # Text highlights
SEPIA = (112, 66, 20)            # Vintage tone
```

## Common Patterns

### Location Introduction
```json
[
  {"name": "branded_transition", "params": {"transition_type": "green_flash", "direction": "in"}},
  {"name": "map_highlight", "params": {"location_name": "ROME", "box_position": [0.4, 0.3, 0.2, 0.15]}}
]
```

### Character Introduction
```json
[
  {"name": "character_highlight", "params": {"character_name": "JULIUS CAESAR", "label_position": "top"}},
  {"name": "branded_transition", "params": {"transition_type": "purple_wipe", "direction": "out"}}
]
```

### Historical Quote
```json
{
  "name": "news_overlay",
  "params": {
    "headline": "Historical Quote",
    "body_text": "The famous words of...",
    "highlighted_phrases": ["famous"],
    "start_time": 0.5,
    "duration": 5.0
  }
}
```

## Testing

```bash
# Test all effects
python test_effects.py \
  --image-dir data/test_webp_input \
  --duration 6 \
  --output data/output/test_branding.mp4 \
  --tools-file docs/branding/tools_branding_test.json
```

## Font Setup

**Windows**: Century fonts pre-installed
**Linux**: `sudo apt-get install fonts-liberation`

## Performance

| Effect | Time/Segment | Memory |
|--------|--------------|--------|
| Map | ~1.0s | ~3MB |
| Character | ~1.2s | ~7MB |
| News | ~0.8s | ~5MB |
| Transitions | ~0.5s | ~2MB |

# Custom Effects Instructions - Usage Guide

The effects director now supports custom instructions to guide the creative decisions made by the LLM when selecting or generating effects.

## How to Use

1. **Edit `config.py`** and set the `EFFECTS_CUSTOM_INSTRUCTIONS` parameter:

```python
# Example: Request subject focus with name display
EFFECTS_CUSTOM_INSTRUCTIONS = "Make sure there's subject focus edits displaying the subject's name"

# Example: Request specific mood and pacing
EFFECTS_CUSTOM_INSTRUCTIONS = "Create intense, fast-paced effects with dramatic zooms. Emphasize tension in every shot."

# Example: Request specific visual style
EFFECTS_CUSTOM_INSTRUCTIONS = "Apply vintage film aesthetic with warm tones and subtle grain. Pan slowly to reveal details."
```

2. **Run the pipeline** as normal:

```bash
python main.py mia_story
```

The effects director (both standard and DeepSeek) will incorporate your instructions when making creative decisions.

## How It Works

### Standard Effects Director (`USE_LLM_EFFECTS=True`)

When you provide custom instructions, they are injected into the LLM prompt:

```
CUSTOM CREATIVE REQUIREMENTS:
Make sure there's subject focus edits displaying the subject's name

IMPORTANT: Incorporate these requirements into your effects decisions.
If they mention specific effects or subject focus that aren't in the available tools,
explain in your reasoning how you're addressing them with the available options.
```

The LLM will then select from available effects (pan, zoom, overlays) that best fulfill your requirements.

### DeepSeek Code Generator (`USE_DEEPSEEK_EFFECTS=True`)

DeepSeek will **generate custom Python/MoviePy code** that implements your requirements:

```
CUSTOM CREATIVE REQUIREMENTS:
Make sure there's subject focus edits displaying the subject's name

IMPORTANT: Your generated code must incorporate these requirements. If they mention specific
effects or behaviors not shown in the examples, be creative and implement them using the
available MoviePy capabilities.
```

DeepSeek can create entirely new effects code to match your vision.

## Example Use Cases

### 1. Subject Name Display

```python
EFFECTS_CUSTOM_INSTRUCTIONS = "Display the subject's name (Mia, Whiskers) in stylized text when they first appear"
```

### 2. Narrative Pacing

```python
EFFECTS_CUSTOM_INSTRUCTIONS = "First segment: slow, establishing. Second segment: build tension with faster motion. Final segment: explosive reveal."
```

### 3. Stylistic Consistency

```python
EFFECTS_CUSTOM_INSTRUCTIONS = "Apply consistent vintage film look: warm color grading, subtle vignette, and light film grain on all segments"
```

### 4. Subject-Aware Motion

```python
EFFECTS_CUSTOM_INSTRUCTIONS = "Always keep detected subjects (faces, characters) centered in frame. Use YOLO detection to anchor all camera movements."
```

### 5. Genre-Specific Effects

```python
EFFECTS_CUSTOM_INSTRUCTIONS = "Fantasy storybook style: magical sparkle overlays, dreamy soft focus, and gentle floating camera movements"
```

## Tips

- **Be specific** about what you want, but allow creative interpretation
- **Mention subjects** by name if you want character-specific effects
- **Describe mood** rather than technical details for best results
- **Leave empty** (`""`) for default AI-driven creative decisions
- **Combine requirements** by listing them in one string

## Technical Details

- Instructions are passed to both `effects_director.py` (standard) and `deepseek_effects_director.py` (code generator)
- The LLM receives your instructions along with segment context (transcript, visual description, CLIP analysis)
- Standard director will adapt existing effects; DeepSeek can generate new code
- Safety validation still applies to all generated code

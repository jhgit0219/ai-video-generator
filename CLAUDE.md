# AI Video Generator – Agent Development Guide

This is an AI-powered video generator using Python for orchestration, MoviePy for compositing, YOLO + CLIP for subject detection, PIL for effects, and Ollama for LLM-driven creative decisions. It turns text scripts into cinematic videos with automated effects, transitions, and stylistic enhancements.

Read more about the purpose and business logic at the end. This doc is for building.

## Initial Setup (already done)

Before you started, the following commands were already run: `python -m venv venv`, set everything up, and `pip install -r requirements.txt` installed all packages. The app is now running.

## Commands useful in development

- `pip install <package>`: install a new package
- `pip freeze > requirements.txt`: update requirements after adding packages
- `.\\venv\\Scripts\\python.exe main.py --script data/input/script.json`: run the full pipeline
- `.\\venv\\Scripts\\python.exe test_effects.py --image-dir data/test_webp_input --duration 6 --output data/output/test.mp4 --tools-file tools.json`: test effects in isolation

Nearly all changes you make will result in needing to re-run the test harness. If you make changes to effects or pipeline logic, re-run `test_effects.py` to validate.

## Linting and Formatting

Code is not automatically formatted. Keep it clean manually. Type hints are preferred for all function signatures.

## Testing

- Quick effects tests: `python test_effects.py --image-dir <dir> --duration <seconds> --output <path>`
- Full pipeline tests: `python main.py --script data/input/<script>.json`
- Visual validation required – always check the rendered .mp4 output

## Architecture Overview

The pipeline has 6 conceptual stages:

1. **Subject Detection & Precompute** (YOLO segmentation + CLIP reranking → face anchor detection)
2. **Geometric Reframing** (zoom_on_subject – face-anchored crop/zoom)
3. **Final Scaling** (zoom_to_cover – aspect-ratio-preserving resize + crop to RENDER_SIZE)
4. **Stylistic Effects** (neon_overlay, subject_outline, temporal_zoom, etc.)
5. **Camera Motion** (LLM-directed pan_up/down/left/right, ken_burns, zoom_pan)
6. **Late Overlays** (paneled_text applied AFTER scaling to prevent cropping)

**Critical ordering:** Geometric → Stylistic → Motion → Scaling → Late overlays. Out-of-order operations cause black bars, aspect ratio distortion, or cropped effects.

See `pipeline/CLAUDE.md` for detailed pipeline architecture and `pipeline/renderer/CLAUDE.md` for rendering and effects specifics.

## Configuration

Config lives in `config.py`. Key settings:

- `RENDER_SIZE`: final output dimensions (currently 1920x1080 landscape)
- `FPS`: frames per second (5 for fast tests, 24-30 for production)
- `VIDEO_CODEC`: "h264_nvenc" (NVIDIA GPU encoding) or "libx264" (CPU fallback)
- `USE_LLM_EFFECTS`: enable/disable Ollama LLM for effects planning
- `ENABLE_AI_UPSCALE`: enable Real-ESRGAN upscaling

## Debugging

Most dev time is spent debugging effects composition and timing issues. Here's how to do it well:

- **Visual inspection is mandatory.** Always render and watch the .mp4 to see what actually happened.
- **Check logs:** Effects log frame transformations, anchor detection, and timing. Look for `[effects]`, `[subject_detection]`, and `[main] Stage X` messages.
- **Isolate effects:** Comment out tools in the JSON file one by one to find which effect is causing issues.
- **Frame size tracking:** Log `clip.size` before and after each effect to catch aspect ratio changes or unexpected resizing.
- **Test with short durations:** Use `--duration 3` for fast iteration when debugging timing-sensitive effects.

### MoviePy Gotchas

**Critical patterns to follow:**

- `with_position()` on resized clips creates black bars → use `transform()` instead
- `resized()` and `cropped()` can flatten time-based animations → use `transform()` for time-aware effects
- Always resize uniformly (same factor for width and height) to preserve aspect ratio, then crop

Common failure modes:

- **Black bars during pan:** Pan effects must uniformly upscale, then crop. See `pipeline/renderer/video_generator.py` `_pan_horizontal/vertical` for the correct pattern.
- **Horizontal/vertical squish:** Non-uniform resize (e.g., `resize((w*1.15, h))`) distorts aspect ratio. Always scale both dimensions equally.
- **Paneled text cropped away:** Apply paneled_text at Stage 6 (after `zoom_to_cover`) to prevent scaling from cropping it.
- **Static zoom animation:** If zoom appears frozen, check that `transform()` is used instead of `resized()`, and verify `anim_duration` is set.

## Code Style and Documentation

Prefer self-explanatory method and variable names over comments. However, **every function must have a detailed docstring** following Python best practices with Args and Returns sections:

```python
def apply_variable_zoom_on_subject(
    clip: VideoClip,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    target_scale: float = 1.5,
    anim_duration: Optional[float] = None,
) -> VideoClip:
    """
    Apply face-anchored geometric zoom that reframes the composition.

    Detects subject via YOLO+CLIP, finds face anchor, then smoothly zooms
    to target_scale over anim_duration (if specified) or clip duration.

    Args:
        clip: Input video clip to zoom.
        bbox: Optional normalized (x, y, w, h) subject bounding box.
              If None, auto-detects via YOLO+CLIP.
        target_scale: Zoom factor (e.g., 1.5 = 150% zoom).
        anim_duration: Duration in seconds to complete zoom, then hold.
                       If None, animates over entire clip duration.

    Returns:
        Zoomed VideoClip with face-anchored framing.
    """
    # Implementation...
```

Keep inline comments minimal. Use them only for non-obvious algorithmic choices or MoviePy workarounds.

## Fandom Agency Coding Standards

This project follows the Fandom Agency coding standards outlined in `Coding_standards.pdf`. When writing code, adhere to these guidelines:

### General Principles

- Code readability is paramount - write code that explains itself
- Follow PEP 8 as the foundation, with specific overrides noted below
- Prefer refactoring complex code over adding comments

### Naming Conventions

- **Packages/Modules**: `snake_case`
- **Classes**: `PascalCase` (e.g., `CustomJpegGenerator`, not `CustomJPEGGenerator`)
- **Functions**: `snake_case`
- **Variables**: `snake_case` with meaningful names (no single characters except `i` for counters)
- **Constants**: `UPPERCASE`
- **Private variables/functions**: `_snake_case_with_leading_underscore`
- **Keywords conflicts**: Add trailing underscore (e.g., `input_`)

### Comments

- **Avoid comments** - prefer self-explanatory code
- Let user write their own comments instead.

### Code Style

- **Indentation**: 4 spaces (no tabs)
- **Function length**: ~50 lines max (soft limit)
- **File length**: 0-500 lines approximately
- **Line ending**: Always end Python modules with a blank line

### Function Parameters

Use hanging indents for long parameter lists:

```python
def function_name(
    self,
    longer_variable_one_name,
    longer_variable_two_name,
):
```

### Control Flow

Avoid nested if statements - use early returns instead:

```python
# Good
if user != actual_username:
    return InvalidUsername()
if password != actual_password:
    return InvalidPassword()

session = Session(username, password)
```

### Error handling

Do not abuse error handling. Only enclose code that can raise an exception inside a try except block.

As much as possible, avoid raising `Exception`, use a more focused exception class.

### Type Hints

- **Required** for all function parameters and return types (unless return is None)
- Add space between variable name and type: `variable: str`
- Example: `def function_name(param: str, count: int = 10) -> int:`

### Docstrings

- **Required** for all classes and public functions
- Use triple double quotes: `"""`
- No space between opening quotes and first word
- Must end with period
- Follow reStructuredText format:

```python
"""Brief summary of what the function does.

:param param_name: Parameter description
:raises ErrorType: Error description
:return: Return description
"""
```

Do not use `rtype`, let type hinting fufill that purposes.

### Logging

- Use proper logging, never print statements in production
- Example: `logger = logging.getLogger(__name__)`

> If the project has an internal logging library, use that instead.

### Best Practices

- Eliminate duplicate code - extract to functions/modules
- Don't compare boolean variables to True/False directly
- Use meaningful variable names that explain purpose
- Keep functions focused on single responsibility

Maybe modular or something, but it's important when writing code.

## Official Documentation Links

- [MoviePy Documentation](https://zulko.github.io/moviepy/) – video compositing and clip manipulation
- [Pillow (PIL) Documentation](https://pillow.readthedocs.io/) – image processing and effects
- [Ultralytics YOLO](https://docs.ultralytics.com/) – object detection and segmentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) – CLIP and other models
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) – local LLM API

## Current State and Known Issues

**Working correctly:**

- Face-anchored zoom with `anim_duration` timing control
- Typewriter text panels with `cps` and `start_time`/`duration`
- Transform-based panning with uniform upscale (no black bars, no distortion)
- Temporal zoom overlay for dynamic motion
- Subject detection with YOLO+CLIP reranking and face anchoring

**Pending improvements:**

- Harden intrinsic zoom animation in `zoom_on_subject` to avoid needing `temporal_zoom` overlay
- Add `animate_out` parameter to `paneled_text` for exit animations
- Expose more panel customization via CLI (width fraction, colors)

## Context File

`context_state.json` contains session state, recent fixes, and architecture documentation. Update it when major changes or fixes are applied.

---

**Questions?** Check the logs first, visually inspect the output second, then dive into the specific effect module causing trouble. Most issues are timing/ordering problems or MoviePy API misuse.

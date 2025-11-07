"""Gradio web interface for AI Video Generator.

Provides browser-based interface for generating videos from text scripts
without requiring command-line usage or technical knowledge.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from config import (
    FPS,
    RENDER_SIZE,
    USE_LLM_EFFECTS,
    ENABLE_AI_UPSCALE,
    VIDEO_CODEC,
)
from main import full_pipeline

logger = logging.getLogger(__name__)


def parse_script_input(
    text_input: str,
    json_file: Optional[gr.File]
) -> Tuple[Optional[dict], Optional[str]]:
    """Parse user script input from text or JSON file.

    :param text_input: Raw text script input from text box.
    :param json_file: Uploaded JSON file (if provided).
    :return: Tuple of (parsed_script_dict, error_message).
    """
    # Priority: JSON file upload > text input
    if json_file is not None:
        try:
            with open(json_file.name, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
            return script_data, None
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON file: {e}"
        except Exception as e:
            return None, f"Error reading JSON file: {e}"

    if text_input.strip():
        # Try parsing as JSON first
        try:
            script_data = json.loads(text_input)
            return script_data, None
        except json.JSONDecodeError:
            # Treat as plain text - create simple segment structure
            script_data = {
                "segments": [
                    {
                        "transcript": text_input.strip(),
                        "start": 0.0,
                        "end": 10.0,  # Default 10 seconds
                        "tools": []
                    }
                ]
            }
            return script_data, None

    return None, "Please provide a script (text or JSON file)"


def generate_video(
    text_input: str,
    json_file: Optional[gr.File],
    resolution: str,
    fps: int,
    use_llm: bool,
    use_upscale: bool,
    enable_agents: bool,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Generate video from script with user-specified settings.

    :param text_input: Raw text script input.
    :param json_file: Uploaded JSON file (if provided).
    :param resolution: Output resolution (e.g., "1920x1080").
    :param fps: Frames per second.
    :param use_llm: Enable LLM-based effects planning.
    :param use_upscale: Enable AI upscaling.
    :param enable_agents: Enable segment analysis agents.
    :param progress: Gradio progress tracker.
    :return: Tuple of (output_video_path, status_message).
    """
    progress(0, desc="Parsing script...")

    # Parse input
    script_data, error = parse_script_input(text_input, json_file)
    if error:
        return None, f"❌ Error: {error}"

    # Save script to temporary JSON file
    progress(0.1, desc="Preparing script...")
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False,
        encoding='utf-8'
    ) as tmp_script:
        json.dump(script_data, tmp_script, indent=2)
        script_path = tmp_script.name

    try:
        # Parse resolution
        progress(0.15, desc="Configuring settings...")
        width, height = map(int, resolution.split('x'))

        # Override config temporarily
        import config
        original_settings = {
            'RENDER_SIZE': config.RENDER_SIZE,
            'FPS': config.FPS,
            'USE_LLM_EFFECTS': config.USE_LLM_EFFECTS,
            'ENABLE_AI_UPSCALE': config.ENABLE_AI_UPSCALE,
        }

        config.RENDER_SIZE = (width, height)
        config.FPS = fps
        config.USE_LLM_EFFECTS = use_llm
        config.ENABLE_AI_UPSCALE = use_upscale

        # Generate output filename
        progress(0.2, desc="Starting video generation...")
        output_filename = f"generated_{Path(script_path).stem}.mp4"
        output_path = Path("data/output") / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run pipeline with progress updates
        logger.info(f"[gradio] Starting video generation: {output_path}")
        logger.info(f"[gradio] Settings: {width}x{height}, {fps}fps, LLM={use_llm}, Upscale={use_upscale}, Agents={enable_agents}")

        def progress_callback(stage: str, percent: float):
            """Update progress bar during pipeline execution."""
            progress(0.2 + (percent * 0.7), desc=f"{stage}...")

        # Run full pipeline
        full_pipeline(
            script_path=script_path,
            output_path=str(output_path),
            enable_agents=enable_agents,
            progress_callback=progress_callback
        )

        # Restore original settings
        progress(0.95, desc="Finalizing...")
        for key, value in original_settings.items():
            setattr(config, key, value)

        # Clean up temp script
        os.unlink(script_path)

        progress(1.0, desc="Complete!")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        return str(output_path), f"✅ Video generated successfully! ({file_size_mb:.1f} MB)"

    except Exception as e:
        logger.error(f"[gradio] Video generation failed: {e}", exc_info=True)

        # Restore original settings
        import config
        if 'original_settings' in locals():
            for key, value in original_settings.items():
                setattr(config, key, value)

        # Clean up temp script
        if os.path.exists(script_path):
            os.unlink(script_path)

        return None, f"❌ Generation failed: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create Gradio interface with all components.

    :return: Gradio Blocks interface.
    """
    with gr.Blocks(
        title="AI Video Generator",
        theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("""
        # 🎬 AI Video Generator

        Transform your script into a cinematic video with AI-powered effects, transitions, and visuals.

        **Quick Start:**
        1. Paste your script text or upload a JSON file
        2. Adjust settings (optional)
        3. Click "Generate Video"
        4. Download your video when complete
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Script Input")

                text_input = gr.Textbox(
                    label="Script Text",
                    placeholder=(
                        "Paste your script here (plain text or JSON format)\n\n"
                        "Example:\n"
                        "The lost city of Atlantis has fascinated historians for centuries. "
                        "Ancient philosopher Herodotus first documented tales of this "
                        "mysterious civilization..."
                    ),
                    lines=10,
                    max_lines=20
                )

                json_file = gr.File(
                    label="Or Upload JSON Script",
                    file_types=['.json'],
                    type='filepath'
                )

                gr.Markdown("*Tip: JSON file upload takes priority over text input*")

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")

                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=[
                        "1920x1080",  # 1080p landscape
                        "1280x720",   # 720p landscape
                        "3840x2160",  # 4K landscape
                        "1080x1920",  # 1080p portrait (TikTok/Reels)
                        "720x1280",   # 720p portrait
                    ],
                    value="1920x1080"
                )

                fps = gr.Slider(
                    label="Frame Rate (FPS)",
                    minimum=5,
                    maximum=60,
                    step=1,
                    value=FPS,
                    info="Higher = smoother but slower to render"
                )

                use_llm = gr.Checkbox(
                    label="Enable LLM Effects Planning",
                    value=USE_LLM_EFFECTS,
                    info="Use Ollama to intelligently plan effects"
                )

                enable_agents = gr.Checkbox(
                    label="Enable Segment Analysis Agents",
                    value=True,
                    info="Fetch Wikipedia data & visual style analysis"
                )

                use_upscale = gr.Checkbox(
                    label="Enable AI Upscaling",
                    value=ENABLE_AI_UPSCALE,
                    info="⚠️ Much slower but higher quality"
                )

        with gr.Row():
            generate_btn = gr.Button(
                "🎬 Generate Video",
                variant="primary",
                size="lg"
            )

        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

        with gr.Row():
            video_output = gr.Video(
                label="Generated Video",
                format="mp4"
            )

        gr.Markdown("""
        ---
        ### 📚 Script Format

        **Plain Text:** Just paste your script and the system will create a single video segment.

        **JSON Format:** For advanced control with multiple segments:
        ```json
        {
          "segments": [
            {
              "transcript": "Your text here...",
              "start": 0.0,
              "end": 10.0,
              "tools": []
            }
          ]
        }
        ```

        ### ⚡ Performance Tips

        - **Fast rendering:** 1280x720, 5-10 FPS, disable upscaling
        - **Best quality:** 1920x1080, 24-30 FPS, enable all features
        - **4K production:** 3840x2160, 30 FPS, enable upscaling (very slow!)

        ### 🎨 Available Effects

        The system automatically applies:
        - Subject detection & smart zoom
        - Branded transitions (swipe, zoom-through)
        - Character highlights & newspaper frames
        - Map highlights with visual markers
        - Dynamic text overlays
        - Content-aware visual effects
        """)

        # Connect generate button
        generate_btn.click(
            fn=generate_video,
            inputs=[
                text_input,
                json_file,
                resolution,
                fps,
                use_llm,
                use_upscale,
                enable_agents,
            ],
            outputs=[video_output, status_output]
        )

    return interface


def main():
    """Launch Gradio web interface."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    logger.info("[gradio] Starting AI Video Generator web interface...")

    # Create interface
    interface = create_interface()

    # Launch
    interface.launch(
        server_name="0.0.0.0",  # Allow external access (Docker)
        server_port=7860,
        share=False,  # Set True for temporary public URL
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()

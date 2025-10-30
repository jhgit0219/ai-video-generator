"""
Main module for the AI Video Generator.
Orchestrates the complete pipeline from input parsing to final video rendering.
Supports command-line arguments for input files or auto-detects from INPUT_DIR.
"""

import os
import sys
import json
import asyncio
import argparse
from glob import glob
from typing import Optional
from pathlib import Path
from utils.logger import setup_logger
from utils.helpers import ensure_directory, initialize_required_directories
from pipeline.parser import parse_input, extract_script_context
from pipeline.scraper import collect_images_for_segments
from pipeline.ai_filter import rank_images
from pipeline.postprocessing import apply_effects, overlay_subtitles
from pipeline.renderer import render_final_video
from pipeline.director_agent import supervise_segments, refine_initial_queries

from config import INPUT_DIR, TEMP_IMAGES_DIR, OUTPUT_DIR, SKIP_FLAG

logger = setup_logger(__name__)

async def async_generate_video(input_json_path: str, audio_file: str) -> Optional[str]:
    """
    Async video generation pipeline to support async scraping.
    
    Args:
        input_json_path: Path to input JSON script
        audio_file: Path to audio file
    
    Returns:
        Path to generated video or None if failed
    """
    try:
        # Initialize all required directories on startup
        initialize_required_directories()
        
        # Extract story name from input file for cache isolation
        story_name = Path(input_json_path).stem  # e.g., "mia_story" or "lost_labyrinth_script"
        logger.info(f"Processing story: {story_name}")
        
        # Use story-specific temp directory to prevent cache conflicts
        story_temp_dir = os.path.join(TEMP_IMAGES_DIR, story_name)
        
        # Ensure story-specific directory exists
        ensure_directory(story_temp_dir)
        
        logger.info("Starting video generation pipeline")

        # 1. Parse input
        segments, script_data = parse_input(input_json_path)
        script_context = extract_script_context(script_data)
        logger.info(f"Script context extracted:\n{script_context}")

        # Check if SKIP_FLAG is enabled and ranked manifest exists
        ranked_manifest_path = Path(story_temp_dir) / "ranked_manifest.json"
        should_skip = SKIP_FLAG  # Use local variable to avoid scoping issues
        
        if should_skip and ranked_manifest_path.exists():
            logger.info(f"SKIP_FLAG=True -- Loading existing images and rankings for '{story_name}' from manifest")
            logger.info("   Skipping: scraping, ranking, and director agent supervision")
            
            # Load ranked manifest and attach to segments
            try:
                with open(ranked_manifest_path, "r", encoding="utf-8") as f:
                    ranked_data = json.load(f)
                
                for i, seg in enumerate(segments):
                    seg_key = f"segment_{i}"
                    if seg_key in ranked_data:
                        seg_data = ranked_data[seg_key]
                        
                        # Restore images list (local paths from ranked list)
                        ranked_list = seg_data.get("ranked", [])
                        seg.images = [entry["path"] for entry in ranked_list if "path" in entry]
                        
                        # Restore selected image (from "selected" object, not "selected_image" string)
                        selected_obj = seg_data.get("selected", {})
                        seg.selected_image = selected_obj.get("path", "")
                        
                        # Restore ranking debug info (build from ranked list)
                        seg.ranking_debug = {
                            "top3": ranked_list[:3] if ranked_list else []
                        }
                        
                        logger.info(f"[OK] Loaded segment_{i}: {len(seg.images)} images, selected: {seg.selected_image}")
                    else:
                        logger.warning(f"[WARNING] No ranked data found for segment_{i}")
                        seg.images = []
                        seg.selected_image = ""
                
                logger.info("[SUCCESS] Successfully loaded all segments from ranked manifest")
                
            except Exception as e:
                logger.error(f"Error loading ranked manifest: {e}")
                logger.info("Falling back to full pipeline...")
                should_skip = False  # Force full pipeline on error
        
        if not should_skip:
            # 1.5. Refine initial queries with cinematic director (ASYNC)
            logger.info("Refining initial queries with cinematic director...")
            segments = await refine_initial_queries(segments, script_context)
            
            # 2. Scrape images (ASYNC)
            logger.info("Collecting images asynchronously...")
            segments = await collect_images_for_segments(segments, temp_dir=story_temp_dir)

            # 3. Rank and select best images (SYNC)
            logger.info("Ranking images with CLIP...")
            segments = rank_images(segments, temp_dir=story_temp_dir)

            # 3.5. Director Agent supervision (ASYNC)
            logger.info("Starting Director Agent supervision...")
            segments = await supervise_segments(segments)
        else:
            logger.info("[SKIP] Skipped scraping, ranking, and director supervision (using cached results)")

        # 4. Apply post-processing effects (SYNC)
        segments = apply_effects(segments, audio_file)

        # 5. Add subtitle overlays (SYNC)
        segments = overlay_subtitles(segments, audio_file)

        # 6. Render final video (SYNC)
        # Extract output name from input JSON filename
        output_name = Path(input_json_path).stem  # e.g., "mia_story" or "lost_labyrinth_script"
        output_path = render_final_video(segments, audio_file, output_name=output_name)

        logger.info(f"Video generation complete: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return None


def generate_video(input_json_path: str, audio_file: str) -> Optional[str]:
    """
    Sync wrapper around async pipeline for convenience.
    """
    return asyncio.run(async_generate_video(input_json_path, audio_file))


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='AI Video Generator - Create videos from scripts with AI-selected imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Auto-detect files in data/input/
  python main.py mia_story                          # Use mia_story.json and mia_story.mp3
  python main.py lost_labyrinth_script              # Use lost_labyrinth_script.json and .mp3
  python main.py script.json audio.mp3              # Explicit file names
        """
    )
    
    parser.add_argument(
        'script',
        nargs='?',
        help='Script JSON file (with or without .json extension) or base name'
    )
    
    parser.add_argument(
        'audio',
        nargs='?',
        help='Audio MP3 file (with or without .mp3 extension)'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    input_file = None
    audio_file = None
    
    if args.script:
        # User provided script name
        script_path = Path(INPUT_DIR) / args.script
        
        # Try with .json extension if not present
        if not script_path.suffix:
            script_path = script_path.with_suffix('.json')
        
        if not script_path.exists():
            print(f"Error: Script file not found: {script_path}")
            sys.exit(1)
        
        input_file = str(script_path)
        
        # If audio not provided, try to find matching audio file
        if not args.audio:
            # Try same base name with .mp3
            audio_path = script_path.with_suffix('.mp3')
            if audio_path.exists():
                audio_file = str(audio_path)
            else:
                # Fall back to ElevenLabs file
                elevenlabs_files = list(Path(INPUT_DIR).glob("ElevenLabs*.mp3"))
                if elevenlabs_files:
                    audio_file = str(elevenlabs_files[0])
                    print(f"Warning: No matching audio, using fallback: {audio_file}")
                else:
                    print(f"Error: No audio file found for {script_path.name}")
                    sys.exit(1)
        else:
            # User provided audio
            audio_path = Path(INPUT_DIR) / args.audio
            if not audio_path.suffix:
                audio_path = audio_path.with_suffix('.mp3')
            
            if not audio_path.exists():
                print(f"Error: Audio file not found: {audio_path}")
                sys.exit(1)
            
            audio_file = str(audio_path)
    else:
        # Auto-detect files (original behavior)
        json_files = glob(os.path.join(INPUT_DIR, "*.json"))
        if not json_files:
            print(f"Error: No JSON script found in {INPUT_DIR}")
            print("Usage: python main.py <script_name> [audio_name]")
            sys.exit(1)
        input_file = json_files[0]

        audio_files = glob(os.path.join(INPUT_DIR, "*.mp3"))
        if not audio_files:
            print(f"Error: No audio file found in {INPUT_DIR}")
            sys.exit(1)
        audio_file = audio_files[0]

    print(f"Using script: {input_file}")
    print(f"Using audio: {audio_file}")

    output_path = generate_video(input_file, audio_file)
    if output_path:
        print(f"Video generated successfully: {output_path}")
    else:
        print("Video generation failed")

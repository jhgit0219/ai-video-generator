"""
Benchmark parallel rendering approaches: disk, pipe, chunk
Run with SKIP_FLAG=True to isolate rendering performance
"""
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.parser import parse_input
from pipeline.renderer import render_final_video
from config import FPS, VIDEO_CODEC, PRESET
from utils.logger import setup_logger

logger = setup_logger(__name__)


def benchmark_rendering(script_path: str, audio_path: str):
    """Benchmark all three rendering methods."""

    # Parse script
    segments = parse_input(script_path)

    # Load ranked manifest (assumes SKIP_FLAG workflow already ran)
    from pathlib import Path
    import json

    story_name = Path(script_path).stem
    manifest_path = Path(f"data/temp_images/{story_name}/ranked_manifest.json")

    if not manifest_path.exists():
        logger.error(f"Ranked manifest not found: {manifest_path}")
        logger.error("Please run once with SKIP_FLAG=False to generate ranked manifest")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Attach selected images to segments
    for seg, entry in zip(segments, manifest):
        seg.selected_image = entry["selected_image"]

    logger.info(f"\n{'='*70}\nBENCHMARKING PARALLEL RENDERING\n{'='*70}")
    logger.info(f"Script: {script_path}")
    logger.info(f"Segments: {len(segments)}")
    logger.info(f"Total duration: {segments[-1].end_time:.1f}s")
    logger.info(f"FPS: {FPS}, Codec: {VIDEO_CODEC}, Preset: {PRESET}")

    results = {}
    methods = ["disk", "pipe", "chunk"]

    for method in methods:
        logger.info(f"\n{'='*70}\nBENCHMARKING: {method.upper()}\n{'='*70}")

        # Temporarily set parallel method
        import config
        original_method = config.PARALLEL_RENDER_METHOD
        config.PARALLEL_RENDER_METHOD = method

        output_path = f"data/output/benchmark_{method}_{story_name}.mp4"

        start = time.time()
        try:
            render_final_video(segments, audio_path, output_name=output_path)
            elapsed = time.time() - start
            results[method] = elapsed

            total_duration = segments[-1].end_time
            realtime_factor = total_duration / elapsed

            logger.info(f"[{method}] Complete: {elapsed:.2f}s ({realtime_factor:.2f}x realtime)")
        except Exception as e:
            logger.error(f"[{method}] FAILED: {e}")
            results[method] = None
        finally:
            config.PARALLEL_RENDER_METHOD = original_method

    # Print comparison
    logger.info(f"\n{'='*70}\nRESULTS\n{'='*70}")
    logger.info(f"{'Method':<15} {'Time (s)':<12} {'Speedup':<12} {'Realtime Factor'}")
    logger.info("-" * 70)

    baseline = results.get("disk")
    total_duration = segments[-1].end_time

    for method in sorted(results.keys(), key=lambda m: results[m] if results[m] else float('inf')):
        elapsed = results[method]
        if elapsed is None:
            logger.info(f"{method:<15} {'FAILED':<12}")
        else:
            speedup = baseline / elapsed if baseline else 1.0
            realtime = total_duration / elapsed
            logger.info(f"{method:<15} {elapsed:<12.2f} {speedup:<12.2f}x {realtime:.2f}x")

    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark parallel rendering")
    parser.add_argument("--script", default="data/input/mia_story.json", help="Script JSON file")
    parser.add_argument("--audio", default="data/input/mia_story.mp3", help="Audio file")

    args = parser.parse_args()

    benchmark_rendering(args.script, args.audio)

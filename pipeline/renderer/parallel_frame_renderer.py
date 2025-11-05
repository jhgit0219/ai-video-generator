"""
Parallel frame rendering for multi-threaded video encoding.
Bypasses MoviePy's single-threaded write_videofile() bottleneck.
"""
import os
import io
import time
import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method, get_start_method
from functools import partial
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np

from utils.logger import setup_logger
from config import TEMP_IMAGES_DIR, MAX_WORKERS, THREADS_PER_WORKER

logger = setup_logger(__name__)

TEMP_FRAMES_DIR = Path(TEMP_IMAGES_DIR) / "frame_sequence"

# Worker initialization function to set up Python path in each worker process
def _init_worker():
    """Initialize worker process with correct Python path for imports."""
    import sys
    from pathlib import Path as PathLib

    # Add project root to sys.path
    current_file = PathLib(__file__).resolve()
    project_root = current_file.parent.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Add venv site-packages for Windows multiprocessing spawn
    venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
        sys.path.insert(0, str(venv_site_packages))


def render_video_parallel(
    clip,
    output_path: str,
    audio_file: Optional[str] = None,
    method: str = "pipe",
    fps: int = 30,
    codec: str = "libx264",
    preset: str = "medium",
    workers: Optional[int] = None,
    quality: int = 95,
    chunk_duration: int = 10,
) -> str:
    """
    Render video using parallel frame generation.

    Args:
        clip: MoviePy VideoClip with effects applied
        output_path: Output video file path
        audio_file: Optional audio file to mux
        method: "disk", "pipe", or "chunk"
        fps: Frames per second
        codec: Video codec (libx264 or h264_nvenc)
        preset: Encoding preset
        workers: Number of worker processes (None = auto)
        quality: JPEG quality for disk/pipe methods
        chunk_duration: Chunk duration in seconds for chunk method

    Returns:
        Output path
    """
    if workers is None:
        workers = min(cpu_count(), MAX_WORKERS)
    else:
        workers = min(workers, MAX_WORKERS)

    logger.info(f"[parallel_render] Method: {method}, Workers: {workers}/{cpu_count()} cores, FPS: {fps}")

    try:
        if method == "disk":
            return _render_disk_pipeline(clip, output_path, audio_file, fps, codec, preset, workers, quality)
        elif method == "pipe":
            return _render_pipe_pipeline(clip, output_path, audio_file, fps, codec, preset, workers, quality)
        elif method == "chunk":
            return _render_chunk_pipeline(clip, output_path, audio_file, fps, codec, preset, workers, chunk_duration)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        logger.error(f"[parallel_render] Failed: {e}")
        raise


# ============================================================================
# APPROACH A: DISK-BASED
# ============================================================================

def _render_single_frame_disk(frame_idx: int, clip, output_dir: Path, fps: int, quality: int):
    """Render one frame to disk."""
    t = frame_idx / fps
    frame = clip.get_frame(t)

    img = Image.fromarray(frame)
    frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
    img.save(frame_path, format='JPEG', quality=quality, optimize=True)

    return frame_idx


def _render_disk_pipeline(clip, output_path, audio_file, fps, codec, preset, workers, quality):
    """Render frames to disk, then encode with FFMPEG."""
    frames_dir = TEMP_FRAMES_DIR / f"disk_{int(time.time())}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_frames = int(clip.duration * fps)
        logger.info(f"[disk] Rendering {total_frames} frames to {frames_dir}")

        # Render frames in parallel
        render_func = partial(_render_single_frame_disk, clip=clip, output_dir=frames_dir, fps=fps, quality=quality)

        with Pool(workers) as pool:
            for i, _ in enumerate(pool.imap(render_func, range(total_frames)), 1):
                if i % 30 == 0:
                    logger.info(f"[disk] {i}/{total_frames} frames")

        logger.info(f"[disk] Encoding from disk")
        _encode_from_disk(frames_dir, output_path, fps, codec, preset, audio_file)

        return output_path
    finally:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)


def _encode_from_disk(frames_dir: Path, output_path: str, fps: int, codec: str, preset: str, audio_file: Optional[str]):
    """Encode JPEG sequence from disk to MP4."""
    frame_pattern = str(frames_dir / "frame_%06d.jpg")

    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", frame_pattern]

    if audio_file and os.path.exists(audio_file):
        cmd.extend(["-i", audio_file, "-c:a", "aac", "-shortest"])

    if codec == "h264_nvenc":
        cmd.extend(["-c:v", "h264_nvenc", "-preset", preset, "-b:v", "10M"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", preset, "-crf", "18"])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFMPEG failed: {result.stderr}")


# ============================================================================
# APPROACH B: PURE PIPE
# ============================================================================

def _render_single_frame_pipe(frame_idx: int, clip, fps: int, quality: int):
    """Render one frame to JPEG bytes."""
    t = frame_idx / fps
    frame = clip.get_frame(t)

    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, optimize=True)

    return buf.getvalue()


def _render_pipe_pipeline(clip, output_path, audio_file, fps, codec, preset, workers, quality):
    """Render frames in parallel, pipe directly to FFMPEG."""
    total_frames = int(clip.duration * fps)
    logger.info(f"[pipe] Rendering {total_frames} frames via pipe")

    # Start FFMPEG process
    cmd = ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-"]

    if audio_file and os.path.exists(audio_file):
        cmd.extend(["-i", audio_file, "-c:a", "aac", "-shortest"])

    if codec == "h264_nvenc":
        cmd.extend(["-c:v", "h264_nvenc", "-preset", preset, "-b:v", "10M"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", preset, "-crf", "18"])

    cmd.append(str(output_path))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Render frames in parallel and pipe to FFMPEG
    render_func = partial(_render_single_frame_pipe, clip=clip, fps=fps, quality=quality)

    try:
        with Pool(workers) as pool:
            for i, jpeg_bytes in enumerate(pool.imap(render_func, range(total_frames)), 1):
                proc.stdin.write(jpeg_bytes)
                if i % 30 == 0:
                    logger.info(f"[pipe] {i}/{total_frames} frames")

        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"FFMPEG failed: {stderr}")

        logger.info(f"[pipe] Complete")
        return output_path
    except Exception as e:
        proc.kill()
        raise


# ============================================================================
# APPROACH C: CHUNKED MEZZANINE
# ============================================================================

def _render_chunk(chunk_idx: int, clip, chunk_start: float, chunk_end: float, output_dir: Path, fps: int, codec: str, preset: str):
    """Render one chunk to intermediate codec."""
    chunk_path = output_dir / f"chunk_{chunk_idx:04d}.mp4"

    # Extract subclip
    chunk_clip = clip.subclipped(chunk_start, chunk_end)

    # Use MoviePy to render chunk (single-threaded per chunk, but chunks are parallel)
    chunk_clip.write_videofile(
        str(chunk_path),
        codec=codec,
        preset=preset,
        fps=fps,
        audio=False,
        verbose=False,
        logger=None
    )

    return str(chunk_path)


def _render_chunk_pipeline(clip, output_path, audio_file, fps, codec, preset, workers, chunk_duration):
    """Render video in chunks, concatenate with FFMPEG."""
    chunks_dir = TEMP_FRAMES_DIR / f"chunks_{int(time.time())}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_duration = clip.duration
        num_chunks = int(np.ceil(total_duration / chunk_duration))

        logger.info(f"[chunk] Rendering {num_chunks} chunks ({chunk_duration}s each)")

        # Create chunk tasks
        chunk_tasks = []
        for i in range(num_chunks):
            start = i * chunk_duration
            end = min((i + 1) * chunk_duration, total_duration)
            chunk_tasks.append((i, start, end))

        # Render chunks in parallel
        render_func = partial(_render_chunk, clip=clip, output_dir=chunks_dir, fps=fps, codec=codec, preset=preset)

        chunk_paths = []
        with Pool(workers) as pool:
            for path in pool.starmap(render_func, [(i, s, e) for i, s, e in chunk_tasks]):
                chunk_paths.append(path)

        # Concatenate chunks
        logger.info(f"[chunk] Concatenating {len(chunk_paths)} chunks")
        _concatenate_chunks(chunk_paths, output_path, audio_file)

        return output_path
    finally:
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)


def _concatenate_chunks(chunk_paths: list, output_path: str, audio_file: Optional[str]):
    """
    Concatenate video chunks using FFMPEG concat demuxer with stream copy.

    This uses `-c:v copy` which is FAST (no re-encoding) - should take <1 second
    for most videos. If slow, check that all chunks have identical codec/settings.
    """
    # Get ffmpeg executable (try imageio-ffmpeg first, then system ffmpeg)
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        ffmpeg_exe = "ffmpeg"  # Fall back to system ffmpeg

    concat_file = Path(chunk_paths[0]).parent / "concat_list.txt"

    logger.info(f"[concat] Creating concat list with {len(chunk_paths)} chunks")

    # Verify all chunks exist and log their sizes
    total_size_mb = 0
    for i, path in enumerate(chunk_paths):
        if not os.path.exists(path):
            logger.error(f"[concat] Missing chunk {i}: {path}")
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            total_size_mb += size_mb
            logger.debug(f"[concat] Chunk {i}: {size_mb:.1f} MB")

    logger.info(f"[concat] Total input size: {total_size_mb:.1f} MB")

    # Write absolute paths to concat file (safer for Windows)
    with open(concat_file, 'w') as f:
        for path in chunk_paths:
            # Convert to absolute path and use forward slashes for ffmpeg
            abs_path = str(Path(path).absolute()).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    # Build command - use stream copy for FAST concat (no re-encoding)
    cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file)]

    if audio_file and os.path.exists(audio_file):
        logger.info(f"[concat] Adding audio track: {audio_file}")
        cmd.extend(["-i", audio_file, "-c:a", "aac", "-shortest"])

    # CRITICAL: -c:v copy = stream copy (no re-encoding, should be <1s for most videos)
    cmd.extend(["-c:v", "copy", str(output_path)])

    logger.info(f"[concat] Running FFMPEG concat (stream copy mode, should be fast)")
    logger.debug(f"[concat] Command: {' '.join(cmd)}")

    concat_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_time = time.time() - concat_start

    if result.returncode != 0:
        logger.error(f"[concat] FFMPEG failed after {concat_time:.2f}s")
        logger.error(f"[concat] stderr: {result.stderr}")
        raise RuntimeError(f"FFMPEG concat failed: {result.stderr}")

    # Check output size
    if os.path.exists(output_path):
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"[concat] ✅ Concat completed in {concat_time:.2f}s ({len(chunk_paths)} chunks)")
        logger.info(f"[concat] Output size: {output_size_mb:.1f} MB")

        # Warn if concat took too long (should be <2s for stream copy)
        if concat_time > 5.0:
            logger.warning(f"[concat] ⚠️  Concat took {concat_time:.1f}s - expected <2s with stream copy!")
            logger.warning(f"[concat] This suggests FFMPEG may be re-encoding instead of stream copying.")
            logger.warning(f"[concat] Check that all chunks use identical codec/settings (libx264 ultrafast)")
    else:
        logger.error(f"[concat] Output file not created: {output_path}")


# ============================================================================
# APPROACH D: TRUE PARALLEL (Data-based reconstruction)
# ============================================================================

def _create_segment_batches(segments: List[Dict[str, Any]], target_batch_duration: float = 20.0) -> List[List[Dict[str, Any]]]:
    """
    Group segments into batches respecting segment boundaries.

    This ensures effects and transitions render cleanly without mid-segment cuts.

    Args:
        segments: List of segment dicts
        target_batch_duration: Target duration per batch in seconds

    Returns:
        List of segment batches
    """
    batches = []
    current_batch = []
    current_duration = 0.0

    for seg in segments:
        seg_duration = seg['end_time'] - seg['start_time']

        # Start new batch if adding this segment exceeds target (and we have segments)
        if current_duration + seg_duration > target_batch_duration and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_duration = 0.0

        current_batch.append(seg)
        current_duration += seg_duration

    # Add final batch
    if current_batch:
        batches.append(current_batch)

    return batches


def render_video_parallel_v2(
    segments: List[Dict[str, Any]],
    output_path: str,
    audio_file: Optional[str] = None,
    fps: int = 30,
    codec: str = "libx264",
    preset: str = "medium",
    workers: Optional[int] = None,
    chunk_duration: int = 20,
) -> str:
    """
    Render video using TRUE parallel segment-based rendering.

    Each worker renders a batch of complete segments (no mid-segment cuts),
    avoiding boundary issues with effects and transitions.

    Args:
        segments: List of segment dicts with {image_path, start_time, end_time, effects_plan, ...}
        output_path: Output video file path
        audio_file: Optional audio file to mux
        fps: Frames per second
        codec: Video codec
        preset: Encoding preset
        workers: Number of worker processes (None = auto)
        chunk_duration: Target duration per batch (segments grouped to approximate this)

    Returns:
        Output path
    """
    if workers is None:
        workers = min(cpu_count(), MAX_WORKERS)
    else:
        workers = min(workers, MAX_WORKERS)

    # Calculate total video duration from segments
    total_duration = segments[-1]['end_time'] if segments else 0.0

    # Create segment-based batches (respects boundaries!)
    segment_batches = _create_segment_batches(segments, target_batch_duration=chunk_duration)
    num_chunks = len(segment_batches)

    logger.info(f"[parallel_v2] Rendering {num_chunks} segment batches (~{chunk_duration}s each) with {workers} workers")
    logger.info(f"[parallel_v2] Total video duration: {total_duration:.1f}s")
    logger.info(f"[parallel_v2] Batch sizes: {[len(batch) for batch in segment_batches]} segments")
    logger.info(f"[parallel_v2] Expected parallel rounds: {(num_chunks + workers - 1) // workers} (with {workers} workers)")
    logger.info(f"[parallel_v2] If you see batches completing one-by-one, workers are rendering sequentially (BAD!)")

    # Start timing
    start_time = time.time()

    chunks_dir = TEMP_FRAMES_DIR / f"chunks_v2_{int(time.time())}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Set PYTHONPATH environment variable so worker processes can find modules
    import os as os_module
    current_pythonpath = os_module.environ.get('PYTHONPATH', '')
    project_root = Path(__file__).resolve().parent.parent.parent
    new_pythonpath = str(project_root)
    if current_pythonpath:
        new_pythonpath = f"{new_pythonpath}{os_module.pathsep}{current_pythonpath}"
    os_module.environ['PYTHONPATH'] = new_pythonpath

    logger.info(f"[parallel_v2] Set PYTHONPATH={new_pythonpath}")

    try:
        # Create batch tasks (segment-based, not time-based!)
        batch_tasks = []
        for i, batch in enumerate(segment_batches):
            batch_tasks.append((i, batch))

        # Render batches in parallel
        render_func = partial(
            _render_segment_batch,
            output_dir=chunks_dir,
            fps=fps,
            codec=codec,
            preset=preset
        )

        chunk_paths = []
        render_start = time.time()
        with Pool(workers, initializer=_init_worker) as pool:
            # Use imap_unordered to get results as they complete (truly non-blocking)
            # This allows monitoring progress while workers run in parallel
            results = pool.starmap_async(render_func, batch_tasks)

            # Wait for completion and collect paths
            chunk_paths = results.get()

        logger.info(f"[parallel_v2] All {num_chunks} batches completed")

        render_time = time.time() - render_start
        logger.info(f"[parallel_v2] ⚡ Rendered {num_chunks} batches in {render_time:.2f}s ({render_time/num_chunks:.2f}s per batch)")

        # Concatenate chunks
        concat_start = time.time()
        logger.info(f"[parallel_v2] Concatenating {len(chunk_paths)} chunks")
        _concatenate_chunks(chunk_paths, output_path, audio_file)
        concat_time = time.time() - concat_start
        logger.info(f"[parallel_v2] ⚡ Concatenation completed in {concat_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"[parallel_v2] ✅ TOTAL TIME: {total_time:.2f}s (rendering={render_time:.2f}s, concat={concat_time:.2f}s)")
        logger.info(f"[parallel_v2] ⚡ Speedup: {workers}x workers processing {total_duration:.1f}s of video in {render_time:.2f}s")

        return output_path
    except Exception as e:
        logger.error(f"[parallel_v2] Rendering failed: {e}")
        # Clean up partial output if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"[parallel_v2] Cleaned up partial output: {output_path}")
            except Exception as cleanup_err:
                logger.warning(f"[parallel_v2] Failed to clean up partial output: {cleanup_err}")
        raise
    finally:
        # Always clean up temporary chunk directory
        if chunks_dir.exists():
            try:
                shutil.rmtree(chunks_dir)
                logger.info(f"[parallel_v2] Cleaned up temporary chunks: {chunks_dir}")
            except Exception as cleanup_err:
                logger.warning(f"[parallel_v2] Failed to clean up chunks dir: {cleanup_err}")


def _render_segment_batch(
    batch_idx: int,
    segment_batch: List[Dict[str, Any]],
    output_dir: Path,
    fps: int,
    codec: str,
    preset: str
) -> str:
    """
    Render a batch of complete segments (no mid-segment cuts).

    This avoids boundary issues with effects and transitions by rendering
    complete segments from start to end.

    Args:
        batch_idx: Batch index for file naming
        segment_batch: List of segment dicts to render
        output_dir: Output directory for batch file
        fps: Frames per second
        codec: Video codec (use libx264 for parallel workers!)
        preset: Encoding preset

    Returns:
        Path to rendered batch file
    """
    import sys
    import os
    import logging
    import numpy as np
    from pathlib import Path as PathLib
    from PIL import Image as PILImage
    from moviepy import ImageClip, concatenate_videoclips
    from pipeline.renderer.video_generator import CinematicEffectsAgent

    logger = logging.getLogger(__name__)
    batch_path = output_dir / f"batch_{batch_idx:04d}.mp4"

    import time as time_module
    batch_start = time_module.time()
    logger.info(f"[batch_{batch_idx}] START rendering {len(segment_batch)} segments")

    # Render each complete segment (no slicing!)
    segment_clips = []
    effects_agent = CinematicEffectsAgent()

    for i, seg in enumerate(segment_batch):
        seg_duration = seg['end_time'] - seg['start_time']

        # Deserialize pre-computed subject data (convert lists back to numpy arrays)
        if seg.get('precomputed_subject_data'):
            precomp = seg['precomputed_subject_data']
            if precomp.get('mask') is not None and isinstance(precomp['mask'], list):
                # Convert mask list back to numpy array
                mask_list = precomp['mask']
                mask_shape = precomp.get('mask_shape')
                if mask_shape:
                    precomp['mask'] = np.array(mask_list, dtype=np.uint8).reshape(mask_shape)
                else:
                    precomp['mask'] = np.array(mask_list, dtype=np.uint8)
                logger.debug(f"[batch_{batch_idx}] Segment {i}: Deserialized precomputed mask (shape={precomp['mask'].shape})")

        # Load image and create clip
        img_path = seg.get('selected_image') or seg.get('image_path')
        if not img_path or not os.path.exists(img_path):
            # Use black frame as fallback
            black = np.zeros((1080, 1920, 3), dtype=np.uint8)
            seg_clip = ImageClip(black).with_duration(seg_duration)
            logger.warning(f"[batch_{batch_idx}] Segment {i}: Missing image, using black placeholder")
        else:
            img = PILImage.open(img_path).convert('RGB')
            img_array = np.array(img)
            seg_clip = ImageClip(img_array).with_duration(seg_duration)

            # Apply effects if plan exists
            if seg.get('effects_plan'):
                try:
                    logger.debug(f"[batch_{batch_idx}] Segment {i}: Applying effects plan")
                    # Pass segment dict to apply_plan so it can access precomputed_subject_data
                    seg_clip = effects_agent.apply_plan(seg_clip, seg['effects_plan'], segment=seg)
                except Exception as e:
                    import traceback
                    logger.error(f"[batch_{batch_idx}] Segment {i}: Effects failed: {e}")
                    logger.error(f"[batch_{batch_idx}] Traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[batch_{batch_idx}] Segment {i}: No effects_plan")

            # CRITICAL: Apply zoom-to-cover to ensure consistent frame size (1920x1080)
            if seg_clip.size != (1920, 1080):
                curr_w, curr_h = seg_clip.size
                target_w, target_h = 1920, 1080

                # Calculate scale to cover both dimensions
                scale_w = target_w / curr_w
                scale_h = target_h / curr_h
                scale = max(scale_w, scale_h)

                # Resize maintaining aspect ratio
                new_w = int(curr_w * scale)
                new_h = int(curr_h * scale)
                seg_clip = seg_clip.resized(width=new_w, height=new_h)

                # Crop to exact target size from center
                if seg_clip.size != (target_w, target_h):
                    x1 = (new_w - target_w) // 2
                    y1 = (new_h - target_h) // 2
                    x2 = x1 + target_w
                    y2 = y1 + target_h
                    seg_clip = seg_clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)

        segment_clips.append(seg_clip)

    # Concatenate all segment clips in this batch
    if len(segment_clips) == 1:
        batch_clip = segment_clips[0]
    else:
        batch_clip = concatenate_videoclips(segment_clips, method='compose')

    logger.info(f"[batch_{batch_idx}] Concatenated {len(segment_clips)} segments, duration={batch_clip.duration:.2f}s")

    # Render batch to file
    # CRITICAL: Use CPU codec (libx264) for parallel workers
    # GPU codecs (h264_nvenc) can't be shared across multiple processes due to CUDA context conflicts
    # Import config here to avoid circular imports in worker process
    from config import THREADS_PER_WORKER

    batch_clip.write_videofile(
        str(batch_path),
        codec=codec,  # Use provided codec (should be libx264 for workers)
        fps=fps,
        audio=False,
        threads=THREADS_PER_WORKER,  # Limit threads per worker to prevent system overload
        preset=preset
    )

    batch_time = time_module.time() - batch_start
    logger.info(f"[batch_{batch_idx}] FINISHED in {batch_time:.1f}s → {batch_path}")

    # CRITICAL: Clean up memory after rendering
    # MoviePy clips hold references to large numpy arrays and can cause memory leaks
    try:
        # Close and delete all clips
        if hasattr(batch_clip, 'close'):
            batch_clip.close()
        del batch_clip

        for clip in segment_clips:
            if hasattr(clip, 'close'):
                clip.close()
        del segment_clips

        # Clean up effects agent (may hold YOLO/CLIP models)
        del effects_agent

        # Force garbage collection
        import gc
        gc.collect()

        logger.debug(f"[batch_{batch_idx}] Memory cleanup completed")
    except Exception as cleanup_err:
        logger.warning(f"[batch_{batch_idx}] Memory cleanup warning: {cleanup_err}")

    return str(batch_path)

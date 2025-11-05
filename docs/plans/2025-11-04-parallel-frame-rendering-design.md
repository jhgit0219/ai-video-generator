# Parallel Frame Rendering Design

**Date**: 2025-11-04
**Problem**: MoviePy's `write_videofile()` is single-threaded (16% CPU on 16-thread system), resulting in 2.6 it/s rendering speed.
**Solution**: Pre-render frames in parallel, pipe to FFMPEG for encoding.

## Architecture

```
1. Upscale images (existing batch process)
   ↓
2. Build MoviePy clips with effects (existing)
   ↓
3. Parallel frame rendering (NEW)
   - Use multiprocessing.Pool with N workers
   - Each worker renders frames via clip.get_frame(t)
   - Output: JPEG bytes or disk files
   ↓
4. FFMPEG encoding (NEW)
   - Read frames from pipe or disk
   - Multi-threaded H.264/NVENC encoding
   ↓
5. Cleanup temp files
```

## Three Implementation Approaches

### A. Disk-based
- Render frames to temp JPEG files
- FFMPEG reads from disk
- **Pros**: Simple, guaranteed to work
- **Cons**: Disk I/O overhead

### B. Pure Pipe
- Use `pool.imap()` for ordered frame generation
- Pipe JPEG bytes directly to FFMPEG stdin
- **Pros**: Zero disk I/O, maximum throughput
- **Cons**: Requires careful buffer management

### C. Chunked Mezzanine
- Render chunks to ProRes/H.264 intermediate
- Concatenate with FFMPEG concat demuxer
- **Pros**: Handles very long videos well
- **Cons**: Most complex implementation

## Implementation Details

**New Module**: `pipeline/renderer/parallel_frame_renderer.py`
- `render_video_parallel(clip, output_path, method, **kwargs)` - Main entry point
- `_render_disk_pipeline()` - Approach A
- `_render_pipe_pipeline()` - Approach B
- `_render_chunk_pipeline()` - Approach C

**Config Settings** (`config.py`):
```python
PARALLEL_RENDER_METHOD = "disabled"  # "disk", "pipe", "chunk", or "disabled"
FRAME_RENDER_WORKERS = None  # Auto-detect CPU count
FRAME_SEQUENCE_QUALITY = 95  # JPEG quality
CHUNK_DURATION = 10  # Seconds per chunk (for chunk method)
```

**Integration** (`video_generator.py`):
```python
if PARALLEL_RENDER_METHOD != "disabled":
    render_video_parallel(final_video, output_path, method=PARALLEL_RENDER_METHOD, ...)
else:
    final_video.write_videofile(output_path, **write_params)  # Original
```

## Benchmarking Plan

Run all three approaches with `SKIP_FLAG=True` to isolate rendering performance:
1. Disk-based (baseline)
2. Pure pipe (zero I/O)
3. Chunked mezzanine

Measure: Total time, CPU utilization, throughput (fps)

## Error Handling

- Fallback to original MoviePy pipeline on failure
- Clean temp files in `finally` blocks
- Memory-bounded queues (max 100 frames buffered)
- Graceful degradation on worker crashes

## Expected Performance

- **Current**: 2.6 it/s (1 thread, 16% CPU)
- **Target**: 10-15 it/s (8-16 threads, 80-100% CPU)
- **Speedup**: ~4-6x faster rendering

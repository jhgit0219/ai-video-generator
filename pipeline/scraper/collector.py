import os
import json
import asyncio
from pathlib import Path
from typing import List
from pipeline.parser import VideoSegment
from utils.helpers import cleanup_temp_files, ensure_directory, cleanup_images_only
from .google_scraper import GoogleImagesScraper
from .utils import download_manifest_images  # <-- import the async downloader
from config import TEMP_IMAGES_DIR, MAX_IMAGES_PER_SEGMENT, MAX_CONCURRENT_SCRAPER, SKIP_FLAG
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize Path objects at module level
TEMP_IMAGES_PATH = Path(TEMP_IMAGES_DIR)
MANIFEST_PATH = TEMP_IMAGES_PATH / "image_manifest.json"

async def collect_images_for_segments(
    segments,
    limit_per_segment=MAX_IMAGES_PER_SEGMENT,
    max_concurrent=MAX_CONCURRENT_SCRAPER,
    auto_download=True,
    wipe_temp: bool = True,
    base_index: int | None = None,
    temp_dir: str | None = None,
):
    """Scrape or reload images for all segments depending on SKIP_FLAG.
    
    Args:
        temp_dir: Story-specific temp directory (e.g., data/temp_images/mia_story).
                  If None, uses TEMP_IMAGES_DIR from config.
    """
    # Use story-specific temp directory if provided, otherwise fall back to config
    working_temp_dir = temp_dir if temp_dir is not None else TEMP_IMAGES_DIR
    ensure_directory(working_temp_dir)
    manifest_path = Path(working_temp_dir) / "image_manifest.json"

    # -------------------------------
    # CASE 1: Skip scraping, reuse manifest
    # -------------------------------
    if SKIP_FLAG and manifest_path.exists():
        logger.info("SKIP_FLAG=True — skipping scraping, redownloading images from manifest.")
        try:
            # Step 1 - clear only existing images
            cleanup_images_only()

            # Step 2 - redownload everything listed in manifest
            await download_manifest_images(
                manifest_path=str(manifest_path),
                dest_dir=working_temp_dir,
                max_concurrent=5,
            )

            # Step 3 - populate segments with local image paths
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            for i, seg in enumerate(segments):
                seg_index = (base_index + i) if base_index is not None else i
                seg_key = f"segment_{seg_index}"
                if seg_key in manifest:
                    seg.images = [entry["path"] for entry in manifest[seg_key] if "path" in entry]
                    logger.info(f"Loaded {len(seg.images)} re-downloaded local images for {seg_key}")
                else:
                    seg.images = []
                    logger.warning(f"No data found for {seg_key} in manifest.")
            return segments

        except Exception as e:
            logger.error(f"Error re-downloading from manifest: {e}")
            logger.info("Falling back to full scrape mode...")

    # -------------------------------
    # CASE 2: Fresh scrape (SKIP_FLAG=False)
    # -------------------------------
    if not SKIP_FLAG:
        if wipe_temp:
            # Default behavior: wipe temp files to produce a fresh scrape
            cleanup_temp_files()
            logger.info("Wiping temp files for fresh scrape")
        else:
            # Incremental mode: preserve existing temp files/manifest and only update requested segments
            logger.info("Incremental scrape mode: preserving existing temp files/manifest")
        logger.info(f"SKIP_FLAG=${SKIP_FLAG}")
        scraper = GoogleImagesScraper()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_segment(i, segment):
            query = segment.visual_query or segment.transcript
            logger.info(f"Scraping images for segment {i}: '{query}'")
            async with semaphore:
                try:
                    results = await scraper.scrape(query, segment_id=i, max_images=limit_per_segment)
                    segment.images = [r.url for r in results]
                    logger.info(f"[{i}] Collected {len(segment.images)} URLs for '{query}'")
                except Exception as e:
                    logger.error(f"Error scraping '{query}': {e}")
                    segment.images = []
            return segment

        segments = await asyncio.gather(*(scrape_segment(i, s) for i, s in enumerate(segments)))

        # Save structured manifest
        working_temp_path = Path(working_temp_dir)
        manifest = {}
        for i, seg in enumerate(segments):
            seg_index = (base_index + i) if base_index is not None else i
            seg_name = f"segment_{seg_index}"
            seg_dir = working_temp_path / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)

            structured = []
            for j, url in enumerate(seg.images):
                filename = os.path.basename(url.split("?")[0]) or f"img_{j}.jpg"
                if "." not in filename:
                    filename += ".jpg"
                path = str((seg_dir / filename).resolve())

                structured.append({
                    "id": str(j),
                    "url": url,
                    "path": path
                })

            manifest[seg_name] = structured

        # Merge with existing manifest if present
        existing_manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    existing_manifest = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read existing manifest: {e}")

        # Update only segments we processed
        for seg_name, structured in manifest.items():
            existing_manifest[seg_name] = structured

        # Write merged manifest back
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(existing_manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Updated structured image manifest for {len(manifest)} segment(s)")


        logger.info(f"Saved structured image manifest to {manifest_path}")

        if auto_download:
            logger.info(f"Starting selective image download for {len(manifest)} updated segment(s)...")
            await download_manifest_images(
                manifest_path=str(manifest_path),
                dest_dir=working_temp_dir,
                max_concurrent=5,
                target_segments=list(manifest.keys()),  # <-- new param
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            for i, seg in enumerate(segments):
                seg_index = (base_index + i) if base_index is not None else i
                seg_key = f"segment_{seg_index}"
                if seg_key in manifest:
                    seg.images = [entry["path"] for entry in manifest[seg_key] if "path" in entry]
                    logger.info(f"Updated segment {seg_index} to use {len(seg.images)} local image paths")
                else:
                    seg.images = []
                    logger.warning(f"No image data found for {seg_key} in manifest after download")
        except Exception as e:
            logger.error(f"Failed to reload manifest paths into segments: {e}")

    return segments

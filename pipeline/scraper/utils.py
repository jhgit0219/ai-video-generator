import os
import aiofiles
import random
import asyncio
import json
import aiohttp
from aiohttp import ClientTimeout
from pathlib import Path
from playwright.async_api import async_playwright
from .proxy import next_proxy, proxies_enabled, playwright_proxy, aiohttp_proxy
from config import PROXY_APPLY_TO_PLAYWRIGHT
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def playwright_fallback(url: str, path: Path):
    """Playwright fallback that handles JS-based redirects and embedded figures."""
    try:
        async with async_playwright() as p:
            launch_kwargs = {"headless": True}
            if PROXY_APPLY_TO_PLAYWRIGHT and proxies_enabled():
                conf = playwright_proxy()
                if conf:
                    launch_kwargs["proxy"] = conf
            browser = await p.chromium.launch(**launch_kwargs)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            logger.info(f"Playwright navigating: {url}")
            resp = await page.goto(url, timeout=60000, wait_until="load")

            # JS redirect detection
            if page.url != url:
                logger.info(f"JS redirect detected: {url} → {page.url}")
                new_url = page.url
                async with aiohttp.ClientSession() as session:
                    async with session.get(new_url, timeout=30, proxy=aiohttp_proxy()) as r:
                        if r.status == 200:
                            async with aiofiles.open(path, "wb") as f:
                                await f.write(await r.read())
                            logger.info(f"Downloaded via JS redirect: {new_url}")
                            await browser.close()
                            return True

            # Meta refresh redirect
            try:
                meta_tag = await page.locator('meta[http-equiv="refresh"]').first
                meta_refresh = await meta_tag.get_attribute("content") if meta_tag else None
            except Exception:
                meta_refresh = None

            if meta_refresh and "url=" in meta_refresh.lower():
                new_url = meta_refresh.split("url=")[-1].strip()
                logger.info(f"Meta refresh redirect detected: {url} → {new_url}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(new_url, timeout=30, proxy=aiohttp_proxy()) as r:
                        if r.status == 200:
                            async with aiofiles.open(path, "wb") as f:
                                await f.write(await r.read())
                            logger.info(f"Downloaded via meta-refresh: {new_url}")
                            await browser.close()
                            return True

            # Extract first visible <img>
            await asyncio.sleep(3)
            img = await page.query_selector("img")
            if not img:
                for f in page.frames:
                    img = await f.query_selector("img")
                    if img:
                        break

            if img:
                src = await img.get_attribute("src")
                if src and src.startswith("http"):
                    logger.info(f"Extracted <img src> from page: {src}")
                    async with aiohttp.ClientSession() as session:
                        async with session.get(src, timeout=30, proxy=aiohttp_proxy()) as r:
                            if r.status == 200:
                                async with aiofiles.open(path, "wb") as f:
                                    await f.write(await r.read())
                                logger.info(f"Downloaded extracted image src: {src}")
                                await browser.close()
                                return True

            # Fallback: full-page screenshot
            await page.screenshot(path=path, full_page=True)
            logger.warning(f"Screenshot fallback used for {url}")
            await browser.close()
            return True

    except Exception as e:
        logger.error(f"Playwright fallback failed for {url}: {e}")
    return False


# -----------------------------
# Manifest-based batch downloader
# -----------------------------

async def download_manifest_images(manifest_path: str, dest_dir: str, max_concurrent: int = 5, target_segments: list[str] | None = None):
    """Download all images listed in a JSON manifest asynchronously."""

    if not os.path.exists(manifest_path):
        logger.warning(f"Manifest file not found: {manifest_path}")
        return

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read {manifest_path}: {e}")
        return

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        ),
        "Referer": "https://www.google.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    timeout = ClientTimeout(total=45, connect=10, sock_connect=10, sock_read=30)

    async def download_image(url: str, dest: Path, session: aiohttp.ClientSession, max_retries: int = 3):
        filename = dest / os.path.basename(url.split("?")[0] or f"img_{abs(hash(url))}.jpg")
        if not filename.suffix:
            filename = filename.with_suffix(".jpg")

        async with semaphore:
            for attempt in range(1, max_retries + 1):
                try:
                    proxy_url = next_proxy()
                    async with session.get(url, timeout=45, allow_redirects=True, proxy=aiohttp_proxy()) as resp:
                        if resp.status == 200:
                            async with aiofiles.open(filename, "wb") as f:
                                async for chunk in resp.content.iter_chunked(8192):
                                    await f.write(chunk)
                            logger.info(f"✅ Downloaded {filename.name}")
                            return

                        elif resp.status in (403, 429, 500, 503):
                            logger.warning(f"Retryable {resp.status} for {url} (attempt {attempt})")
                        else:
                            logger.warning(f"Bad status {resp.status} for {url}")
                            return

                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed for {url}: {e}")

                delay = min(2 ** attempt + random.uniform(0, 2), 10)
                logger.info(f"Retrying {url} in {delay:.1f}s...")
                await asyncio.sleep(delay)

            # If all attempts fail → use Playwright
            logger.error(f"aiohttp failed after {max_retries} attempts. Trying Playwright for {url}")
            await playwright_fallback(url, filename)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = []
        for seg_name, entries in manifest.items():
            # Skip segments not in target list (if provided)
            if target_segments and seg_name not in target_segments:
                continue

            seg_dir = Path(dest_dir) / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)

            # structured manifest support
            for entry in entries:
                url = entry["url"] if isinstance(entry, dict) else entry
                tasks.append(download_image(url, seg_dir, session))

        if not tasks:
            logger.info("No matching segments found for download (all skipped).")
            return

        await asyncio.gather(*tasks)

    logger.info("Manifest image download complete.")

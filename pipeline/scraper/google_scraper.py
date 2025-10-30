import os
import json
import random
import asyncio
import aiohttp
from pathlib import Path
from urllib.parse import urlparse
from typing import List
from playwright.async_api import async_playwright
from utils.logger import setup_logger
from config import (
    SEARCH_ENGINE_URL,
    MAX_SCRAPED_IMAGES,
    SCROLL_SLEEP,
    MAX_SCROLL_ROUNDS,
    PLAYWRIGHT_HEADFUL,
    TEMP_IMAGES_DIR,
    SKIP_FLAG,
    ENABLE_CLIP_FILTER,
    CLIP_RELEVANCE_THRESHOLD,
    THUMBNAIL_ATTEMPT_LIMIT,
    CAPTCHA_HANDLING,
    CAPTCHA_MAX_RETRIES,
    CAPTCHA_WAIT_SECONDS,
    CAPTCHA_RANDOMIZE_UA,
    PROXY_SCRAPER_BROWSING,
    SCRAPER_REQUERY_MAX_RETRIES,
    PROXY_LOG_EXTERNAL_IP,
)
from .models import ImageResult
from pipeline.ai_filter.semantic_filter import clip_text_relevance
from .proxy import next_proxy, proxies_enabled, playwright_proxy, get_proxy_identity


logger = setup_logger(__name__)


class GoogleImagesScraper:
    """Async Playwright-based Google Images scraper"""

    STOCK_DOMAINS = [
        "shutterstock.com",
        "gettyimages.com",
        "istockphoto.com",
        "adobe.com",
        "alamy.com",
        "dreamstime.com",
        "123rf.com",
        "depositphotos.com",
        "bigstockphoto.com",
        "freepik.com",
        "pexels.com",
        "pixabay.com",
        "canva.com",
        "facebook.com",
    ]

    SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".gif"]

    def __init__(self, headful: bool = PLAYWRIGHT_HEADFUL):
        self.headful = headful
        Path(TEMP_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    def _is_stock(self, url: str) -> bool:
        return any(d in url for d in self.STOCK_DOMAINS)

    async def _is_valid_image_url(self, url: str) -> bool:
        """Check if image URL is likely valid (extension or verified MIME type)."""
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path.lower())[1]

        # Fast check for common extensions
        if ext in self.SUPPORTED_EXTS:
            return True

        # Skip clearly invalid or embedded URLs
        if url.startswith("data:") or url.endswith("/"):
            return False

        # Fallback: HEAD check to confirm MIME type
        try:
            async with aiohttp.ClientSession() as session:
                proxy_url = next_proxy()
                async with session.head(url, timeout=5, allow_redirects=True, proxy=proxy_url) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    if resp.status == 200 and ctype.startswith("image/"):
                        logger.debug(f"MIME check passed for {url} ({ctype})")
                        return True
                    else:
                        logger.info(f"MIME check failed for {url} ({ctype})")
                        return False
        except Exception as e:
            logger.warning(f"HEAD check failed for {url}: {e}")
            return False

    async def _scroll_to_bottom(self, page):
        """Continuously scroll and click 'See more' when available until no new content loads."""
        logger.info("Scrolling to load all image thumbnails...")

        last_height = await page.evaluate("document.body.scrollHeight")
        stable_rounds = 0

        for i in range(MAX_SCROLL_ROUNDS):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(SCROLL_SLEEP)

            # Try clicking pagination buttons (handles both old and new layouts)
            try:
                # Try "See more anyway" first (newer Chrome layouts)
                if await page.locator("text=See more anyway").count() > 0:
                    await page.locator("text=See more anyway").first.click()
                    await asyncio.sleep(2)
                    logger.info("Clicked 'See more anyway' button.")
                # Try "See more" (common variant)
                elif await page.locator("text=See more").count() > 0:
                    await page.locator("text=See more").first.click()
                    await asyncio.sleep(2)
                    logger.info("Clicked 'See more' button.")
                # Try "More results" (older Chrome layouts)
                elif await page.locator("text=More results").count() > 0:
                    await page.locator("text=More results").first.click()
                    await asyncio.sleep(2)
                    logger.info("Clicked 'More results' button.")
            except Exception:
                pass

            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                stable_rounds += 1
                if stable_rounds >= 2:
                    logger.info("Scrolling stabilized — stopping.")
                    break
            else:
                stable_rounds = 0
                last_height = new_height

        logger.info("Finished scrolling through Google Images results.")


    async def _click_see_more(self, page) -> bool:
        """Click pagination control if present. Prefer 'See more*' or 'More results'. Returns True if clicked."""
        try:
            loc = page.locator("text=See more")
            if await loc.count() > 0:
                target = loc.first
                await target.scroll_into_view_if_needed()
                await asyncio.sleep(0.2)
                await target.click()
                logger.info("Clicked pagination control: text('See more*')")
                await asyncio.sleep(SCROLL_SLEEP)
                return True
        except Exception:
            pass
        try:
            loc2 = page.locator("text=More results")
            if await loc2.count() > 0:
                target = loc2.first
                await target.scroll_into_view_if_needed()
                await asyncio.sleep(0.2)
                await target.click()
                logger.info("Clicked pagination control: text('More results')")
                await asyncio.sleep(SCROLL_SLEEP)
                return True
        except Exception:
            pass
        return False

    async def _load_all_results(self, page):
        """
        Scrolls for a fixed number of rounds, optionally clicking 'See more' if found.
        Stops after MAX_SCROLL_ROUNDS or if content stabilizes.
        """
        logger.info("Loading all image thumbnails (bounded scroll)...")
        last_height = await page.evaluate("document.body.scrollHeight")
        stable_rounds = 0

        for i in range(int(MAX_SCROLL_ROUNDS or 8)):
            # Scroll a little, pause, wiggle to trigger lazy loads
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await asyncio.sleep(SCROLL_SLEEP)
            await page.mouse.wheel(0, random.randint(800, 1400))  # helps trigger lazy loading
            await asyncio.sleep(0.5)

            # Try optional "See more" buttons if they exist
            try:
                if await page.locator("text=See more anyway").count() > 0:
                    await page.locator("text=See more anyway").first.click()
                    await asyncio.sleep(1.5)
                    logger.info("Clicked 'See more anyway' button.")
                elif await page.locator("text=More results").count() > 0:
                    await page.locator("text=More results").first.click()
                    await asyncio.sleep(1.5)
                    logger.info("Clicked 'More results' button.")
            except Exception:
                pass

            # Check for height stabilization
            new_height = await page.evaluate("document.body.scrollHeight")
            if abs(new_height - last_height) < 200:  # tolerate small pixel jitter
                stable_rounds += 1
            else:
                stable_rounds = 0
            last_height = new_height

            if stable_rounds >= 2:
                logger.info(f"Scrolling stabilized after {i+1} rounds.")
                break

        logger.info(f"Scrolling completed — ran {i+1} rounds total.")


    async def _collect_preview_urls(self, page, max_images: int, query: str) -> List[dict]:
        """Collect image preview URLs by clicking on image elements, matching test.py logic."""
        results = []
        below_threshold_candidates = []  # Track images that didn't pass threshold

        # Wait for the images section to appear (matching test.py selector)
        await page.wait_for_selector('div[data-attrid="images universal"]', timeout=15000)

        # Select all image elements (matching test.py)
        image_elements = await page.query_selector_all('div[data-attrid="images universal"]')
        total_thumbs = len(image_elements)
        logger.info(f"Found {total_thumbs} image elements on the page.")

        if not image_elements:
            return []

        idx = 0
        success_count = 0
        attempts = 0

        # Process each image element (limit to THUMBNAIL_ATTEMPT_LIMIT)
        for idx, image_element in enumerate(image_elements):
            if success_count >= max_images:
                logger.info(f"Reached target of {max_images} images.")
                break
            
            # Stop after attempting THUMBNAIL_ATTEMPT_LIMIT thumbnails
            if attempts >= THUMBNAIL_ATTEMPT_LIMIT:
                logger.info(f"Reached thumbnail attempt limit ({THUMBNAIL_ATTEMPT_LIMIT}).")
                break
            
            attempts += 1

            try:
                logger.info(f"Processing image {idx + 1}/{total_thumbs}...")

                # Scroll element into view to ensure it's visible and clickable
                try:
                    await image_element.scroll_into_view_if_needed(timeout=3000)
                    await asyncio.sleep(0.3)
                except Exception:
                    logger.debug(f"Could not scroll element {idx + 1} into view")
                    # Continue anyway, it might still be clickable

                # Add random delay to reduce anti-bot detection
                await asyncio.sleep(random.uniform(0.3, 0.8))

                # Click the image element with a shorter timeout to fail fast
                try:
                    await image_element.click(timeout=5000)
                except Exception as e:
                    logger.warning(f"Could not click element {idx + 1} (not visible/clickable), skipping")
                    continue
                
                # Wait a bit for the preview to load
                await asyncio.sleep(0.8)
                
                # Try to get the large preview image with retry logic
                img_tag = None
                for attempt in range(2):  # Try twice
                    try:
                        img_tag = await page.wait_for_selector(
                            "img.sFlh5c.FyHeAf.iPVvYb[jsaction]", 
                            timeout=3000,
                            state="attached"
                        )
                        if img_tag:
                            break
                    except Exception:
                        if attempt == 0:
                            # Try clicking again if first attempt failed
                            await asyncio.sleep(0.5)
                            try:
                                await image_element.click(timeout=3000)
                            except Exception:
                                pass
                            await asyncio.sleep(0.5)
                        
                if not img_tag:
                    # Fallback: try query_selector directly
                    img_tag = await page.query_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]")
                if not img_tag:
                    logger.warning(f"No img tag found for index {idx + 1}")
                    continue

                img_url = await img_tag.get_attribute("src")
                if not img_url or img_url.startswith("data:") or self._is_stock(img_url):
                    continue

                # Validate supported extension or MIME type before saving
                if not await self._is_valid_image_url(img_url):
                    logger.info(f"Skipping unsupported or invalid image: {img_url}")
                    continue

                image_description = await img_tag.get_attribute("alt") or "N/A"
                source_el = await page.query_selector('(//div[@jsname="figiqf"]/a[@class="YsLeY"])[2]')
                source_url = await source_el.get_attribute("href") if source_el else "N/A"
                source_name = urlparse(source_url).netloc.replace("www.", "") if source_url != "N/A" else "N/A"

                # CLIP relevance filter (uses refined cinematic query for semantic matching)
                if ENABLE_CLIP_FILTER:
                    logger.debug(f"Checking CLIP relevance for: '{image_description[:80]}'")
                    logger.debug(f"Against refined query: '{query}'")
                    is_relevant, score = clip_text_relevance(query, image_description, return_score=True)
                    logger.debug(f"CLIP score: {score:.3f} (threshold: {CLIP_RELEVANCE_THRESHOLD})")
                    if not is_relevant:
                        logger.info(
                            f"Filtered out low CLIP relevance (score={score:.3f} < {CLIP_RELEVANCE_THRESHOLD}): {image_description[:80]}"
                        )
                        # Store as below-threshold candidate for fallback
                        below_threshold_candidates.append({
                            "url": img_url,
                            "description": image_description,
                            "source_url": source_url,
                            "source_name": source_name,
                            "clip_score": score
                        })
                        continue
                else:
                    logger.debug("CLIP filter disabled, accepting image")

                results.append({
                    "url": img_url,
                    "description": image_description,
                    "source_url": source_url,
                    "source_name": source_name,
                })
                success_count += 1

                logger.info(f"[{success_count}] {source_name} — {img_url[:100]}")

            except Exception as e:
                logger.warning(f"Error processing image {idx + 1}: {e}")
                continue

        # Fallback: if no images passed threshold, take top 5 by CLIP score
        if len(results) == 0 and len(below_threshold_candidates) > 0:
            logger.warning(f"No images met threshold ({CLIP_RELEVANCE_THRESHOLD}). Taking top {min(5, len(below_threshold_candidates))} by CLIP score as fallback.")
            # Sort by CLIP score descending and take top 5
            below_threshold_candidates.sort(key=lambda x: x["clip_score"], reverse=True)
            top_candidates = below_threshold_candidates[:min(5, len(below_threshold_candidates))]
            for candidate in top_candidates:
                # Remove clip_score before adding to results (not needed in final output)
                score = candidate.pop("clip_score")
                results.append(candidate)
                logger.info(f"[Fallback] {candidate['source_name']} (score={score:.3f}) — {candidate['url'][:100]}")

        logger.info(f"Collected {len(results)} successful images out of {attempts} thumbnails attempted.")
        return results

    def _build_exclusion_query(self, query: str) -> str:
        """Append -site: filters for stock domains using the existing STOCK_DOMAINS."""
        exclusions = " ".join(f"-site:{domain}" for domain in self.STOCK_DOMAINS)
        return f"{query} {exclusions}"

    async def scrape(self, query: str, segment_id: int | None = None, max_images: int | None = None) -> List[ImageResult]:
        manifest_path = Path(TEMP_IMAGES_DIR) / "image_manifest.json"
        filtered_query = self._build_exclusion_query(query)
        url = SEARCH_ENGINE_URL + filtered_query.replace(" ", "+")
        results: List[ImageResult] = []

        async with async_playwright() as p:
            launch_kwargs = {"headless": not self.headful}
            if PROXY_SCRAPER_BROWSING and proxies_enabled():
                conf = playwright_proxy()
                if conf:
                    launch_kwargs["proxy"] = conf
            browser = await p.chromium.launch(**launch_kwargs)

            async def new_context_and_page():
                ua = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0 Safari/537.36"
                ) if not CAPTCHA_RANDOMIZE_UA else self._random_user_agent()
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=ua,
                )
                return context, await context.new_page()

            context, page = await new_context_and_page()

            # Create watcher task with proper cancellation (manual mode only)
            watcher_task = None
            if str(CAPTCHA_HANDLING).lower() == "manual":
                watcher_task = asyncio.create_task(self._watch_for_recaptcha(page, browser))

            try:
                # Optional: log external IP via proxy once per scrape session
                if PROXY_LOG_EXTERNAL_IP and proxies_enabled():
                    iptxt = await get_proxy_identity()
                    if iptxt:
                        logger.info(f"[proxy] External identity via proxy: {iptxt}")
                await page.goto(url, timeout=60000)
                await self._accept_consent(page)

                # Lightweight captcha detection and handling based on config
                if await self._has_captcha(page):
                    mode = str(CAPTCHA_HANDLING).lower()
                    if mode == "skip":
                        logger.warning("Captcha detected; skipping this query due to CAPTCHA_HANDLING=skip")
                        return []
                    elif mode == "retry":
                        attempt = 0
                        while attempt < int(CAPTCHA_MAX_RETRIES):
                            attempt += 1
                            logger.info(f"Captcha detected; retrying with new context (attempt {attempt}/{CAPTCHA_MAX_RETRIES})")
                            try:
                                await page.close()
                            except Exception:
                                pass
                            try:
                                await context.close()
                            except Exception:
                                pass
                            context, page = await new_context_and_page()
                            await page.goto(url, timeout=60000)
                            await self._accept_consent(page)
                            if not await self._has_captcha(page):
                                break
                        if await self._has_captcha(page):
                            logger.warning("Captcha persists after retries; skipping this query")
                            return []
                    else:
                        logger.info("Captcha detected; proceeding with manual handling via watcher")

                # Define local_max early for use in collection
                local_max = max_images or MAX_SCRAPED_IMAGES

                # Scroll to load images (matching test.py simple scroll behavior)
                logger.info("Scrolling to load images...")
                previous_height = await page.evaluate("document.body.scrollHeight")
                scroll_attempts = 0
                max_scroll_attempts = int(MAX_SCROLL_ROUNDS or 3)
                reached_bottom = False
                
                while scroll_attempts < max_scroll_attempts:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(SCROLL_SLEEP)
                    scroll_attempts += 1
                    
                    new_height = await page.evaluate("document.body.scrollHeight")
                    if new_height == previous_height:
                        # Height stopped changing - we're at the bottom
                        reached_bottom = True
                        logger.info(f"Reached bottom after {scroll_attempts} scrolls.")
                        break
                    else:
                        previous_height = new_height
                
                # Always check for "See more anyway" button after scrolling (whether we hit bottom or max attempts)
                if reached_bottom or scroll_attempts >= max_scroll_attempts:
                    logger.info("Checking for 'See more anyway' button...")
                    try:
                        see_more_btn = await page.query_selector('text="See more anyway"')
                        if see_more_btn:
                            await see_more_btn.click()
                            logger.info("Clicked 'See more anyway' button to load additional results")
                            await asyncio.sleep(2)
                            # Do one final scroll to load newly unlocked content
                            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            await asyncio.sleep(SCROLL_SLEEP)
                        else:
                            logger.info("No 'See more anyway' button found.")
                    except Exception as e:
                        logger.debug(f"Could not find or click 'See more anyway' button: {e}")
                
                logger.info(f"Scroll complete after {scroll_attempts} attempts.")

                # Wait for images section (matching test.py)
                await page.wait_for_selector('div[data-attrid="images universal"]', timeout=15000)

                # Re-check captcha after load cycle and, if in retry mode, recreate context
                if await self._has_captcha(page) and str(CAPTCHA_HANDLING).lower() == "retry":
                    attempt = 0
                    while attempt < int(CAPTCHA_MAX_RETRIES):
                        attempt += 1
                        logger.info(f"Captcha after load; retrying with new context (attempt {attempt}/{CAPTCHA_MAX_RETRIES})")
                        try:
                            await page.close()
                        except Exception:
                            pass
                        try:
                            await context.close()
                        except Exception:
                            pass
                        context, page = await new_context_and_page()
                        await page.goto(url, timeout=60000)
                        await self._accept_consent(page)
                        if not await self._has_captcha(page):
                            break
                    if await self._has_captcha(page):
                        logger.warning("Captcha persists after post-load retries; skipping this query")
                        return []

                # Collect preview URLs after scrolling is complete
                image_dicts = await self._collect_preview_urls(page, local_max, query)

                # --- NEW LOGIC: check semantic survival ---
                if ENABLE_CLIP_FILTER and len(image_dicts) == 0:
                    logger.warning(f"[scraper] No valid images passed semantic filter for '{query}'.")
                    retries = 0
                    max_retries = int(SCRAPER_REQUERY_MAX_RETRIES or 0)
                    while retries < max_retries and len(image_dicts) == 0:
                        retries += 1
                        logger.info(f"[scraper] Triggering Director Agent re-query (attempt {retries})...")
                        # Lazy import to avoid circular dependency
                        from pipeline.director_agent import llm_refine_query
                        new_query = await llm_refine_query(type("Seg", (), {"visual_query": query, "topic": "N/A", "content_type": "N/A", "transcript": query})(), [])
                        if new_query == query:
                            break
                        query = new_query
                        filtered_query = self._build_exclusion_query(query)
                        url = SEARCH_ENGINE_URL + filtered_query.replace(" ", "+")
                        await page.goto(url, timeout=60000)
                        await self._load_all_results(page)
                        # Re-check captcha after requery load and bail if still present in retry mode
                        if await self._has_captcha(page) and str(CAPTCHA_HANDLING).lower() == "retry":
                            logger.info("Captcha detected after requery; giving up on this segment (retry mode)")
                            break
                        image_dicts = await self._collect_preview_urls(page, local_max, query)

                # Convert dicts to ImageResult objects, preserving all metadata
                results = [
                    ImageResult(
                        url=img["url"],
                        title=img.get("description"),
                        source_url=img.get("source_url"),
                        path=None
                    )
                    for img in image_dicts
                ]
            finally:
                # Cancel watcher task gracefully before closing browser and contexts
                if watcher_task is not None:
                    watcher_task.cancel()
                    try:
                        await watcher_task
                    except asyncio.CancelledError:
                        logger.debug("reCAPTCHA watcher cancelled successfully")

                try:
                    await page.close()
                except Exception:
                    pass
                try:
                    await context.close()
                except Exception:
                    pass
                await browser.close()
            logger.info(f"[scraper] Final collected {len(results)} images after semantic filtering and retries.")
            return results

    async def _accept_consent(self, page):
        selectors = [
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Accept")',
            '#L2AGLb',
            'form[action*="consent"] button',
        ]
        for sel in selectors:
            try:
                btn = await page.query_selector(sel)
                if btn:
                    await btn.click()
                    await asyncio.sleep(1)
                    logger.info(f"Accepted consent popup with selector: {sel}")
                    return True
            except Exception:
                continue
        return False
    
    async def _watch_for_recaptcha(self, page, browser):
        """Continuously detect and handle reCAPTCHA, with graceful shutdown and human intervention support."""
        try:
            while True:
                # Check every 2–3 seconds
                await asyncio.sleep(random.uniform(2.0, 3.0))
                
                # Check if browser/page is still alive
                if browser and not browser.is_connected():
                    logger.debug("Browser disconnected, stopping reCAPTCHA watcher")
                    break
                    
                try:
                    # Check if page is still valid
                    await page.title()
                except Exception:
                    logger.debug("Page closed, stopping reCAPTCHA watcher")
                    break
                
                frames = page.frames
                recaptcha_frame = next((f for f in frames if "recaptcha" in f.url), None)
                if not recaptcha_frame:
                    continue

                # Check for checkbox-style captcha first
                checkbox = await recaptcha_frame.query_selector('.recaptcha-checkbox-border')
                if checkbox:
                    logger.info("🤖 Detected reCAPTCHA checkbox — attempting to click.")
                    await checkbox.click(delay=random.randint(100, 250))
                    await asyncio.sleep(random.uniform(1.5, 2.5))
                    
                    # Check if it passed or requires challenge
                    checked = await recaptcha_frame.evaluate(
                        """() => document.querySelector('.recaptcha-checkbox')?.getAttribute('aria-checked')"""
                    )
                    if checked == "true":
                        logger.info("✅ reCAPTCHA checkbox verified successfully!")
                        break
                    else:
                        # Check if challenge appeared (image grid, audio, etc.)
                        challenge_frame = next((f for f in page.frames if "recaptcha/api2/bframe" in f.url), None)
                        if challenge_frame:
                            logger.warning("⚠️  reCAPTCHA requires human intervention (image/audio challenge detected)")
                            logger.warning("⏸️  Scraper paused - please solve the captcha manually in the browser window")
                            
                            # Wait for human to solve it (check every 3 seconds up to configured wait)
                            max_wait = int(CAPTCHA_WAIT_SECONDS or 300)
                            elapsed = 0
                            while elapsed < max_wait:
                                await asyncio.sleep(3)
                                elapsed += 3
                                
                                # Check if captcha was solved
                                try:
                                    checked = await recaptcha_frame.evaluate(
                                        """() => document.querySelector('.recaptcha-checkbox')?.getAttribute('aria-checked')"""
                                    )
                                    if checked == "true":
                                        logger.info("✅ Human solved reCAPTCHA successfully! Resuming scraping...")
                                        return
                                except Exception:
                                    pass
                                
                                # Check if challenge frame disappeared (captcha solved)
                                current_challenge = next((f for f in page.frames if "recaptcha/api2/bframe" in f.url), None)
                                if not current_challenge:
                                    logger.info("✅ reCAPTCHA challenge resolved! Resuming scraping...")
                                    return
                                    
                            logger.error("❌ Timeout waiting for human to solve reCAPTCHA (5 minutes)")
                            break
                        else:
                            logger.debug("Checkbox clicked but status unclear, will retry")
                            
        except asyncio.CancelledError:
            logger.debug("reCAPTCHA watcher task cancelled")
            raise
        except Exception as e:
            logger.warning(f"reCAPTCHA watcher error: {e}")
            # Don't crash the whole scraper if watcher fails
    
    async def _has_captcha(self, page) -> bool:
        """Lightweight check if a reCAPTCHA seems present on current page."""
        try:
            # Frame-level check
            for f in page.frames:
                if "recaptcha" in (f.url or ""):
                    return True
            # Common selectors
            if await page.locator('iframe[src*="recaptcha"]').count() > 0:
                return True
            if await page.locator('.g-recaptcha, #recaptcha, div[aria-label*="captcha"]').count() > 0:
                return True
        except Exception:
            return False
        return False

    def _random_user_agent(self) -> str:
        """Return a Windows Chrome UA with consistent major version but randomized minor parts."""
        # Keep Chrome 121 major version for consistent Google Images layout
        major = 121
        build = random.randint(6167, 6200)
        patch = random.randint(160, 185)
        return (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.{build}.{patch} Safari/537.36"
        )


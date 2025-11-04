# Scraper Module

Web scraping system for collecting images from Google Images using Playwright with proxy rotation and CAPTCHA handling.

## Key Files

- `collector.py`: High-level scraper interface with retry logic
- `google_scraper.py`: Playwright-based Google Images scraper
- `proxy.py`: Rotating proxy manager with health checks
- `models.py`: Data models for scraping results
- `utils.py`: Shared utilities (URL validation, filename sanitization)

## Scraping Flow

1. Build search URL with query
2. Launch Playwright browser (with proxy if enabled)
3. Navigate to Google Images
4. Scroll to load more results (configurable rounds)
5. Extract image URLs from thumbnails
6. Download images in parallel
7. Filter by semantic relevance (CLIP) if enabled
8. Return list of downloaded ImageResult objects

## Configuration

See `config.py` for all settings:

**Concurrency & Limits:**

- `MAX_CONCURRENT_SCRAPER`: Parallel scraper tasks
- `MAX_SCRAPED_IMAGES`: Target number of successful downloads
- `SCROLL_PAUSES`: Number of scroll actions per search
- `THUMBNAIL_ATTEMPT_LIMIT`: Max thumbnails to try before stopping

**Proxy Settings:**

- `USE_PROXIES`: Enable/disable proxy rotation
- `ROTATING_PROXY_ENDPOINT`: Proxy URL (from .env)
- `PROXY_ROTATION_MODE`: "round_robin" or "random"
- `PROXY_APPLY_TO_PLAYWRIGHT`: Use proxy for browser traffic
- `PROXY_SCRAPER_BROWSING`: Use proxy for image downloads

**CAPTCHA Handling:**

- `CAPTCHA_HANDLING`: "manual", "retry", or "skip"
  - `"manual"`: Wait up to `CAPTCHA_WAIT_SECONDS` for human to solve
  - `"retry"`: Discard context, retry with fresh browser (up to `CAPTCHA_MAX_RETRIES`)
  - `"skip"`: Immediately skip query on CAPTCHA detection
- `CAPTCHA_RANDOMIZE_UA`: Randomize user agent on retry

**CLIP Filtering:**

- `ENABLE_CLIP_FILTER`: Semantic filtering during scraping
- `CLIP_RELEVANCE_THRESHOLD`: Min similarity to accept image

## Requery Logic

If semantic filter rejects all images, scraper can requery automatically:

- `SCRAPER_REQUERY_MAX_RETRIES`: Max requery attempts
- On requery, appends context words to query (e.g., "portrait" → "portrait photo realistic")

## Proxy Management

`proxy.py` handles rotating proxies:

- Health check on startup via `PROXY_HEALTHCHECK_URL`
- Logs external IP if `PROXY_LOG_EXTERNAL_IP=True`
- Round-robin or random selection per request

**Debugging proxy issues:**

- Check `[proxy]` logs for health check results
- Verify `.env` has correct credentials
- Test with `PROXY_LOG_EXTERNAL_IP=True` to confirm rotation

## CAPTCHA Handling Strategies

**Manual mode** (default for debugging):

- Browser window stays open
- Agent pauses execution
- Human solves CAPTCHA
- Continues after `CAPTCHA_WAIT_SECONDS` or user closes browser

**Retry mode** (production):

- On CAPTCHA, discards browser context
- Launches fresh browser with randomized UA
- Retries up to `CAPTCHA_MAX_RETRIES`
- Useful for automated workflows

**Skip mode**:

- Immediately moves to next query
- Use when CAPTCHA rate is low and time is critical

## Debugging Scraping Issues

**No images downloaded:**

- Check `[scraper]` logs for thumbnail extraction failures
- Verify search query returns results in browser manually
- Check if CAPTCHA is blocking (enable `PLAYWRIGHT_HEADFUL=True`)

**Low-quality images:**

- Adjust `CLIP_RELEVANCE_THRESHOLD` (lower = more permissive)
- Check `[ai_filter]` logs for CLIP similarity scores

**Proxy issues:**

- Verify proxy credentials in `.env`
- Check `[proxy] health check` logs
- Test without proxy first (`USE_PROXIES=False`)

**CAPTCHA blocking:**

- Switch to "retry" mode with randomized UA
- Reduce concurrent scrapers (`MAX_CONCURRENT_SCRAPER=1`)
- Add delays between requests

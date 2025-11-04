# Pipeline Architecture

The pipeline orchestrates the full video generation flow: parsing scripts → scraping images → AI filtering → rendering → audio overlay.

## Key Modules

- `director_agent.py`: Agentic orchestrator that plans segments, selects images, and retries on failure
- `parser.py`: Parses JSON scripts into VideoSegment objects with transcript, visual_query, duration
- `scraper/`: Google Images scraping with Playwright, proxy rotation, CAPTCHA handling
- `ai_filter/`: CLIP-based semantic filtering and ranking of scraped images
- `renderer/`: Video composition, effects application, and export (see `renderer/CLAUDE.md`)
- `postprocessing/`: Audio overlay and final video assembly

## Director Agent Flow

1. Parse script into segments
2. For each segment:
   - Generate visual query via LLM (if needed)
   - Scrape images matching query
   - Rank images via CLIP + resolution + sharpness
   - Select best image
   - Apply effects and render clip
3. Concatenate clips with transitions
4. Overlay audio track
5. Export final video

**Retry logic:** If CLIP ranking finds no good matches (below `RANK_MIN_CLIP_SIM`), Director Agent can requery up to `DIRECTOR_MAX_RETRIES` times with refined visual queries.

## Configuration Knobs

See `config.py` for all settings. Key ones:

- `DIRECTOR_MAX_RETRIES`: LLM requery attempts per segment
- `SCRAPER_REQUERY_MAX_RETRIES`: In-scraper requery attempts when semantic filter rejects all images
- `CLIP_RELEVANCE_THRESHOLD`: Minimum semantic similarity during scraping
- `RANK_MIN_CLIP_SIM`: Minimum similarity for final ranking inclusion

## Debugging Director Agent

- Check `[director_agent] segment X/Y` logs for progress
- Look for `[director_agent] retry N/M` to see requery behavior
- If images are consistently rejected, lower `RANK_MIN_CLIP_SIM` or check visual_query quality

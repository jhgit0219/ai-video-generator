"""
Proxy rotation utilities for aiohttp and Playwright.
Handles rotating endpoint or a list of proxies, and provides helpers
to format proxies for aiohttp (single URL) and Playwright (server + creds).
"""
from __future__ import annotations
import random
import itertools
from typing import Optional
import aiohttp
from aiohttp import ClientTimeout
from config import PROXY_HEALTHCHECK_URL
from urllib.parse import urlparse
from config import (
    USE_PROXIES,
    ROTATING_PROXY_ENDPOINT,
    PROXY_LIST,
    PROXY_ROTATION_MODE,
)

# Build an iterator for round-robin mode
_cycle_iter = itertools.cycle(PROXY_LIST) if PROXY_LIST else None


def proxies_enabled() -> bool:
    return bool(USE_PROXIES and (ROTATING_PROXY_ENDPOINT or PROXY_LIST))


def next_proxy() -> Optional[str]:
    """Return the next proxy URL to use, or None if proxies disabled.
    - If ROTATING_PROXY_ENDPOINT is set, always return that (provider rotates internally).
    - Else use PROXY_LIST with rotation mode.
    """
    if not proxies_enabled():
        return None
    if ROTATING_PROXY_ENDPOINT:
        return ROTATING_PROXY_ENDPOINT
    if not PROXY_LIST:
        return None
    mode = (PROXY_ROTATION_MODE or "round_robin").lower()
    if mode == "random":
        return random.choice(PROXY_LIST)
    # default round_robin
    global _cycle_iter
    if _cycle_iter is None:
        _cycle_iter = itertools.cycle(PROXY_LIST)
    return next(_cycle_iter)


def aiohttp_proxy() -> Optional[str]:
    """Return a proxy URL string suitable for aiohttp's proxy= parameter."""
    return next_proxy() if proxies_enabled() else None


def playwright_proxy() -> Optional[dict]:
    """Return a dict suitable for Playwright launch proxy option.
    Splits credentials out of the URL as Playwright expects:
      {"server": "http://host:port", "username": "user", "password": "pass"}
    Returns None if proxies are disabled or URL is invalid.
    """
    if not proxies_enabled():
        return None


async def get_proxy_identity(url: Optional[str] = None, timeout: int = 10) -> Optional[str]:
    """Fetch an IP-echo endpoint through the proxy and return its text (external IP or payload).
    Returns None on failure. Uses aiohttp with current proxy selection.
    """
    target = url or PROXY_HEALTHCHECK_URL
    proxy = aiohttp_proxy()
    if not proxy:
        return None
    try:
        to = ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=to) as session:
            async with session.get(target, proxy=proxy) as resp:
                if resp.status == 200:
                    return (await resp.text()).strip()
    except Exception:
        return None
    return None
    url = next_proxy()
    if not url:
        return None
    try:
        parsed = urlparse(url)
        server = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        proxy_conf = {"server": server}
        if parsed.username:
            proxy_conf["username"] = parsed.username
        if parsed.password:
            proxy_conf["password"] = parsed.password
        return proxy_conf
    except Exception:
        return None

"""Retrying, rate-limited GET shared by the DailyMed and RxNav clients."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import httpx


class RateLimiter:
    """Serialises callers and spaces them at least `1 / rate` seconds apart. 0 disables it."""

    def __init__(self, rate_per_s: float) -> None:
        self._interval = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self._interval


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int,
    backoff_s: float,
    before_request: Callable[[], Awaitable[None]] | None = None,
    **kwargs,
) -> httpx.Response:
    """GET, retrying timeouts, connection errors, 429 and 5xx with exponential backoff.

    `before_request` runs before every attempt, retries included, so a rate limiter passed
    here also paces the retries. Other 4xx responses raise immediately: retrying a bad
    request cannot help.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        if before_request is not None:
            await before_request()
        try:
            resp = await client.get(url, **kwargs)
        except httpx.TransportError as e:
            last = e
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                last = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code} for {url}", request=resp.request, response=resp
                )
            else:
                resp.raise_for_status()
                return resp
        if attempt < retries:
            await asyncio.sleep(backoff_s * 2**attempt)
    assert last is not None
    raise last

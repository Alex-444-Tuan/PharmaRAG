"""Small retrying GET shared by the DailyMed and RxNav clients."""

from __future__ import annotations

import asyncio

import httpx


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int,
    backoff_s: float,
    **kwargs,
) -> httpx.Response:
    """GET, retrying timeouts, connection errors, 429 and 5xx with exponential backoff.

    Other 4xx responses raise immediately: retrying a bad request cannot help.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
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

"""Key-value cache interface (spec section 5, Redis keys). Contract file.

Values are bytes; callers serialise. RedisCache in production, InMemoryCache in tests
and when CACHE_BACKEND=memory.
"""

import logging
import time
from typing import Protocol

_log = logging.getLogger(__name__)


class Cache(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_s: int) -> None: ...


class InMemoryCache:
    """Process-local TTL cache. Fine for one API process and for tests."""

    def __init__(self, clock=time.monotonic) -> None:
        self._data: dict[str, tuple[float, bytes]] = {}
        self._clock = clock

    async def get(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if self._clock() >= expires:
            del self._data[key]
            return None
        return value

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        self._data[key] = (self._clock() + ttl_s, value)


class RedisCache:
    """Redis-backed cache (spec section 5). Fail-soft: a Redis error is logged and treated
    as a miss / no-op, so a cache outage slows requests down but never fails them.
    `/readyz` reports Redis health separately."""

    def __init__(self, url: str, *, client=None) -> None:
        import redis.asyncio as redis

        self._r = client or redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)

    async def get(self, key: str) -> bytes | None:
        try:
            return await self._r.get(key)
        except Exception as e:  # noqa: BLE001 - any Redis failure degrades to a miss
            _log.warning("redis get failed: %s", type(e).__name__)
            return None

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        try:
            await self._r.set(key, value, ex=ttl_s)
        except Exception as e:  # noqa: BLE001
            _log.warning("redis set failed: %s", type(e).__name__)

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        await self._r.aclose()

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

    async def incr(self, key: str, ttl_s: int) -> int | None:
        """Atomically increment a counter, setting its TTL only when the key is created.
        Returns the new value, or None on a backend fault (callers fail open)."""
        ...


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

    async def incr(self, key: str, ttl_s: int) -> int | None:
        # No await between read and write: atomic on one event loop.
        current = await self.get(key)
        if current is None:
            self._data[key] = (self._clock() + ttl_s, b"1")
            return 1
        expires, _ = self._data[key]
        n = int(current) + 1
        self._data[key] = (expires, str(n).encode())
        return n


class RedisCache:
    """Redis-backed cache (spec section 5). Fail-soft: a Redis error is logged and treated
    as a miss / no-op, so a cache outage slows requests down but never fails them.
    `/readyz` reports Redis health separately."""

    # After a failure, skip Redis for this long: an unreachable (not refusing) Redis would
    # otherwise cost a 0.5 s timeout on every cache operation of every request (NFR-1).
    COOLDOWN_S = 5.0

    def __init__(self, url: str, *, client=None, clock=time.monotonic) -> None:
        import redis.asyncio as redis

        self._r = client or redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        self._clock = clock
        self._down_until = 0.0

    def _available(self) -> bool:
        return self._clock() >= self._down_until

    def _fault(self, op: str, e: Exception) -> None:
        self._down_until = self._clock() + self.COOLDOWN_S
        _log.warning("redis %s failed: %s", op, type(e).__name__)

    async def get(self, key: str) -> bytes | None:
        if not self._available():
            return None
        try:
            return await self._r.get(key)
        except Exception as e:  # noqa: BLE001 - any Redis failure degrades to a miss
            self._fault("get", e)
            return None

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        if not self._available():
            return
        try:
            await self._r.set(key, value, ex=ttl_s)
        except Exception as e:  # noqa: BLE001
            self._fault("set", e)

    async def incr(self, key: str, ttl_s: int) -> int | None:
        """SET NX EX then INCR in one MULTI: atomic, and works on any Redis version
        (EXPIRE ... NX needs Redis 7)."""
        if not self._available():
            return None
        try:
            async with self._r.pipeline(transaction=True) as p:
                p.set(key, 0, ex=ttl_s, nx=True)
                p.incr(key)
                _, n = await p.execute()
            return int(n)
        except Exception as e:  # noqa: BLE001
            self._fault("incr", e)
            return None

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        await self._r.aclose()

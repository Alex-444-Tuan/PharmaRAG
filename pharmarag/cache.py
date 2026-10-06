"""Key-value cache interface (spec section 5, Redis keys). Contract file.

Values are bytes; callers serialise. Cycle 4 adds the Redis implementation.
Until then the app uses InMemoryCache.
"""

import time
from typing import Protocol


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

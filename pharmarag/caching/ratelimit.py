"""Fixed-window rate limiting (spec sections 10 and 12): `rl:{client}:{minute}`, 60 s.

Two implementations of `RateLimiter`:

- `RedisRateLimiter` counts in Redis with `INCR` and `EXPIRE NX` in one MULTI/EXEC, so two API
  processes share one budget and a crash can never leave a counter without a TTL.
- `LocalRateLimiter` counts in process memory. It serves `CACHE_BACKEND=memory`, where there is
  one process and no Redis.

Both are fail-OPEN where a failure is possible: if Redis errors, the request is allowed and the
error is logged. A rate limiter that took the API down with the cache would be a worse failure
than a few unthrottled requests (the LLM call has its own timeout and the NLI model its own
queue). `/readyz` reports Redis health separately.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from pharmarag.cache import RedisCache
from pharmarag.ops import RATE_WINDOW_S, RateDecision, rate_key

_log = logging.getLogger(__name__)


def _decide(count: int, limit: int, now: float) -> RateDecision:
    """`count` includes the current request. Request number `limit` is still allowed."""
    if count <= limit:
        return RateDecision(True, limit - count, 0)
    # The window ends at the next multiple of RATE_WINDOW_S. Never tell a client to retry in 0 s.
    retry = max(1, int(RATE_WINDOW_S - (now % RATE_WINDOW_S)))
    return RateDecision(False, 0, retry)


class RedisRateLimiter:
    # SPEC: section 5 gives the key `rl:{client ip}:{minute}` with a 60 s TTL and section 12 says
    # "fixed window". A limit of 0 or less switches the limiter off (allow everything).
    def __init__(
        self,
        cache: RedisCache,
        limit: int,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._r = cache._r  # the limiter needs INCR, which the get/set Cache protocol lacks
        self._limit = limit
        self._clock = clock

    async def hit(self, client_id: str) -> RateDecision:
        if self._limit <= 0:
            return RateDecision(True, self._limit, 0)
        now = self._clock()
        key = rate_key(client_id, int(now // RATE_WINDOW_S))
        try:
            async with self._r.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, RATE_WINDOW_S, nx=True)  # only the first hit sets the TTL
                count = int((await pipe.execute())[0])
        except Exception as e:  # noqa: BLE001 - any Redis failure fails open
            _log.warning("rate limiter failing open: redis %s", type(e).__name__)
            return RateDecision(True, self._limit, 0)
        return _decide(count, self._limit, now)


class LocalRateLimiter:
    """Same window arithmetic, counters in this process. Old windows are dropped on every hit,
    so the dict holds one entry per client seen in the current minute."""

    def __init__(self, limit: int, *, clock: Callable[[], float] = time.time) -> None:
        self._limit = limit
        self._clock = clock
        self._counts: dict[tuple[str, int], int] = {}

    async def hit(self, client_id: str) -> RateDecision:
        if self._limit <= 0:
            return RateDecision(True, self._limit, 0)
        now = self._clock()
        minute = int(now // RATE_WINDOW_S)
        self._counts = {k: v for k, v in self._counts.items() if k[1] == minute}
        key = (client_id, minute)
        self._counts[key] = count = self._counts.get(key, 0) + 1
        return _decide(count, self._limit, now)

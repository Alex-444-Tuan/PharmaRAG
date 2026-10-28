"""Fixed-window rate limiting (spec sections 10 and 12): `rl:{client}:{minute}`, 60 s.

One limiter over any `Cache`. The counter is `cache.incr(rate_key(client, minute), 60)`:

- On `RedisCache` that is SET NX EX plus INCR in one MULTI, so two API processes share one
  budget and a counter can never outlive its window.
- On `InMemoryCache` (`CACHE_BACKEND=memory`, one process) it is the same arithmetic in memory.

Fail-OPEN: `incr` returns None on a backend fault, and the request is allowed. A limiter that
took the API down with the cache would be a worse failure than a few unthrottled requests (the
LLM call has its own timeout and the NLI model its own queue). `/readyz` reports Redis health
separately, and RedisCache's cooldown keeps an unreachable Redis from adding a timeout to every
request.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from pharmarag.cache import Cache
from pharmarag.ops import RATE_WINDOW_S, RateDecision, rate_key

_log = logging.getLogger(__name__)


class FixedWindowRateLimiter:
    # SPEC: section 5 gives the key `rl:{client ip}:{minute}` with a 60 s TTL and section 12 says
    # "fixed window". A limit of 0 or less switches the limiter off (allow everything).
    def __init__(self, cache: Cache, limit: int, *, clock: Callable[[], float] = time.time) -> None:
        self._cache = cache
        self._limit = limit
        self._clock = clock

    async def hit(self, client_id: str) -> RateDecision:
        if self._limit <= 0:
            return RateDecision(True, self._limit, 0)
        now = self._clock()
        count = await self._cache.incr(
            rate_key(client_id, int(now // RATE_WINDOW_S)), RATE_WINDOW_S
        )
        if count is None:
            _log.warning("rate limiter failing open: cache fault")
            return RateDecision(True, self._limit, 0)
        if count <= self._limit:  # request number `limit` is still allowed
            return RateDecision(True, self._limit - count, 0)
        # The window ends at the next multiple of RATE_WINDOW_S. Never say "retry in 0 s".
        return RateDecision(False, 0, max(1, int(RATE_WINDOW_S - (now % RATE_WINDOW_S))))

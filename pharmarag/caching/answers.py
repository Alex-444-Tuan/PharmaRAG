"""Answer cache (spec section 5, `ans:*`): final response JSON for ANSWER_TTL_S.

The key (built by `pharmarag.ops.answer_key`) already carries `corpus_version`, the effective
flags and the sorted RxCUIs, so this class only moves JSON in and out. It never decides what is
cacheable: the caller (Lane B's route) puts only complete, non-degraded answers.
"""

from __future__ import annotations

import json
import logging

from pharmarag.cache import Cache
from pharmarag.ops import ANSWER_TTL_S

_log = logging.getLogger(__name__)


class CacheAnswerCache:
    def __init__(self, cache: Cache) -> None:
        self._cache = cache
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> dict | None:
        raw = await self._cache.get(key)
        if raw is not None:
            try:
                value = json.loads(raw)
            except ValueError:
                value = None  # a corrupt entry is a miss
            if isinstance(value, dict):
                self.hits += 1
                return value
        self.misses += 1
        return None

    async def put(self, key: str, response: dict) -> None:
        try:
            raw = json.dumps(response, ensure_ascii=False, sort_keys=True).encode()
        except (TypeError, ValueError) as e:
            _log.warning("answer not cached, not JSON-serialisable: %s", type(e).__name__)
            return
        await self._cache.set(key, raw, ANSWER_TTL_S)


class NullAnswerCache:
    """ANSWER_CACHE=false (spec 11.3: the latency eval runs with the answer cache off)."""

    async def get(self, key: str) -> dict | None:
        return None

    async def put(self, key: str, response: dict) -> None:
        return None

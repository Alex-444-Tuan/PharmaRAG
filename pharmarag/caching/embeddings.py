"""Query-embedding cache (spec section 5, `emb:q:{model}:{sha1}`, 7 days).

The value is the float32 vector as little-endian bytes: 384 * 4 = 1536 bytes, no JSON float
round trip, so a cached vector is bit-identical to a fresh one. The model name is part of the key,
so changing EMBEDDING_MODEL never serves vectors from the old model; the length check below also
rejects an entry of the wrong size.
"""

from __future__ import annotations

import asyncio

import numpy as np

from pharmarag.cache import Cache
from pharmarag.models.embedder import Embedder
from pharmarag.ops import QUERY_EMBEDDING_TTL_S, query_embedding_key

_DTYPE = np.dtype("<f4")


class CachedQueryEmbeddings:
    def __init__(self, embedder: Embedder, cache: Cache) -> None:
        self._embedder = embedder
        self._cache = cache
        self.hits = 0
        self.misses = 0

    async def embed_query(self, text: str) -> np.ndarray:
        key = query_embedding_key(self._embedder.model_name, text)
        raw = await self._cache.get(key)
        if raw is not None and len(raw) == self._embedder.dim * _DTYPE.itemsize:
            self.hits += 1
            return np.frombuffer(raw, dtype=_DTYPE).astype(np.float32)  # astype copies: writable
        self.misses += 1
        # The model is CPU-bound and holds a lock; keep it off the event loop.
        vec = await asyncio.to_thread(self._embedder.embed_query, text)
        vec = np.asarray(vec, dtype=np.float32)
        await self._cache.set(key, vec.astype(_DTYPE).tobytes(), QUERY_EMBEDDING_TTL_S)
        return vec

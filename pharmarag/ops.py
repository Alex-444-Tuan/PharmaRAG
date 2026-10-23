"""Operational interfaces and Redis key builders (spec sections 5, 10, 12). Contract file.

Lane A implements the protocols (pharmarag/caching/), Lane B wires them into the app.
Key formats follow the spec section 5 table exactly; build keys only through these helpers.
"""

import hashlib
from dataclasses import dataclass
from typing import Protocol

RESOLVE_TTL_S = 24 * 3600  # rx:resolve:*  (already used by the resolver)
FACTS_TTL_S = 24 * 3600  # rx:facts:*
QUERY_EMBEDDING_TTL_S = 7 * 24 * 3600  # emb:q:*
ANSWER_TTL_S = 24 * 3600  # ans:*
RATE_WINDOW_S = 60  # rl:*


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def query_embedding_key(model: str, query: str) -> str:
    return f"emb:q:{model}:{_sha1(query)}"


def answer_key(corpus_version: str, flags: str, question: str, rxcuis: list[str]) -> str:
    """`flags` is a canonical string of the effective flags and verify mode, e.g.
    'resolver=on,verifier=on,verify=stream'. RxCUIs are sorted so order never matters."""
    return f"ans:{corpus_version}:{flags}:{_sha1(question + '|' + ','.join(sorted(rxcuis)))}"


def rate_key(client_id: str, minute: int) -> str:
    return f"rl:{client_id}:{minute}"


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    remaining: int
    retry_after_s: int  # 0 when allowed


class RateLimiter(Protocol):
    async def hit(self, client_id: str) -> RateDecision:
        """Count one request for `client_id` in the current fixed one-minute window."""
        ...


class AnswerCache(Protocol):
    async def get(self, key: str) -> dict | None: ...

    async def put(self, key: str, response: dict) -> None: ...


class QueryEmbeddingCache(Protocol):
    async def embed_query(self, text: str):
        """Return the query vector (np.ndarray), from cache or by embedding and storing it."""
        ...

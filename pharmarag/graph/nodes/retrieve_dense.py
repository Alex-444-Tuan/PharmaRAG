"""retrieve_dense: embed the search query and rank chunks by cosine similarity (section 8)."""

import asyncio

from pharmarag.app.observability import probe_cache
from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Hit, RAGState
from pharmarag.retrieval.dense import dense_search
from pharmarag.retrieval.scope import scopes


async def _query_vector(deps: Deps, query: str):
    """The query vector, from the Redis-backed cache when this build has one (section 5)."""
    if deps.query_embeddings is None:
        # The model runs on CPU and blocks; keep it off the event loop so the lexical branch
        # (and other requests) keep moving.
        return await asyncio.to_thread(deps.embedder.embed_query, query)
    with probe_cache() as probe:
        vec = await deps.query_embeddings.embed_query(query)
    if deps.metrics is not None:
        deps.metrics.cache_result("query_embedding", hit=not probe.missed)
    return vec


async def retrieve_dense(state: RAGState, deps: Deps) -> dict:
    keys = scopes(state)  # raises before any work if a mention has no filter
    attempt = state.get("retrieval_attempt", 1)
    vec = await _query_vector(deps, state["search_query"])
    cfg = deps.thresholds.retrieval
    hits: list[Hit] = []
    async with deps.pool.connection() as conn:
        for key in keys:
            hits += await dense_search(
                conn,
                vec,
                ingredient_key=key,
                sections=state.get("sections"),
                top_k=cfg.top_k_per_branch,
                attempt=attempt,
            )
    return {"hits": hits}

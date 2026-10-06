"""retrieve_dense: embed the search query and rank chunks by cosine similarity (section 8)."""

import asyncio

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Hit, RAGState
from pharmarag.retrieval.dense import dense_search
from pharmarag.retrieval.scope import scopes


async def retrieve_dense(state: RAGState, deps: Deps) -> dict:
    keys = scopes(state)  # raises before any work if a mention has no filter
    attempt = state.get("retrieval_attempt", 1)
    # The model runs on CPU and blocks; keep it off the event loop so the lexical branch
    # (and other requests) keep moving.
    vec = await asyncio.to_thread(deps.embedder.embed_query, state["search_query"])
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

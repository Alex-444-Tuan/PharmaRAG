"""retrieve_lexical: full-text ranking over chunk text (section 8)."""

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Hit, RAGState
from pharmarag.retrieval.lexical import lexical_search
from pharmarag.retrieval.scope import scopes


async def retrieve_lexical(state: RAGState, deps: Deps) -> dict:
    keys = scopes(state)
    attempt = state.get("retrieval_attempt", 1)
    cfg = deps.thresholds.retrieval
    hits: list[Hit] = []
    async with deps.pool.connection() as conn:
        for key in keys:
            hits += await lexical_search(
                conn,
                state["search_query"],
                ingredient_key=key,
                sections=state.get("sections"),
                top_k=cfg.top_k_per_branch,
                attempt=attempt,
            )
    return {"hits": hits}

"""retrieve_lexical: full-text ranking over chunk text (section 8)."""

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Hit, RAGState
from pharmarag.resolver.normalize import normalize
from pharmarag.retrieval.lexical import lexical_search, strip_drug_terms
from pharmarag.retrieval.scope import scopes


def drug_terms(state: RAGState, key: str) -> list[str]:
    """The name-bearing words of the drug behind `key`: its mention(s) after the resolver's own
    normalisation, and its resolved name.

    The raw mention is not used: "hydroxyzine 25 mg" would also strip "25" and "mg", and a
    mention that carries a form word ("amlodipine/olmesartan dose") would strip "dose".
    `normalize` drops exactly the dose and form words, so they stay in the query.
    """
    terms: list[str] = []
    for r in state.get("resolutions", []):
        if r.status == "resolved" and r.ingredient_key == key:
            terms += [normalize(r.mention), r.name or ""]
    return terms


async def retrieve_lexical(state: RAGState, deps: Deps) -> dict:
    keys = scopes(state)
    attempt = state.get("retrieval_attempt", 1)
    cfg = deps.thresholds.retrieval
    hits: list[Hit] = []
    async with deps.pool.connection() as conn:
        for key in keys:
            query = state["search_query"]
            if key is not None:
                # Inside the filter the drug's name matches every chunk and carries no signal
                # (see `strip_drug_terms`). The unfiltered path keeps the full query.
                query = strip_drug_terms(query, drug_terms(state, key))
                if not query:
                    continue  # nothing but the drug name was asked: no lexical hits for this key
            hits += await lexical_search(
                conn,
                query,
                ingredient_key=key,
                sections=state.get("sections"),
                top_k=cfg.top_k_per_branch,
                attempt=attempt,
            )
    return {"hits": hits}

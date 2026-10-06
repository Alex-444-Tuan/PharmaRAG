"""fetch_rx_facts: RxNorm facts branch (spec section 8, "RxNorm facts branch").

Runs in the same superstep as the two label branches. Facts are extra citable evidence (brands,
dose forms, strengths), never a filter and never required, so this branch fails soft.
"""

import logging

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Fact, RAGState

logger = logging.getLogger(__name__)


def rxcuis_of(ingredient_keys: list[str]) -> list[str]:
    """Ingredient RxCUIs behind the keys. A key is its sorted RxCUIs joined by '+'."""
    parts = (rxcui for key in ingredient_keys for rxcui in key.split("+"))
    return list(dict.fromkeys(p for p in parts if p))


async def fetch_rx_facts(state: RAGState, deps: Deps) -> dict:
    rxcuis = rxcuis_of(state.get("ingredient_keys") or [])
    if not rxcuis or deps.rx_facts is None:
        return {"rx_facts": []}
    try:
        found = await deps.rx_facts.facts(rxcuis)
    except Exception:
        # SPEC: section 6 gives RxNav calls a node RetryPolicy, but Lane A's client already
        # retries internally. If it still fails, the label evidence can answer the question on
        # its own (NFR-4: an RxNav outage must not take the service down), so this branch
        # returns no facts instead of failing the request.
        logger.warning("RxNorm facts unavailable; using label evidence only", exc_info=True)
        return {"rx_facts": []}
    attempt = state.get("retrieval_attempt", 1)
    cap = deps.thresholds.retrieval.max_facts
    return {
        "rx_facts": [
            Fact(id=f"r{i}", rxcui=f.rxcui, text=f.text, retrieval_attempt=attempt)
            for i, f in enumerate(found[:cap], start=1)
        ]
    }

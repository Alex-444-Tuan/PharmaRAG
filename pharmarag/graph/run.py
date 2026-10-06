"""Entry point used by the API and the eval harness: one request through the compiled graph."""

import time
import uuid

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Flags, RAGState


def effective_flags(requested: Flags, deps: Deps) -> Flags:
    """Switch a stage off when this build has no implementation of it (see `Deps`)."""
    return Flags(
        resolver=requested.resolver and deps.resolver_available,
        verifier=requested.verifier and deps.verifier_available,
    )


async def run_ask(
    graph,
    deps: Deps,
    *,
    question: str,
    pinned_rxcuis: list[str] | None = None,
    flags: Flags | None = None,
    request_id: str | None = None,
    debug: bool = False,
) -> dict:
    t0 = time.perf_counter()
    flags = effective_flags(flags or Flags(), deps)
    state: RAGState = {
        "question": question,
        "flags": flags,
        "pinned_rxcuis": pinned_rxcuis or [],
        "request_id": request_id or uuid.uuid4().hex,
        "search_query": question,
        "retrieval_attempt": 1,
        "generation_attempt": 1,
        "hits": [],
        "rx_facts": [],
        "timings": {},
    }
    final = await graph.ainvoke(
        state, config={"recursion_limit": deps.thresholds.graph.recursion_limit}
    )
    response = final["response"]
    response["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    if debug:
        response["debug"] = {
            "request_id": state["request_id"],
            "flags": flags.model_dump(),
            "intent": final.get("intent"),
            "mentions": final.get("mentions", []),
            "sections": final.get("sections", []),
            "search_query": final.get("search_query"),
            "timings_ms": final.get("timings", {}),
            "removed_sentences": [s.model_dump() for s in final.get("removed_sentences", [])],
        }
    return response

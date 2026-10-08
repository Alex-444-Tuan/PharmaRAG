"""Entry point used by the API and the eval harness: one request through the compiled graph."""

import time
import uuid
from collections.abc import AsyncIterator

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Flags, RAGState
from pharmarag.grounding.stream import resolve_verify_mode


def effective_flags(requested: Flags, deps: Deps) -> Flags:
    """The flags a request really runs with.

    1. Outside EVAL_MODE the requested flags are ignored: both stages are on. This is enforced
       here and not only in the API, so no caller can run a production request unverified.
    2. A stage this build does not have (`Deps.resolver` or `Deps.verifier` is None) is switched
       off, so the request runs as the AT-1 / AT-2 baseline and says so in `meta`.
    3. `VERIFY_MODE=off` switches the verifier off for every request, in EVAL_MODE only (the
       latency eval uses it as its `off` arm). Outside EVAL_MODE `resolve_verify_mode` ignores it.
    """
    if not deps.settings.eval_mode:
        requested = Flags()
    return Flags(
        resolver=requested.resolver and deps.resolver_available,
        verifier=requested.verifier
        and deps.verifier_available
        and resolve_verify_mode(deps.settings) != "off",
    )


def initial_state(
    question: str,
    *,
    flags: Flags,
    pinned_rxcuis: list[str] | None = None,
    request_id: str | None = None,
) -> RAGState:
    return {
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


def _start(
    deps: Deps,
    question: str,
    pinned_rxcuis: list[str] | None,
    flags: Flags | None,
    request_id: str | None,
) -> tuple[RAGState, dict]:
    flags = effective_flags(flags or Flags(), deps)
    state = initial_state(question, flags=flags, pinned_rxcuis=pinned_rxcuis, request_id=request_id)
    return state, {"recursion_limit": deps.thresholds.graph.recursion_limit}


def _finish(final: dict, state: RAGState, t0: float, debug: bool) -> dict:
    response = final["response"]
    response["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    if debug:
        flags = state["flags"]
        response["debug"] = {
            "request_id": state["request_id"],
            "flags": flags.model_dump(),
            "intent": final.get("intent"),
            "mentions": final.get("mentions", []),
            "ingredient_keys": final.get("ingredient_keys", []),
            "resolutions": [r.model_dump() for r in final.get("resolutions", [])],
            # NFR-4: some mention was resolved from the local alias table because RxNav was down.
            "degraded": any(r.degraded for r in final.get("resolutions", [])),
            "sections": final.get("sections", []),
            "search_query": final.get("search_query"),
            "support_ratio": final.get("support_ratio"),
            "timings_ms": final.get("timings", {}),
            "removed_sentences": [s.model_dump() for s in final.get("removed_sentences", [])],
        }
    return response


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
    state, config = _start(deps, question, pinned_rxcuis, flags, request_id)
    final = await graph.ainvoke(state, config=config)
    return _finish(final, state, t0, debug)


async def run_ask_stream(
    graph,
    deps: Deps,
    *,
    question: str,
    pinned_rxcuis: list[str] | None = None,
    flags: Flags | None = None,
    request_id: str | None = None,
    debug: bool = False,
) -> AsyncIterator[dict]:
    """One request as server-sent events (spec section 10): `{"event": ..., "data": ...}` dicts.

    `drugs` (when the resolver ran), then one `sentence` per sentence released to the client,
    then `done` with the full response, the same payload `run_ask` returns. `done` is the
    authoritative answer. Every released sentence is in it, in order (FR-6).

    An exception from the graph propagates to the caller after the events already yielded.
    """
    t0 = time.perf_counter()
    state, config = _start(deps, question, pinned_rxcuis, flags, request_id)
    config["configurable"] = {"sse": True}
    last: dict = {}
    async for mode, chunk in graph.astream(state, config=config, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield chunk
        else:
            last = chunk
    yield {"event": "done", "data": _finish(last, state, t0, debug)}

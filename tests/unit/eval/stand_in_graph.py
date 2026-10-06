"""A stand-in for lane B's `resolve` node, so the eval's system arm can be tested end to end.

Lane B builds the real graph in parallel. This module wires the cycle 1 nodes that exist on main
(analyze, retrieval, fuse, answer, finalize) together with a minimal `resolve` node that calls
our `Resolver`, and it adds `candidates` to a clarification response the way section 10 says the
API does. It is test code only: when lane B's graph is merged the eval runs against that graph
and this file is no longer needed.
"""

import time

from langgraph.graph import END, START, StateGraph

from pharmarag.graph.deps import Deps
from pharmarag.graph.nodes.analyze import analyze
from pharmarag.graph.nodes.answer import answer
from pharmarag.graph.nodes.finalize import DECLINED_INTENTS, finalize
from pharmarag.graph.nodes.fuse_grade import fuse_grade
from pharmarag.graph.nodes.retrieve_dense import retrieve_dense
from pharmarag.graph.nodes.retrieve_lexical import retrieve_lexical
from pharmarag.graph.routing import route_after_fuse
from pharmarag.graph.state import Flags, RAGState
from pharmarag.resolver.base import Resolver

STATUS_OF = {
    "ambiguous": "needs_clarification",
    "unknown": "drug_not_recognised",
    "not_covered": "not_covered",
}


def _timed(name, fn, deps):
    async def run(state: RAGState) -> dict:
        t0 = time.perf_counter()
        out = await fn(state, deps)
        return {**out, "timings": {name: round((time.perf_counter() - t0) * 1000, 2)}}

    run.__name__ = name
    return run


def build_system_graph(deps: Deps, resolver: Resolver):
    async def resolve(state: RAGState, _deps: Deps) -> dict:
        mentions = state.get("mentions", [])
        if not (state.get("flags") or Flags()).resolver or not mentions:
            return {}
        resolutions = await resolver.resolve(mentions, pinned_rxcuis=state.get("pinned_rxcuis"))
        out: dict = {"resolutions": resolutions}
        bad = next((r for r in resolutions if r.status != "resolved"), None)
        if bad is not None:
            out["status"] = STATUS_OF[bad.status]
        else:
            out["ingredient_keys"] = [r.ingredient_key for r in resolutions]
        return out

    async def finalize_with_candidates(state: RAGState, _deps: Deps) -> dict:
        out = await finalize(state, deps)
        if out["status"] == "needs_clarification":
            out["response"]["candidates"] = [
                c.model_dump() for r in state.get("resolutions", []) for c in r.candidates
            ]
        return out

    def after_analyze(state: RAGState) -> list[str]:
        return ["finalize"] if state.get("intent") in DECLINED_INTENTS else ["resolve"]

    def after_resolve(state: RAGState) -> list[str]:
        return ["finalize"] if state.get("status") else ["retrieve_dense", "retrieve_lexical"]

    g = StateGraph(RAGState)
    for name, fn in (
        ("analyze", analyze),
        ("resolve", resolve),
        ("retrieve_dense", retrieve_dense),
        ("retrieve_lexical", retrieve_lexical),
        ("fuse_grade", fuse_grade),
        ("answer", answer),
        ("finalize", finalize_with_candidates),
    ):
        g.add_node(name, _timed(name, fn, deps))
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", after_analyze, ["resolve", "finalize"])
    g.add_conditional_edges(
        "resolve", after_resolve, ["retrieve_dense", "retrieve_lexical", "finalize"]
    )
    g.add_edge(["retrieve_dense", "retrieve_lexical"], "fuse_grade")
    g.add_conditional_edges("fuse_grade", route_after_fuse, ["answer", "finalize"])
    g.add_edge("answer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()

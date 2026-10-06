"""Compile the LangGraph state machine (spec section 6).

Cycle 1 shape:

    analyze -> (declined -> finalize)
            -> retrieve_dense | retrieve_lexical (one superstep) -> fuse_grade
            -> (weak -> finalize) -> answer -> finalize

Cycle 2 adds `resolve`, `fetch_rx_facts`, `rewrite_query`, and the retry edges.
"""

import time
from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from pharmarag.graph.deps import Deps
from pharmarag.graph.nodes.analyze import analyze
from pharmarag.graph.nodes.answer import answer
from pharmarag.graph.nodes.finalize import finalize
from pharmarag.graph.nodes.fuse_grade import fuse_grade
from pharmarag.graph.nodes.retrieve_dense import retrieve_dense
from pharmarag.graph.nodes.retrieve_lexical import retrieve_lexical
from pharmarag.graph.routing import route_after_analyze, route_after_fuse
from pharmarag.graph.state import RAGState

NodeFn = Callable[[RAGState, Deps], Awaitable[dict]]


def _node(name: str, fn: NodeFn, deps: Deps) -> Callable[[RAGState], Awaitable[dict]]:
    """Bind deps and record the node's wall time (ms) in `timings` (NFR-5)."""

    async def run(state: RAGState) -> dict:
        t0 = time.perf_counter()
        out = await fn(state, deps)
        return {**out, "timings": {name: round((time.perf_counter() - t0) * 1000, 2)}}

    run.__name__ = name
    return run


def build_graph(deps: Deps):
    g = StateGraph(RAGState)
    for name, fn in (
        ("analyze", analyze),
        ("retrieve_dense", retrieve_dense),
        ("retrieve_lexical", retrieve_lexical),
        ("fuse_grade", fuse_grade),
        ("answer", answer),
        ("finalize", finalize),
    ):
        g.add_node(name, _node(name, fn, deps))

    g.add_edge(START, "analyze")
    g.add_conditional_edges(
        "analyze",
        route_after_analyze,
        ["retrieve_dense", "retrieve_lexical", "finalize"],
    )
    # A list of sources is a join: fuse_grade runs once, after both branches finished.
    g.add_edge(["retrieve_dense", "retrieve_lexical"], "fuse_grade")
    g.add_conditional_edges("fuse_grade", route_after_fuse, ["answer", "finalize"])
    g.add_edge("answer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()

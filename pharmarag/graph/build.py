"""Compile the LangGraph state machine (spec section 6): nine nodes, one exit.

    analyze -> (declined -> finalize) -> resolve -> (stop status -> finalize)
            -> retrieve_dense | retrieve_lexical | fetch_rx_facts   (one superstep)
            -> fuse_grade -> (weak, attempt 1 -> rewrite_query -> the three branches again)
                          -> (weak, attempt 2 -> finalize)
                          -> answer -> (low support, attempt 1 -> answer) -> finalize

Longest path: analyze, resolve, branches, fuse_grade, rewrite_query, branches, fuse_grade,
answer, answer, finalize = 10 supersteps. LangGraph needs a recursion limit of 11 for that, so
the configured 12 (NFR-6) leaves one spare step.
"""

import time
from collections.abc import Awaitable, Callable
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from pharmarag.graph.deps import Deps
from pharmarag.graph.nodes.analyze import analyze
from pharmarag.graph.nodes.answer import answer
from pharmarag.graph.nodes.fetch_rx_facts import fetch_rx_facts
from pharmarag.graph.nodes.finalize import finalize
from pharmarag.graph.nodes.fuse_grade import fuse_grade
from pharmarag.graph.nodes.resolve import resolve
from pharmarag.graph.nodes.retrieve_dense import retrieve_dense
from pharmarag.graph.nodes.retrieve_lexical import retrieve_lexical
from pharmarag.graph.nodes.rewrite_query import rewrite_query
from pharmarag.graph.routing import (
    BRANCHES,
    route_after_analyze,
    route_after_answer,
    route_after_fuse,
    route_after_resolve,
)
from pharmarag.graph.state import RAGState
from pharmarag.grounding.stream import resolve_verify_mode
from pharmarag.llm.errors import LLMUnavailableError

NodeFn = Callable[[RAGState, Deps], Awaitable[dict]]

# Nodes that call the LLM get a node retry policy (spec section 6, "Two kinds of retry"). The
# provider maps timeouts, HTTP 429 and 5xx to LLMUnavailableError once its own SDK retries are
# spent; anything else (a refusal, bad JSON, a 4xx) is not transient and is not retried.
LLM_NODES = ("analyze", "rewrite_query", "answer")


def llm_retry_policy(*, initial_interval: float = 0.5, jitter: bool = True) -> RetryPolicy:
    """3 attempts, exponential backoff (x2), only on LLMUnavailableError."""
    return RetryPolicy(
        initial_interval=initial_interval,
        backoff_factor=2.0,
        max_attempts=3,
        jitter=jitter,
        retry_on=LLMUnavailableError,
    )


def _node(name: str, fn: NodeFn, deps: Deps) -> Callable[[RAGState], Awaitable[dict]]:
    """Bind deps and record the node's wall time (ms) in `timings` (NFR-5)."""

    async def run(state: RAGState) -> dict:
        t0 = time.perf_counter()
        out = await fn(state, deps)
        return {**out, "timings": {name: round((time.perf_counter() - t0) * 1000, 2)}}

    run.__name__ = name
    return run


def build_graph(deps: Deps, *, llm_retry: RetryPolicy | None = None):
    """Compile the graph. `llm_retry` is for tests that must not sleep through backoff."""
    llm_retry = llm_retry or llm_retry_policy()
    g = StateGraph(RAGState)
    for name, fn in (
        ("analyze", analyze),
        ("resolve", resolve),  # no retry policy: Lane A's RxNav client retries internally
        ("retrieve_dense", retrieve_dense),
        ("retrieve_lexical", retrieve_lexical),
        ("fetch_rx_facts", fetch_rx_facts),
        ("fuse_grade", fuse_grade),
        ("rewrite_query", rewrite_query),
        ("answer", answer),
        ("finalize", finalize),
    ):
        g.add_node(
            name,
            _node(name, fn, deps),
            retry_policy=llm_retry if name in LLM_NODES else None,
        )

    thresholds = deps.thresholds
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route_after_analyze, ["resolve", "finalize"])
    g.add_conditional_edges("resolve", route_after_resolve, [*BRANCHES, "finalize"])
    # A list of sources is a join: fuse_grade runs once, after all three branches finished.
    g.add_edge(BRANCHES, "fuse_grade")
    g.add_conditional_edges(
        "fuse_grade",
        partial(route_after_fuse, thresholds=thresholds),
        ["answer", "rewrite_query", "finalize"],
    )
    for branch in BRANCHES:
        g.add_edge("rewrite_query", branch)
    g.add_conditional_edges(
        "answer",
        partial(
            route_after_answer,
            thresholds=thresholds,
            verify_mode=resolve_verify_mode(deps.settings),
        ),
        ["answer", "finalize"],
    )
    g.add_edge("finalize", END)
    return g.compile()

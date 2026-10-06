"""Conditional edges (spec section 6, "Edges")."""

from pharmarag.graph.nodes.finalize import DECLINED_INTENTS
from pharmarag.graph.state import RAGState


def route_after_analyze(state: RAGState) -> list[str]:
    """Declined intents skip retrieval. Otherwise both retrieval branches run in one superstep.

    Cycle 2 inserts `resolve` between these two and adds `fetch_rx_facts` to the fan-out.
    """
    if state.get("intent") in DECLINED_INTENTS:
        return ["finalize"]
    return ["retrieve_dense", "retrieve_lexical"]


def route_after_fuse(state: RAGState) -> str:
    """Cycle 1: weak evidence goes straight to finalize. Cycle 2 adds the rewrite retry."""
    return "answer" if state.get("evidence_ok") else "finalize"

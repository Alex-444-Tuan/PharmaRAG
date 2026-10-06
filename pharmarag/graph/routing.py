"""Conditional edges (spec section 6, "Edges").

Each function returns node names, so the compiled graph's edge table can be read off this file.
"""

from pharmarag.app.settings import Thresholds
from pharmarag.graph.nodes.finalize import DECLINED_INTENTS
from pharmarag.graph.nodes.resolve import STOP_STATUSES
from pharmarag.graph.state import Flags, RAGState

BRANCHES = ["retrieve_dense", "retrieve_lexical", "fetch_rx_facts"]


def route_after_analyze(state: RAGState) -> list[str]:
    """Declined intents skip everything else. Otherwise the drug mentions are resolved."""
    if state.get("intent") in DECLINED_INTENTS:
        return ["finalize"]
    return ["resolve"]


def route_after_resolve(state: RAGState) -> list[str]:
    """A stop status ends the request. Otherwise the three branches run in one superstep.

    Never falls through to retrieval when `resolve` set a stop status: the branches would run
    without the ingredient_key filter (FR-3, FR-8).
    """
    if state.get("status") in STOP_STATUSES:
        return ["finalize"]
    return list(BRANCHES)


def route_after_fuse(state: RAGState, thresholds: Thresholds) -> str:
    """Evidence ok: answer. Weak on attempt 1: rewrite once. Weak on the last attempt: stop."""
    if state.get("evidence_ok"):
        return "answer"
    if state.get("retrieval_attempt", 1) < thresholds.graph.max_retrieval_attempts:
        return "rewrite_query"
    return "finalize"  # insufficient_evidence, derived in finalize


def route_after_answer(state: RAGState, thresholds: Thresholds) -> str:
    """Regenerate once when the verifier found too few supported sentences.

    SPEC: an answer with no sentences at all (the writer said INSUFFICIENT, or refused) is not
    "low support": the evidence does not answer the question, and rewriting the same prompt
    would not change that. It goes to finalize.
    """
    flags = state.get("flags") or Flags()
    ratio = state.get("support_ratio")
    if not flags.verifier or ratio is None or not state.get("sentences"):
        return "finalize"
    if ratio >= thresholds.grounding.support_ratio_min:
        return "finalize"
    if state.get("generation_attempt", 1) >= thresholds.graph.max_generation_attempts:
        return "finalize"
    return "answer"

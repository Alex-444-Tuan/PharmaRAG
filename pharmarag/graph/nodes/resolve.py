"""resolve: drug resolver node (spec sections 6 and 7).

Turns every drug mention into an ingredient key, or stops the request with a status. Nothing
downstream sees a raw drug name, and nothing downstream ranks without the key (FR-2, FR-3, FR-8).
"""

from pharmarag.graph.deps import Deps
from pharmarag.graph.events import emitter
from pharmarag.graph.nodes.finalize import drug_entries
from pharmarag.graph.state import Flags, RAGState, Resolution, Status
from pharmarag.retrieval.scope import FilterRequiredError

STOP_STATUSES: tuple[Status, ...] = ("needs_clarification", "drug_not_recognised", "not_covered")


def stop_status(resolutions: list[Resolution]) -> Status | None:
    """The status that ends the request early, or None when every mention resolved.

    SPEC: section 6 lists the three stop statuses but not which wins when a question has
    several mentions that fail differently ("ambiguous, unknown, or not in the corpus").
    Precedence: needs_clarification, then drug_not_recognised, then not_covered. Reason: an
    ambiguous name is the only outcome the client can fix by resending with a pinned RxCUI, and
    a clarification may turn an unknown or uncovered mention into something else, so ask first.
    An unrecognised name is a typo to retype, and `not_covered` is final, so it goes last.
    """
    found = {r.status for r in resolutions}
    if "ambiguous" in found:
        return "needs_clarification"
    if "unknown" in found:
        return "drug_not_recognised"
    if "not_covered" in found:
        return "not_covered"
    return None


def ingredient_keys_of(resolutions: list[Resolution]) -> list[str]:
    """One key per distinct resolved drug, in mention order ("Lipitor and atorvastatin" is one)."""
    keys = [r.ingredient_key for r in resolutions if r.status == "resolved" and r.ingredient_key]
    return list(dict.fromkeys(keys))


async def resolve(state: RAGState, deps: Deps) -> dict:
    flags = state.get("flags") or Flags()
    mentions = state.get("mentions", [])

    resolver = deps.resolver
    if not flags.resolver or resolver is None:
        # Section 7, "Baseline for AT-1": `analyze` extracted no mentions and retrieval runs
        # unfiltered whatever the intent. Reached with the flag off (eval mode) or in a build
        # without a resolver, where `run_ask` forces the flag off.
        if mentions:
            # Mentions that nothing can resolve must not become an unfiltered search (FR-3).
            raise FilterRequiredError(
                f"mentions {mentions!r} detected but they cannot be resolved (resolver flag "
                "off or no resolver in this build); refusing to continue without the filter"
            )
        return {"resolutions": [], "ingredient_keys": []}

    if not mentions:
        if state.get("intent") == "class_question":
            # Section 8, "No-mention path": the only unfiltered search.
            return {"resolutions": [], "ingredient_keys": []}
        # SPEC: section 6 sends "no mention and intent is class_question" to retrieval, and is
        # silent on a label_qa or interaction question that names no drug ("What is the max
        # dose?"). Running it unfiltered would rank text from any drug, which FR-3 forbids when
        # a drug was expected. We stop with `drug_not_recognised`: its client action ("ask the
        # user to retype") is the right one, because the user must name the drug. It also
        # covers `analyze` missing a mention. `pinned_rxcuis` alone do not help here: they pin
        # an existing mention, and the client resends the same question.
        return {"resolutions": [], "ingredient_keys": [], "status": "drug_not_recognised"}

    # SPEC: no RetryPolicy on this node. Lane A's RxNav client retries internally (3 attempts,
    # backoff), so a node policy would multiply the attempts.
    resolutions = await resolver.resolve(mentions, pinned_rxcuis=state.get("pinned_rxcuis"))
    if len(resolutions) != len(mentions):
        # A mention without a resolution would silently drop out of the filter.
        raise FilterRequiredError(
            f"resolver returned {len(resolutions)} resolutions for {len(mentions)} mentions"
        )

    # SSE: the client learns which drugs the question was matched to as soon as that is known,
    # before retrieval starts. It is sent for early exits too (clarification, not covered).
    emitter()({"event": "drugs", "data": drug_entries(resolutions)})
    out: dict = {"resolutions": resolutions, "ingredient_keys": ingredient_keys_of(resolutions)}
    status = stop_status(resolutions)
    if status:
        out["status"] = status
        out["ingredient_keys"] = []
    return out

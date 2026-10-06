"""Which `ingredient_key` filters a retrieval node runs with (FR-3)."""

from pharmarag.graph.state import RAGState


class FilterRequiredError(RuntimeError):
    """A drug mention was detected but no ingredient_key reached retrieval.

    Ranking unfiltered here would let another drug's text into the answer (FR-3). This is a
    graph wiring bug, not a user error: `resolve` must stop the request before retrieval.
    """


def scopes(state: RAGState) -> list[str | None]:
    """One entry per retrieval query: an ingredient_key, or None for the no-mention path.

    None is returned only when no drug was mentioned at all.
    """
    keys = state.get("ingredient_keys") or []
    if keys:
        return list(keys)
    if state.get("mentions"):
        raise FilterRequiredError(
            f"mentions {state['mentions']!r} detected but no ingredient_keys; refusing to "
            "rank without the ingredient_key filter"
        )
    return [None]

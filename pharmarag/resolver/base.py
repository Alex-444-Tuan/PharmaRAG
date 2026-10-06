"""Resolver and RxNorm-facts interfaces (spec sections 6, 7, 8). Contract file.

Lane A implements these in pharmarag/resolver/. The graph's `resolve` and
`fetch_rx_facts` nodes depend only on these protocols.
"""

from typing import Protocol

from pharmarag.graph.state import Resolution


class Resolver(Protocol):
    async def resolve(
        self, mentions: list[str], *, pinned_rxcuis: list[str] | None = None
    ) -> list[Resolution]:
        """One Resolution per mention, in order (spec section 7 cascade and gate).

        `pinned_rxcuis` are ingredient RxCUIs the client sent after a clarification. A
        mention whose candidates include a pinned ingredient resolves to it with
        match="pinned". A resolved drug with no label in the corpus has status "not_covered".
        """
        ...


class RxFactsSource(Protocol):
    async def facts(self, ingredient_rxcuis: list[str]) -> list[str]:
        """Short fact lines (related brands, dose forms, strengths) for the given
        ingredients, at most `thresholds.retrieval.max_facts` in total, deterministic order.
        The graph node assigns ids r1, r2, ... and the retrieval attempt.
        """
        ...

"""Grounding verifier interface (spec section 9). Contract file.

Cycle 2 wires the regeneration edge against this protocol with a fake.
Cycle 3 implements it (NLI cross-encoder + number check) and adds stream mode.
"""

from typing import Protocol

from pharmarag.graph.state import Chunk, Fact, Sentence


class Verifier(Protocol):
    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        """Return the same sentences, in order, each with verdict, score, span, and reason set.

        A sentence that already has verdict "unsupported" (e.g. reason "no_citation") is
        returned unchanged.
        """
        ...

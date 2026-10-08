"""Grounding verifier interface (spec section 9). Contract file.

Lane A implements it (candidate spans, number check, NLI cross-encoder).
Lane B drives it: `verify` for VERIFY_MODE=sequential, and `prepare` plus
`verify_sentence` for VERIFY_MODE=stream, where each completed line is checked
while the LLM writes the next one.
"""

from typing import Protocol

from pharmarag.graph.state import Chunk, Fact, Sentence


class PreparedEvidence(Protocol):
    """Opaque per-request evidence index built by `prepare`: span texts, span embeddings,
    and whatever else the verifier needs. Built once per generation, reused per sentence."""


class Verifier(Protocol):
    async def prepare(self, evidence: list[Chunk], facts: list[Fact]) -> PreparedEvidence:
        """Index the evidence spans (incl. bi-encoder span embeddings). Spec section 9:
        start it as soon as evidence is final so it overlaps the LLM's time to first token.
        Must not block the event loop (run model work in a thread)."""
        ...

    async def verify_sentence(self, sentence: Sentence, prepared: PreparedEvidence) -> Sentence:
        """Return `sentence` with verdict, score, span, and reason set. A sentence that is
        already "unsupported" (e.g. reason "no_citation") is returned unchanged.
        Safe to call concurrently for different sentences."""
        ...

    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        """Sequential convenience: prepare once, verify every sentence, keep order."""
        ...

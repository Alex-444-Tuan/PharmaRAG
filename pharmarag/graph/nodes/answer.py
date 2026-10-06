"""answer: writer and grounding verifier node (spec section 9).

The writer runs, then (when the verifier is on) every sentence gets a verdict. If too few
sentences are supported the graph loops back here once, with the rejected sentences listed as
"do not repeat". This node does not drop anything itself: `finalize` is the one gate that
decides what the user sees (FR-6).

`generation_attempt` is the number of the attempt whose sentences are in state. The state
starts it at 1, so "this is attempt 1" is told apart from "attempt 1 is done" by whether
`sentences` exists yet.
"""

from pharmarag.graph.deps import Deps
from pharmarag.graph.state import Flags, RAGState, Sentence
from pharmarag.grounding.writer import write_answer
from pharmarag.llm.errors import LLMRefusalError


def verifier_on(state: RAGState, deps: Deps) -> bool:
    return (state.get("flags") or Flags()).verifier and deps.verifier is not None


def support_ratio(sentences: list[Sentence]) -> float:
    """Supported sentences over all sentences (spec section 9). 0.0 for no sentences."""
    if not sentences:
        return 0.0
    return sum(s.verdict == "supported" for s in sentences) / len(sentences)


async def answer(state: RAGState, deps: Deps) -> dict:
    attempt = (state.get("generation_attempt", 1) if "sentences" in state else 0) + 1
    if attempt > deps.thresholds.graph.max_generation_attempts:
        raise AssertionError("answer reached after the last generation attempt (NFR-6)")

    retrieval_attempt = state.get("retrieval_attempt", 1)
    facts = [f for f in state.get("rx_facts", []) if f.retrieval_attempt == retrieval_attempt][
        : deps.thresholds.retrieval.max_facts
    ]
    evidence = state.get("evidence", [])

    out: dict = {"generation_attempt": attempt}
    rejected: list[Sentence] = []
    if attempt > 1:
        # SPEC: the second attempt replaces the first one's sentences. The first attempt's
        # rejected sentences move to `removed_sentences` (debug payload and meta count); its
        # supported sentences are not carried over, because the regenerated answer is a new
        # draft and merging two drafts would repeat or contradict itself.
        previous = state.get("sentences", [])
        rejected = [s for s in previous if s.verdict != "supported"]
        out["removed_sentences"] = [*state.get("removed_sentences", []), *rejected]

    try:
        result = await write_answer(
            deps.llm,
            state["question"],
            evidence,
            facts,
            temperature=deps.settings.llm_temperature,
            max_tokens=deps.settings.llm_max_tokens,
            rejected=[s.text for s in rejected],
        )
    except LLMRefusalError:
        # No sentences means `insufficient_evidence` in finalize; nothing unverified leaks. On
        # the second attempt keep what the first attempt got verified: that is still true.
        kept = [s for s in state.get("sentences", []) if s.verdict == "supported"]
        return {**out, "sentences": kept if attempt > 1 else []}

    sentences = result.sentences
    if sentences and verifier_on(state, deps):
        assert deps.verifier is not None
        verified = await deps.verifier.verify(sentences, evidence, facts)
        if len(verified) != len(sentences):
            raise RuntimeError(
                f"verifier returned {len(verified)} verdicts for {len(sentences)} sentences"
            )
        sentences = verified
        out["support_ratio"] = support_ratio(sentences)
    # SPEC: with the verifier off, `support_ratio` is not computed and nothing regenerates:
    # there are no verdicts to be low.
    return {**out, "sentences": sentences}

"""answer: writer and grounding verifier node (spec section 9).

The writer runs, and (when the verifier is on) every sentence gets a verdict, in one of three
modes (`VERIFY_MODE`, see `pharmarag.grounding.stream`). If too few sentences are supported the
graph loops back here once, with the rejected sentences listed as "do not repeat". This node does
not drop anything itself: `finalize` is the one gate that decides what the user sees (FR-6).

`generation_attempt` is the number of the attempt whose sentences are in state. The state
starts it at 1, so "this is attempt 1" is told apart from "attempt 1 is done" by whether
`sentences` exists yet.

SSE (`stream: true`). The node also publishes `sentence` events, through custom streaming.
Only sentences `finalize` will show are ever published (FR-6): the same `split_sentences` rule
picks them. Stream mode publishes each sentence the moment it is verified. The other modes
publish after the attempt, and only when no regeneration follows, so a draft that is about to be
replaced is never shown.
"""

from pharmarag.app.settings import Thresholds
from pharmarag.graph.deps import Deps
from pharmarag.graph.events import emitter, is_sse
from pharmarag.graph.nodes.finalize import answer_item, split_sentences
from pharmarag.graph.state import Flags, RAGState, Sentence
from pharmarag.grounding.stream import VerifyMode, generate, resolve_verify_mode
from pharmarag.llm.errors import LLMRefusalError


def verifier_on(state: RAGState, deps: Deps) -> bool:
    return (state.get("flags") or Flags()).verifier and deps.verifier is not None


def request_verify_mode(state: RAGState, deps: Deps) -> VerifyMode:
    """The mode this request runs in. `off` when the request flag says so or there is no
    verifier. Anything else is the configured mode."""
    if not verifier_on(state, deps):
        return "off"
    return resolve_verify_mode(deps.settings)


def support_ratio(sentences: list[Sentence]) -> float:
    """Supported sentences over all sentences (spec section 9). 0.0 for no sentences."""
    if not sentences:
        return 0.0
    return sum(s.verdict == "supported" for s in sentences) / len(sentences)


def wants_regeneration(
    *,
    verifier_flag: bool,
    ratio: float | None,
    sentences: list[Sentence],
    attempt: int,
    thresholds: Thresholds,
    released_early: bool,
) -> bool:
    """Regenerate once when the verifier found too few supported sentences (spec section 6).

    The one rule shared by the `route_after_answer` edge and the node (which must know, before
    it returns, whether its sentences are final).

    SPEC: an answer with no sentences at all (the writer said INSUFFICIENT, or refused) is not
    "low support": the evidence does not answer the question, and rewriting the same prompt
    would not change that. It goes to finalize.

    SPEC (SSE): `released_early` is true for an SSE request in stream mode, where supported
    sentences have already reached the client. A new draft would then either contradict the
    `done` payload or repeat what was shown, so no regeneration happens once a supported
    sentence has been released. Nothing is released when every sentence of attempt 1 was
    rejected, and that case regenerates as usual. A non-SSE request is not affected, so the
    final answers of the three modes stay equal.
    """
    if not verifier_flag or ratio is None or not sentences:
        return False
    if ratio >= thresholds.grounding.support_ratio_min:
        return False
    if attempt >= thresholds.graph.max_generation_attempts:
        return False
    if released_early and any(s.verdict == "supported" for s in sentences):
        return False
    return True


def releases_early(mode: VerifyMode, sse: bool) -> bool:
    return sse and mode == "stream"


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
    verified_before: list[Sentence] = []  # attempt 1's supported sentences, if this is attempt 2
    if attempt > 1:
        # SPEC: a second attempt that writes sentences replaces the first one's. The first
        # attempt's rejected sentences move to `removed_sentences` (debug payload and meta
        # count); its supported sentences are not merged in, because the regenerated answer is
        # a new draft and merging two drafts would repeat or contradict itself. If the second
        # attempt writes nothing (a refusal, or INSUFFICIENT), the first attempt's supported
        # sentences are kept: they were verified, and a failed retry does not make them false.
        previous = state.get("sentences", [])
        rejected = [s for s in previous if s.verdict != "supported"]
        verified_before = [s for s in previous if s.verdict == "supported"]
        out["removed_sentences"] = [*state.get("removed_sentences", []), *rejected]

    mode = request_verify_mode(state, deps)
    sse = is_sse()
    emit = emitter()
    early = releases_early(mode, sse)
    flags = state.get("flags") or Flags()
    sent = 0  # sentence events published so far in this attempt

    def publish(items: list[Sentence]) -> None:
        nonlocal sent
        for sentence in items:
            emit({"event": "sentence", "data": {"index": sent, **answer_item(sentence)}})
            sent += 1

    try:
        generation = await generate(
            deps.llm,
            deps.verifier if mode != "off" else None,
            mode=mode,
            question=state["question"],
            evidence=evidence,
            facts=facts,
            temperature=deps.settings.llm_temperature,
            max_tokens=deps.settings.llm_max_tokens,
            rejected=[s.text for s in rejected],
            release=(lambda s: publish([s])) if early else None,
        )
    except LLMRefusalError:
        # No sentences means `insufficient_evidence` in finalize; nothing unverified leaks.
        if sse and not early:
            publish(split_sentences(verified_before, verifier_on=flags.verifier)[0])
        return {**out, "sentences": verified_before}

    sentences = generation.sentences
    if not sentences:
        if sse and not early:
            publish(split_sentences(verified_before, verifier_on=flags.verifier)[0])
        return {**out, "sentences": verified_before}  # same rule as a refusal
    if mode != "off":
        out["support_ratio"] = support_ratio(sentences)
    # SPEC: with the verifier off, `support_ratio` is not computed and nothing regenerates:
    # there are no verdicts to be low.
    if sse and not early and not wants_regeneration(
        verifier_flag=mode != "off",
        ratio=out.get("support_ratio"),
        sentences=sentences,
        attempt=attempt,
        thresholds=deps.thresholds,
        released_early=False,
    ):
        publish(split_sentences(sentences, verifier_on=flags.verifier)[0])
    return {**out, "sentences": sentences}

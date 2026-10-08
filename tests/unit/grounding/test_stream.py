"""generate(): off / sequential / stream (spec section 9, "Latency design"). No network."""

import asyncio
import time

import pytest

from pharmarag.app.settings import Settings
from pharmarag.graph.state import Sentence
from pharmarag.grounding.stream import generate, resolve_verify_mode
from pharmarag.llm.errors import (
    TRUNCATION_MARKER,
    LLMRefusalError,
    LLMUnavailableError,
)
from pharmarag.llm.replay import Recording, ReplayLLM
from tests.unit.grounding.test_writer import EVIDENCE, FACTS

Q = "What dose?"


def recorded(pieces: list[tuple[str, float]]) -> ReplayLLM:
    rec = Recording(id="q", question=Q, analysis="{}", analysis_ms=0.0, pieces=pieces)
    return ReplayLLM([rec])


class TimedVerifier:
    """Verifies after a delay and logs when each step started and ended (perf_counter, s)."""

    def __init__(self, delay_s: float = 0.0, reject: tuple[str, ...] = (), prepare_s: float = 0.0):
        self.delay_s, self.reject, self.prepare_s = delay_s, reject, prepare_s
        self.log: list[tuple[str, str, float]] = []
        self.sequential_calls = 0

    def _mark(self, what: str, text: str) -> None:
        self.log.append((what, text, time.perf_counter()))

    def at(self, what: str, text: str) -> float:
        return next(t for w, x, t in self.log if w == what and x == text)

    async def prepare(self, evidence, facts):
        self._mark("prepare_start", "")
        await asyncio.sleep(self.prepare_s)
        self._mark("prepare_end", "")
        return "prepared"

    async def verify_sentence(self, sentence: Sentence, prepared) -> Sentence:
        assert prepared == "prepared", "verify_sentence ran before prepare finished"
        if sentence.verdict != "unverified":
            return sentence
        self._mark("start", sentence.text)
        await asyncio.sleep(self.delay_s)
        self._mark("end", sentence.text)
        bad = any(r in sentence.text for r in self.reject)
        return sentence.model_copy(
            update={
                "verdict": "unsupported" if bad else "supported",
                "span": None if bad else "c1:0",
                "score": 0.1 if bad else 0.9,
            }
        )

    async def verify(self, sentences, evidence, facts):
        self.sequential_calls += 1
        prepared = await self.prepare(evidence, facts)
        return [await self.verify_sentence(s, prepared) for s in sentences]


async def run(llm, verifier, mode, *, release=None, **kw):
    return await generate(
        llm,
        verifier,
        mode=mode,
        question=Q,
        evidence=EVIDENCE,
        facts=FACTS,
        temperature=0.0,
        max_tokens=500,
        release=release,
        **kw,
    )


# --- concurrency --------------------------------------------------------------------------------

THREE_LINES = [
    ("Alpha is true. [c1]\n", 10.0),
    ("Beta is true. [c1]\n", 300.0),
    ("Gamma is true. [c1]", 600.0),
]


async def test_verification_of_line_one_overlaps_the_generation_of_line_two():
    """Event ordering on a recorded stream: line 1's check starts and ends while the LLM is still
    writing, and the answer is done long before verify-after-generation would be."""
    verifier = TimedVerifier(delay_s=0.2)
    t0 = time.perf_counter()
    gen = await run(recorded(THREE_LINES), verifier, "stream")
    total = time.perf_counter() - t0
    stream_end = t0 + 0.6  # the recording's last token

    assert [s.verdict for s in gen.sentences] == ["supported"] * 3
    assert verifier.at("start", "Alpha is true.") < t0 + 0.45  # begun while Beta is being written
    assert verifier.at("end", "Alpha is true.") < stream_end - 0.03  # and finished before the LLM
    # Only the last sentence is checked after the stream: 0.6 s writing + at most the checks of
    # Beta and Gamma (0.4 s). Verify-after-generation needs 0.6 + 3 x 0.2 = 1.2 s.
    assert total < 1.12


async def test_sequential_mode_verifies_nothing_until_the_stream_ended():
    verifier = TimedVerifier(delay_s=0.1)
    t0 = time.perf_counter()
    gen = await run(recorded(THREE_LINES), verifier, "sequential")
    assert verifier.sequential_calls == 1
    assert verifier.at("start", "Alpha is true.") >= t0 + 0.6 - 0.01
    assert [s.verdict for s in gen.sentences] == ["supported"] * 3


async def test_the_verifier_runs_while_the_llm_is_still_writing_not_after_it():
    """Deterministic version of the overlap proof, with no wall-clock bounds: the LLM refuses to
    write line three until line one is being verified. If verification only started after the
    stream ended, this would never get past the gate."""
    started = asyncio.Event()

    class GatedLLM:
        model_id = "gated"

        async def stream(self, messages, **kw):
            yield "Alpha is true. [c1]\n"
            yield "Beta is true. [c1]\n"  # Alpha is released as soon as Beta starts
            await asyncio.wait_for(started.wait(), timeout=2)  # waits for Alpha's check
            yield "Gamma is true. [c1]"

    class Signalling(TimedVerifier):
        async def verify_sentence(self, sentence, prepared):
            if sentence.text == "Alpha is true.":
                started.set()
            return await super().verify_sentence(sentence, prepared)

    gen = await run(GatedLLM(), Signalling(), "stream")
    assert len(gen.sentences) == 3


async def test_prepare_starts_before_the_first_token_and_is_awaited_before_the_first_check():
    verifier = TimedVerifier(prepare_s=0.15)
    llm = recorded([("Alpha is true. [c1]", 20.0)])
    t0 = time.perf_counter()
    gen = await run(llm, verifier, "stream")
    assert verifier.at("prepare_start", "") - t0 < 0.01  # not after the writer
    assert verifier.at("prepare_end", "") <= verifier.at("start", "Alpha is true.")
    # Overlap: prepare (150 ms) ran alongside the 20 ms wait for the token, not after it.
    assert time.perf_counter() - t0 < 0.15 + 0.1
    assert gen.sentences[0].verdict == "supported"


# --- equivalence of the modes -------------------------------------------------------------------

REPLIES = [
    "Alpha is true. [c1]\nBeta is false. [c2]\nGamma is true. [c1][r1]",
    "Alpha is true. [c1] Beta is false. [c2]\nGamma has no citation.\nDelta. [c9]",
    "INSUFFICIENT",
    "- Alpha is true. [c1]\n- Beta is false. [c2]",
    f"Alpha is true. [c1]\nBeta is false. [c2]\nGamma is cut of{TRUNCATION_MARKER}",
    f"Alpha is true. [c1]\n{TRUNCATION_MARKER}",
]


@pytest.mark.parametrize("reply", REPLIES)
async def test_sequential_and_stream_give_the_same_final_sentences(reply, make_fake_llm):
    """Same reply, same verdicts: the three modes differ in timing only."""
    results = {}
    for mode in ("off", "sequential", "stream"):
        verifier = TimedVerifier(reject=("false",))
        results[mode] = await run(make_fake_llm([reply]), verifier, mode)
    assert results["sequential"].sentences == results["stream"].sentences
    assert results["sequential"].truncated == results["stream"].truncated
    # `off` writes the same sentences, unchecked.
    assert [s.text for s in results["off"].sentences] == [
        s.text for s in results["stream"].sentences
    ]
    assert all(
        s.verdict in ("unverified", "unsupported") for s in results["off"].sentences
    )  # no_citation ones are already unsupported; nothing is marked supported


async def test_off_mode_needs_no_verifier_and_never_calls_it(make_fake_llm):
    verifier = TimedVerifier()
    gen = await run(make_fake_llm(["Alpha. [c1]"]), verifier, "off")
    assert gen.sentences[0].verdict == "unverified" and verifier.log == []
    gen = await run(make_fake_llm(["Alpha. [c1]"]), None, "stream")  # no verifier: same as off
    assert gen.sentences[0].verdict == "unverified"


# --- release (SSE) ------------------------------------------------------------------------------


async def test_release_is_called_in_order_only_for_supported_sentences_and_while_writing(
    make_fake_llm,
):
    released: list[tuple[str, float]] = []
    verifier = TimedVerifier(delay_s=0.02, reject=("false",))
    reply = "Alpha is true. [c1]\nBeta is false. [c2]\nGamma is true. [c1]"
    t0 = time.perf_counter()
    gen = await run(
        recorded([(reply[:20], 10.0), (reply[20:40], 250.0), (reply[40:], 500.0)]),
        verifier,
        "stream",
        release=lambda s: released.append((s.text, time.perf_counter() - t0)),
    )
    assert [t for t, _ in released] == ["Alpha is true.", "Gamma is true."]
    assert gen.released == 2
    assert released[0][1] < 0.45  # Alpha reached the client before the LLM finished (0.5 s)


async def test_sentences_failing_the_check_are_never_released(make_fake_llm):
    released: list[Sentence] = []
    verifier = TimedVerifier(reject=("Alpha", "Beta", "Gamma"))
    gen = await run(
        make_fake_llm(["Alpha. [c1]\nBeta. [c1]\nGamma. [c1]\nUncited."]),
        verifier,
        "stream",
        release=released.append,
    )
    assert released == [] and gen.released == 0
    assert [s.verdict for s in gen.sentences] == ["unsupported"] * 4


async def test_sequential_mode_releases_nothing_here(make_fake_llm):
    released: list[Sentence] = []
    gen = await run(
        make_fake_llm(["Alpha. [c1]"]), TimedVerifier(), "sequential", release=released.append
    )
    assert released == [] and gen.released == 0


# --- errors -------------------------------------------------------------------------------------


class DyingLLM:
    """Writes `lines`, then raises `error` (a connection dropping mid-answer)."""

    model_id = "dying"

    def __init__(self, text: str, error: Exception):
        self.text, self.error = text, error

    async def stream(self, messages, **kw):
        for piece in self.text.split(" "):
            yield piece + " "
        raise self.error


async def test_a_mid_stream_outage_before_any_release_propagates_for_the_node_retry():
    # The first line is still being written when the connection drops: nothing was verified,
    # so nothing could have been shown.
    llm = DyingLLM("Alpha is tr", LLMUnavailableError("reset"))
    with pytest.raises(LLMUnavailableError):
        await run(llm, TimedVerifier(), "stream", release=lambda s: None)


async def test_a_mid_stream_outage_without_an_sse_listener_always_propagates():
    # Alpha is complete and verified when the outage hits, but with no listener it was shown to
    # nobody, so the whole attempt can be retried from scratch (spec section 6, RetryPolicy).
    llm = DyingLLM("Alpha is true. [c1]\nBeta is tr", LLMUnavailableError("reset"))
    with pytest.raises(LLMUnavailableError):
        await run(llm, TimedVerifier(), "stream")


async def test_a_rejected_sentence_does_not_count_as_released():
    # Alpha was checked and rejected, so the client saw nothing: the retry rule is "no sentence
    # released yet", not "no sentence verified yet".
    llm = DyingLLM("Alpha is false. [c1]\nBeta is tr", LLMUnavailableError("reset"))
    with pytest.raises(LLMUnavailableError):
        await run(llm, TimedVerifier(reject=("false",)), "stream", release=lambda s: None)


async def test_a_mid_stream_outage_after_a_release_finalises_with_what_was_verified():
    released: list[str] = []
    llm = DyingLLM("Alpha is true. [c1]\nBeta is true. [c1]\nGamma is tr", LLMUnavailableError("x"))
    gen = await run(llm, TimedVerifier(), "stream", release=lambda s: released.append(s.text))
    assert gen.interrupted and gen.released == len(released) >= 1
    # Complete lines that were already written are verified too; the cut-off one is dropped.
    assert [s.text for s in gen.sentences][: len(released)] == released
    assert all(s.verdict == "supported" for s in gen.sentences)
    assert "Gamma is tr" not in [s.text for s in gen.sentences]


async def test_a_refusal_after_a_release_keeps_the_released_sentences():
    llm = DyingLLM("Alpha is true. [c1]\nBeta is true. [c1]\nGamma", LLMRefusalError("no"))
    gen = await run(llm, TimedVerifier(), "stream", release=lambda s: None)
    assert gen.interrupted and len(gen.sentences) >= 1


async def test_a_refusal_before_any_release_propagates():
    llm = DyingLLM("Alpha is tr", LLMRefusalError("no"))
    with pytest.raises(LLMRefusalError):
        await run(llm, TimedVerifier(), "stream", release=lambda s: None)


async def test_a_failing_verifier_stops_the_llm_call_and_raises():
    cancelled = asyncio.Event()

    class SlowLLM:
        model_id = "slow"

        async def stream(self, messages, **kw):
            try:
                yield "Alpha is true. [c1]\n"
                yield "Beta is true. [c1]\n"
                yield "Gamma is true. [c1]"
                await asyncio.sleep(30)  # would hang the request if nobody stopped the writer
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class Broken(TimedVerifier):
        async def verify_sentence(self, sentence, prepared):
            raise RuntimeError("NLI model crashed")

    with pytest.raises(RuntimeError, match="NLI model crashed"):
        await asyncio.wait_for(run(SlowLLM(), Broken(), "stream"), timeout=5)
    assert cancelled.is_set()


async def test_a_failing_prepare_raises_and_stops_the_llm(make_fake_llm):
    class BadPrepare(TimedVerifier):
        async def prepare(self, evidence, facts):
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        await run(make_fake_llm(["Alpha. [c1]"]), BadPrepare(), "stream")


async def test_cancelling_the_request_cancels_the_writer_and_prepare():
    seen = {"writer": False, "prepare": False}

    class HangingLLM:
        model_id = "hang"

        async def stream(self, messages, **kw):
            try:
                await asyncio.sleep(30)
                yield "never"
            except asyncio.CancelledError:
                seen["writer"] = True
                raise

    class HangingPrepare(TimedVerifier):
        async def prepare(self, evidence, facts):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                seen["prepare"] = True
                raise

    task = asyncio.create_task(run(HangingLLM(), HangingPrepare(), "stream"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == {"writer": True, "prepare": True}


# --- the mode setting ---------------------------------------------------------------------------


def test_verify_mode_off_is_honoured_only_in_eval_mode():
    prod = Settings(_env_file=None, eval_mode=False, verify_mode="off")
    dev = Settings(_env_file=None, eval_mode=True, verify_mode="off")
    assert resolve_verify_mode(prod) == "stream"  # production can never run unverified (FR-6)
    assert resolve_verify_mode(dev) == "off"
    assert resolve_verify_mode(Settings(_env_file=None, verify_mode="sequential")) == "sequential"

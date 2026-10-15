"""Generate and verify an answer in one of three modes (spec section 9, "Latency design").

    off         the writer only. The AT-2 baseline.
    sequential  write the whole answer, then verify every sentence. Exists to measure what the
                verifier would cost without the in-stream design.
    stream      the writer and the verifier are two coroutines joined by an `asyncio.Queue`. The
                writer pushes each completed sentence; the verifier checks it while the LLM keeps
                writing the next one. Only the last sentence's check adds wall-clock time.

All three read the LLM through `llm.stream`, so the time to first token is the same in every mode
and the latency comparison isolates the verifier. They return the same sentences for the same
reply (the line splitting is one shared class), which is what the equivalence test pins.
"""

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pharmarag.app.settings import Settings
from pharmarag.graph.state import Chunk, Fact, Sentence
from pharmarag.grounding.base import PreparedEvidence, Verifier
from pharmarag.grounding.writer import (
    STRICT_ADDENDUM,
    SYSTEM_PROMPT,
    LineSplitter,
    build_messages,
    parse_line,
    parse_writer_output,
    valid_ids,
)
from pharmarag.llm.base import LLMClient
from pharmarag.llm.errors import LLMRefusalError, LLMUnavailableError

VerifyMode = Literal["off", "sequential", "stream"]


def resolve_verify_mode(settings: Settings) -> VerifyMode:
    """The mode the settings ask for, with `off` honoured only in EVAL_MODE.

    Outside EVAL_MODE nobody may switch verification off (FR-6): the same rule as the request
    flags. `build_runtime` refuses to start in that combination; this is the second lock for a
    caller that built its deps by hand.
    """
    if settings.verify_mode == "off" and not settings.eval_mode:
        return "stream"
    return settings.verify_mode


@dataclass
class Generation:
    sentences: list[Sentence] = field(default_factory=list)
    truncated: bool = False
    # Sentences handed to `release` while the LLM was still writing (stream mode only).
    released: int = 0
    # The LLM failed after sentences were already released, so the answer was finalised with
    # what had been verified up to that point.
    interrupted: bool = False


async def _collect(
    llm: LLMClient,
    messages,
    system: str,
    temperature: float,
    max_tokens: int,
) -> str:
    return "".join(
        [
            piece
            async for piece in llm.stream(
                messages, system=system, temperature=temperature, max_tokens=max_tokens
            )
        ]
    )


async def _cancel(*tasks: asyncio.Task) -> None:
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t


async def generate(
    llm: LLMClient,
    verifier: Verifier | None,
    *,
    mode: VerifyMode,
    question: str,
    evidence: Sequence[Chunk],
    facts: Sequence[Fact],
    temperature: float,
    max_tokens: int,
    rejected: Sequence[str] = (),
    release: Callable[[Sentence], None] | None = None,
) -> Generation:
    """Write the answer and, unless `mode` is off, give every sentence a verdict.

    `release` is called with each sentence the moment it is verified as supported, in order,
    while the LLM is still writing (stream mode). It is how SSE hands sentences to the client.
    In the other modes nothing is released here: the caller decides after the whole attempt.

    Errors. A refusal or `LLMUnavailableError` from the LLM propagates, so the node can treat a
    refusal as "no sentences" and the graph's RetryPolicy can retry an outage, unless sentences
    were already released. A released sentence cannot be recalled, so in that case the error is
    swallowed and the result holds what was verified (`interrupted`). The partial line being
    written when the error hit is dropped.
    """
    messages = build_messages(question, evidence, facts, rejected)
    system = SYSTEM_PROMPT + STRICT_ADDENDUM if rejected else SYSTEM_PROMPT

    if mode == "off" or verifier is None:
        raw = await _collect(llm, messages, system, temperature, max_tokens)
        parsed = parse_writer_output(raw, evidence, facts)
        return Generation(sentences=parsed.sentences, truncated=parsed.truncated)

    if mode == "sequential":
        raw = await _collect(llm, messages, system, temperature, max_tokens)
        parsed = parse_writer_output(raw, evidence, facts)
        verified = await verifier.verify(parsed.sentences, list(evidence), list(facts))
        if len(verified) != len(parsed.sentences):
            raise RuntimeError(
                f"verifier returned {len(verified)} verdicts for {len(parsed.sentences)} sentences"
            )
        return Generation(sentences=verified, truncated=parsed.truncated)

    return await _generate_stream(
        llm,
        verifier,
        messages,
        system,
        evidence,
        facts,
        temperature=temperature,
        max_tokens=max_tokens,
        release=release,
    )


async def _generate_stream(
    llm: LLMClient,
    verifier: Verifier,
    messages,
    system: str,
    evidence: Sequence[Chunk],
    facts: Sequence[Fact],
    *,
    temperature: float,
    max_tokens: int,
    release: Callable[[Sentence], None] | None,
) -> Generation:
    valid = valid_ids(evidence, facts)
    queue: asyncio.Queue[Sentence | None] = asyncio.Queue()
    splitter = LineSplitter()
    verified: list[Sentence] = []
    released = 0

    # Starts now, before the first token is requested, so the span embeddings are computed
    # while the LLM is still working out its first token. Awaited before the first verdict.
    prepare_task = asyncio.create_task(verifier.prepare(list(evidence), list(facts)))

    async def writer() -> None:
        try:
            async for piece in llm.stream(
                messages, system=system, temperature=temperature, max_tokens=max_tokens
            ):
                for line in splitter.feed(piece):
                    for sentence in parse_line(line, valid):
                        queue.put_nowait(sentence)
            for line in splitter.finish():
                for sentence in parse_line(line, valid):
                    queue.put_nowait(sentence)
        finally:
            # Always end the queue, also on error, so the verifier drains what was written and
            # stops instead of waiting forever.
            queue.put_nowait(None)

    async def checker() -> None:
        # One batch per wake-up: the line that was waited for, plus every line the writer queued
        # in the meantime. The real API delivers lines in bursts, so 3 or 4 sentences can be
        # waiting when the stream ends; checking them one by one put N model calls after the last
        # token. Now it is one NLI batch (`verify_many`). Order and FR-6 are untouched: verdicts
        # come back in order, and only `supported` ones are released, in order.
        nonlocal released
        prepared: PreparedEvidence = await prepare_task
        done = False
        while not done:
            first = await queue.get()
            if first is None:
                return
            batch = [first]
            while not queue.empty():
                nxt = queue.get_nowait()
                if nxt is None:
                    done = True
                    break
                batch.append(nxt)
            todo = [s for s in batch if s.verdict == "unverified"]
            if len(todo) > 1:
                checked = await verifier.verify_many(todo, prepared)
                if len(checked) != len(todo):
                    raise RuntimeError(
                        f"verifier returned {len(checked)} verdicts for {len(todo)} sentences"
                    )
            elif todo:
                checked = [await verifier.verify_sentence(todo[0], prepared)]
            else:
                checked = []
            results = iter(checked)
            for sentence in batch:
                if sentence.verdict == "unverified":
                    sentence = next(results)
                verified.append(sentence)
                if sentence.verdict == "supported" and release is not None:
                    release(sentence)
                    released += 1

    writer_task = asyncio.create_task(writer())
    checker_task = asyncio.create_task(checker())
    try:
        await checker_task  # ends when the writer has put its end marker and the queue is empty
    except BaseException:
        # The verifier failed, or the request was cancelled (client gone): stop the LLM call.
        await _cancel(writer_task, prepare_task)
        raise
    try:
        await writer_task
    except (LLMUnavailableError, LLMRefusalError):
        if released == 0:
            raise
        return Generation(sentences=verified, released=released, interrupted=True)
    finally:
        if not prepare_task.done():
            await _cancel(prepare_task)
    return Generation(sentences=verified, truncated=splitter.truncated, released=released)

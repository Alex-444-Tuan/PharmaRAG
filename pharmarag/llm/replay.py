"""Record a real LLM run and replay it with its original timing (spec section 11.3, "Replay mode").

The latency eval must compare VERIFY_MODE off / sequential / stream without LLM network noise
deciding the result. `RecordingLLM` wraps a real client once and stores, per question, the
analyze reply and the writer's text pieces with their arrival times. `ReplayLLM` then plays them
back: the same text, the same gaps between tokens, no network, no API key. Because every mode
replays the identical token stream, any latency difference between modes is the verifier's.

A recording is looked up by the question text inside the prompt, so replay does not depend on
call order and several requests may run at once.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from pharmarag.llm.base import LLMClient, Message

T = TypeVar("T", bound=BaseModel)

_QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)


class NotRecordedError(LookupError):
    """Replay was asked for something the recording does not contain."""


@dataclass
class Recording:
    """One question's recorded LLM traffic."""

    id: str
    question: str
    analysis: str  # the structured reply, as JSON text
    analysis_ms: float  # how long the real analyze call took
    # (text piece, milliseconds since the writer call started) in arrival order. The first
    # entry's time is the time to first token.
    pieces: list[tuple[str, float]] = field(default_factory=list)
    model_id: str = ""
    corpus_version: str = ""
    recorded_at: str = ""

    @property
    def text(self) -> str:
        return "".join(p for p, _ in self.pieces)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Recording:
        raw = json.loads(line)
        raw["pieces"] = [(p, float(t)) for p, t in raw["pieces"]]
        return cls(**raw)


def load_recordings(path: Path) -> list[Recording]:
    return [
        Recording.from_json(line)
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def question_of(messages: Sequence[Message]) -> str | None:
    """The question inside a `<question>` block of the last user message, if any."""
    for m in reversed(messages):
        found = _QUESTION.search(m["content"])
        if found:
            return found.group(1)
    return None


class ReplayLLM:
    """Plays recorded traffic back with the recorded timing. Satisfies `LLMClient`.

    `structured` returns the recorded analysis (after the recorded delay if
    `replay_analysis_delay`). `stream` yields the recorded pieces, each at its recorded offset
    from the start of the call. Offsets are absolute, not accumulated sleeps, so scheduling
    error does not drift over a long answer. `speed` > 1 plays faster (tests only).

    Calls with no matching recording raise `NotRecordedError`: a replay never invents text.
    A regenerated request replays the same writer text again, which is what the writer would
    most likely say to the same evidence; the recorded questions are chosen so that it is rare.
    """

    def __init__(
        self,
        recordings: Sequence[Recording],
        *,
        model_id: str = "replay",
        replay_analysis_delay: bool = True,
        speed: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self._recordings = list(recordings)
        self.replay_analysis_delay = replay_analysis_delay
        self.speed = speed
        self.calls: list[dict] = []

    def _find(self, messages: Sequence[Message]) -> Recording:
        question = question_of(messages)
        matches = [r for r in self._recordings if question is not None and r.question == question]
        if not matches:
            raise NotRecordedError(f"no recording for question {question!r}")
        return matches[0]

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        self.calls.append({"kind": "stream", "messages": list(messages), "system": system})
        rec = self._find(messages)
        start = time.perf_counter()
        for piece, at_ms in rec.pieces:
            wait = start + at_ms / 1000 / self.speed - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            yield piece

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        return "".join(
            [
                p
                async for p in self.stream(
                    messages, system=system, temperature=temperature, max_tokens=max_tokens
                )
            ]
        )

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> T:
        self.calls.append({"kind": "structured", "messages": list(messages), "system": system})
        rec = self._find(messages)
        if self.replay_analysis_delay:
            await asyncio.sleep(rec.analysis_ms / 1000 / self.speed)
        try:
            return schema.model_validate_json(rec.analysis)
        except ValidationError as exc:
            # Only the first structured call (analyze) is recorded; a rewrite has another schema.
            raise NotRecordedError(f"recorded analysis does not fit {schema.__name__}") from exc


class RecordingLLM:
    """Wraps a real client and records the first analyze reply and first writer stream per
    question. Everything is passed through unchanged. Run one request at a time."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self._analysis: dict[str, tuple[str, float]] = {}
        self._pieces: dict[str, list[tuple[str, float]]] = {}

    def __getattr__(self, name: str):
        return getattr(self._inner, name)  # e.g. `usage` of AnthropicLLM

    async def complete(self, messages, **kw) -> str:
        return await self._inner.complete(messages, **kw)

    async def structured(self, messages, schema, **kw):
        question = question_of(messages)
        t0 = time.perf_counter()
        result = await self._inner.structured(messages, schema, **kw)
        if question is not None and question not in self._analysis:
            self._analysis[question] = (result.model_dump_json(), (time.perf_counter() - t0) * 1000)
        return result

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
        question = question_of(messages)
        record = question is not None and question not in self._pieces
        pieces: list[tuple[str, float]] = []
        t0 = time.perf_counter()
        try:
            async for piece in self._inner.stream(messages, **kw):
                if record:
                    pieces.append((piece, (time.perf_counter() - t0) * 1000))
                yield piece
        finally:
            if record and pieces and question is not None:
                self._pieces[question] = pieces

    def recording_for(self, rec_id: str, question: str) -> Recording | None:
        """The finished recording of `question`, or None if either call is missing."""
        if question not in self._analysis or question not in self._pieces:
            return None
        analysis, analysis_ms = self._analysis[question]
        return Recording(
            id=rec_id,
            question=question,
            analysis=analysis,
            analysis_ms=round(analysis_ms, 1),
            pieces=[(p, round(t, 1)) for p, t in self._pieces[question]],
        )

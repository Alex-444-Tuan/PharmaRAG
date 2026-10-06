"""Provider-agnostic LLM interface (spec section 3, decision 4) and a scripted fake.

Contract file: owned by the orchestrator. Provider implementations live in
other modules under pharmarag/llm/ and must satisfy LLMClient.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Literal, Protocol, TypedDict, TypeVar

from pydantic import BaseModel


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    model_id: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]: ...

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> T: ...


Responder = str | Callable[[Sequence[Message], str | None], str]


class FakeLLM:
    """Deterministic LLM for tests and latency replay.

    Responses are consumed in order, one per call. A responder may be a string or a
    callable (messages, system) -> str. `structured` parses the response as JSON into
    the schema. `token_delay_s` sleeps between streamed tokens (whitespace-split), which
    lets the latency eval replay recorded timings without a network.
    """

    def __init__(
        self,
        responses: Sequence[Responder] = (),
        *,
        model_id: str = "fake-llm",
        token_delay_s: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self._responses = list(responses)
        self.token_delay_s = token_delay_s
        self.calls: list[dict] = []

    def _next(self, messages: Sequence[Message], system: str | None, kind: str) -> str:
        self.calls.append({"kind": kind, "messages": list(messages), "system": system})
        if not self._responses:
            raise AssertionError(f"FakeLLM ran out of scripted responses on call {len(self.calls)}")
        r = self._responses.pop(0)
        return r(messages, system) if callable(r) else r

    async def complete(self, messages, *, system=None, temperature=0.0, max_tokens=500) -> str:
        return self._next(messages, system, "complete")

    async def stream(self, messages, *, system=None, temperature=0.0, max_tokens=500):
        pieces = self._next(messages, system, "stream").split(" ")
        for i, piece in enumerate(pieces):
            if self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            yield piece if i == len(pieces) - 1 else piece + " "

    async def structured(
        self, messages, schema: type[T], *, system=None, temperature=0.0, max_tokens=500
    ) -> T:
        return schema.model_validate(json.loads(self._next(messages, system, "structured")))

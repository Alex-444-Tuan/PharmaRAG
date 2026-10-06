"""Anthropic implementation of LLMClient (spec section 3, decision 4).

Structured output uses the API's `output_config.format` JSON-schema mode, so the reply is
schema-constrained by the server. We never regex free text.
"""

import json
import logging
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from pharmarag.llm.base import Message
from pharmarag.llm.errors import (
    TRUNCATION_MARKER,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Model families that reject a `temperature` value (API returns 400): Opus 4.7+, Opus 5.x,
# Sonnet 5.x, Fable, Mythos. The 1.x SDK also dropped the `temperature` argument from
# `messages.create`, so on the families that still accept it we send it through `extra_body`.
_NO_SAMPLING = re.compile(r"claude-(opus-4-[7-9]|opus-[5-9]|sonnet-[5-9]|fable|mythos)")
# Sonnet 5.5 thinks by default and bills thinking tokens against `max_tokens`. With a 500 token
# cap that would eat the answer, so we pick its lowest thinking mode.
_NO_THINKING = re.compile(r"claude-sonnet-5-5")


def _supports_sampling(model_id: str) -> bool:
    return _NO_SAMPLING.match(model_id) is None


@contextmanager
def _map_errors() -> Iterator[None]:
    """Transient failures (connection, 429, 5xx) become LLMUnavailableError. 4xx is a bug and
    propagates unchanged."""
    try:
        yield
    except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
        raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        raise


class AnthropicLLM:
    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("LLM_MODEL must be set to a pinned model ID")
        self.model_id = model_id
        # max_retries=0: the graph's node RetryPolicy (3 attempts, backoff) is the one retry
        # layer. The SDK default of 2 retries underneath it would make 3 x 3 = 9 attempts, about
        # 4.5 minutes of worst-case waiting per node, instead of 3.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key or None, timeout=timeout_s, max_retries=0
        )
        self.sampling_supported = _supports_sampling(model_id)

    def _kwargs(
        self,
        messages: Sequence[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [dict(m) for m in messages],
        }
        if system:
            kw["system"] = system
        if self.sampling_supported:
            kw["extra_body"] = {"temperature": temperature}
        if _NO_THINKING.match(self.model_id):
            kw["thinking"] = {"type": "between_tools"}
        return kw

    def _check_stop(self, stop_reason: str | None, max_tokens: int) -> bool:
        """Raise on refusal. Return True (and log) when the output was cut at max_tokens."""
        if stop_reason == "refusal":
            raise LLMRefusalError("model refused the request")
        if stop_reason == "max_tokens":
            logger.warning(
                "LLM output truncated at max_tokens",
                extra={"model": self.model_id, "max_tokens": max_tokens},
            )
            return True
        return False

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        kw = self._kwargs(messages, system, temperature, max_tokens)
        with _map_errors():
            response = await self._client.messages.create(**kw)
        truncated = self._check_stop(response.stop_reason, max_tokens)
        text = "".join(b.text for b in response.content if b.type == "text")
        return text + TRUNCATION_MARKER if truncated else text

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        kw = self._kwargs(messages, system, temperature, max_tokens)
        with _map_errors():
            async with self._client.messages.stream(**kw) as stream:
                async for piece in stream.text_stream:
                    yield piece
                final = await stream.get_final_message()
        if self._check_stop(final.stop_reason, max_tokens):
            yield TRUNCATION_MARKER

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> T:
        kw = self._kwargs(messages, system, temperature, max_tokens)
        kw["output_config"] = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        }
        with _map_errors():
            response = await self._client.messages.create(**kw)
        if self._check_stop(response.stop_reason, max_tokens):
            raise LLMOutputError("structured output truncated at max_tokens")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return schema.model_validate(json.loads(text))
        except (ValueError, ValidationError) as exc:
            raise LLMOutputError(f"structured output did not match {schema.__name__}") from exc

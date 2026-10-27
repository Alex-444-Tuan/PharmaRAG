"""Google Gemini provider (Interactions API, google-genai SDK), satisfying LLMClient.

Mirrors AnthropicLLM's contract: no SDK-level retries (the graph's node RetryPolicy is the one
retry layer), truncation travels in-band as TRUNCATION_MARKER, and transient faults map to
LLMUnavailableError. Gemini 3.6+ ignores temperature, so it is not sent (same position as
Claude 5.x: default sampling, pinned model ID). Thinking is set to "minimal" so it does not eat
the writer's small max_tokens budget or add latency.

Errors: the Interactions API raises google.genai._gaos.lib.compat_errors (APIStatusError with
.status_code, APIConnectionError/APITimeoutError), not google.genai.errors.
"""

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from pharmarag.llm.base import Message
from pharmarag.llm.errors import (
    TRUNCATION_MARKER,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
)

T = TypeVar("T", bound=BaseModel)

# "minimal" is rejected by gemini-3.8-flash (HTTP 400, verified live); "low" is the lowest it takes.
THINKING_LEVEL = "low"


@contextmanager
def _map_errors() -> Iterator[None]:
    from google.genai import errors as genai_errors
    from google.genai._gaos.lib import compat_errors

    try:
        yield
    except (httpx.TimeoutException, httpx.TransportError, compat_errors.APIConnectionError) as exc:
        raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
    except compat_errors.APIStatusError as exc:
        if exc.status_code == 429 or (exc.status_code or 0) >= 500:
            raise LLMUnavailableError(f"gemini {exc.status_code}") from exc
        raise
    except genai_errors.ServerError as exc:
        raise LLMUnavailableError(f"gemini {exc.code}: {exc.status}") from exc
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise LLMUnavailableError(f"gemini 429: {exc.status}") from exc
        raise


def _as_input(messages: Sequence[Message]) -> str:
    """The app sends single-turn prompts; a multi-message history is flattened with role tags."""
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


class GeminiLLM:
    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        client: Any = None,
    ) -> None:
        if not model_id:
            raise ValueError("LLM_MODEL must be set to a pinned model ID")
        self.model_id = model_id
        self._timeout_s = timeout_s
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key or None)
        self._aio = client.aio
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def _count(self, interaction: Any) -> None:
        self.usage["calls"] += 1
        usage = getattr(interaction, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += getattr(usage, "total_input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(usage, "total_output_tokens", 0) or 0

    def _kwargs(self, messages, system, max_tokens) -> dict:
        kw: dict = {
            "model": self.model_id,
            "input": _as_input(messages),
            "generation_config": {
                "max_output_tokens": max_tokens,
                "thinking_level": THINKING_LEVEL,
            },
            "timeout": self._timeout_s,
        }
        if system:
            kw["system_instruction"] = system
        return kw

    @staticmethod
    def _check_status(interaction: Any) -> bool:
        """Return True when the output was cut at max tokens; raise on failure."""
        status = getattr(interaction, "status", None)
        if status == "incomplete":
            return True
        if status in ("failed", "cancelled"):
            errors = getattr(interaction, "errors", None) or []
            text = json.dumps(errors, default=str).lower()
            if "safety" in text or "blocked" in text:
                raise LLMRefusalError("gemini safety filter blocked the request")
            raise LLMOutputError(f"gemini interaction {status}: {text[:200]}")
        return False

    async def complete(self, messages, *, system=None, temperature=0.0, max_tokens=500) -> str:
        with _map_errors():
            interaction = await self._aio.interactions.create(
                **self._kwargs(messages, system, max_tokens)
            )
        self._count(interaction)
        truncated = self._check_status(interaction)
        text = interaction.output_text or ""
        return text + TRUNCATION_MARKER if truncated else text

    async def stream(
        self, messages, *, system=None, temperature=0.0, max_tokens=500
    ) -> AsyncIterator[str]:
        truncated = False
        with _map_errors():
            events = await self._aio.interactions.create(
                stream=True, **self._kwargs(messages, system, max_tokens)
            )
            async for event in events:
                kind = getattr(event, "event_type", None)
                if kind == "step.delta":
                    delta = event.delta
                    if getattr(delta, "type", None) == "text" and getattr(delta, "text", None):
                        yield delta.text
                elif kind == "interaction.status_update":
                    if getattr(event, "status", None) == "incomplete":
                        truncated = True
                elif kind == "interaction.completed":
                    interaction = getattr(event, "interaction", None)
                    if interaction is not None:
                        self._count(interaction)
                        truncated = self._check_status(interaction) or truncated
                elif kind == "error":
                    err = getattr(event, "error", None)
                    code = getattr(err, "code", None)
                    msg = f"gemini stream error {code}: {getattr(err, 'message', '')}"
                    if code in (429, 500, 502, 503, 504):
                        raise LLMUnavailableError(msg)
                    raise LLMOutputError(msg)
        if truncated:
            yield TRUNCATION_MARKER

    async def structured(
        self, messages, schema: type[T], *, system=None, temperature=0.0, max_tokens=500
    ) -> T:
        kw = self._kwargs(messages, system, max_tokens)
        kw["response_format"] = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema.model_json_schema(),
        }
        with _map_errors():
            interaction = await self._aio.interactions.create(**kw)
        self._count(interaction)
        if self._check_status(interaction):
            raise LLMOutputError("structured output truncated at max_tokens")
        try:
            return schema.model_validate_json(interaction.output_text or "")
        except ValidationError as exc:
            raise LLMOutputError(f"structured output did not match {schema.__name__}") from exc

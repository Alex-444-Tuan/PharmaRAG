"""GeminiLLM against a fake google-genai client (offline)."""

from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

from pharmarag.llm.errors import (
    TRUNCATION_MARKER,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
)
from pharmarag.llm.factory import get_llm
from pharmarag.llm.gemini import GeminiLLM

MSG = [{"role": "user", "content": "hi"}]


def _interaction(text="ok", status="completed", errors=None):
    return NS(
        output_text=text,
        status=status,
        errors=errors,
        usage=NS(total_input_tokens=7, total_output_tokens=3),
    )


class FakeInteractions:
    def __init__(self, result=None, events=None, exc=None):
        self.result, self.events, self.exc, self.calls = result, events, exc, []

    async def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        if kw.get("stream"):

            async def gen():
                for e in self.events:
                    yield e

            return gen()
        return self.result


def _llm(**kw):
    inter = FakeInteractions(**kw)
    return GeminiLLM("gemini-3.8-flash", client=NS(aio=NS(interactions=inter))), inter


async def test_complete_sends_no_temperature_and_minimal_thinking():
    llm, inter = _llm(result=_interaction("hello"))
    assert await llm.complete(MSG, system="sys", max_tokens=123) == "hello"
    kw = inter.calls[0]
    assert kw["model"] == "gemini-3.8-flash" and kw["input"] == "hi"
    assert kw["system_instruction"] == "sys"
    assert kw["generation_config"] == {"max_output_tokens": 123, "thinking_level": "minimal"}
    assert "temperature" not in str(kw)
    assert llm.usage == {"calls": 1, "input_tokens": 7, "output_tokens": 3}


async def test_incomplete_status_appends_truncation_marker():
    llm, _ = _llm(result=_interaction("cut", status="incomplete"))
    assert (await llm.complete(MSG)).endswith(TRUNCATION_MARKER)


async def test_safety_failure_is_a_refusal():
    llm, _ = _llm(result=_interaction("", status="failed", errors=[{"message": "SAFETY block"}]))
    with pytest.raises(LLMRefusalError):
        await llm.complete(MSG)


async def test_structured_parses_json_and_sends_schema():
    class Out(BaseModel):
        x: int

    llm, inter = _llm(result=_interaction('{"x": 4}'))
    assert (await llm.structured(MSG, Out)).x == 4
    assert inter.calls[0]["response_format"]["schema"]["properties"]["x"]["type"] == "integer"

    llm, _ = _llm(result=_interaction('{"x": "nope"}'))
    with pytest.raises(LLMOutputError):
        await llm.structured(MSG, Out)


async def test_stream_yields_text_and_marks_truncation():
    events = [
        NS(event_type="interaction.created"),
        NS(event_type="step.delta", delta=NS(type="text", text="A [c1]\n")),
        NS(event_type="step.delta", delta=NS(type="thought", text="hidden")),
        NS(event_type="step.delta", delta=NS(type="text", text="B")),
        NS(event_type="interaction.completed", interaction=_interaction("A B", "incomplete")),
    ]
    llm, _ = _llm(events=events)
    out = [t async for t in llm.stream(MSG)]
    assert out == ["A [c1]\n", "B", TRUNCATION_MARKER]


async def test_stream_error_event_503_is_unavailable():
    events = [NS(event_type="error", error=NS(code=503, message="overloaded"))]
    llm, _ = _llm(events=events)
    with pytest.raises(LLMUnavailableError):
        [t async for t in llm.stream(MSG)]


async def test_server_error_maps_to_unavailable():
    from google.genai import errors

    llm, _ = _llm(exc=errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}}))
    with pytest.raises(LLMUnavailableError):
        await llm.complete(MSG)


def test_factory_selects_gemini(settings):
    s = settings.model_copy(
        update={"llm_provider": "gemini", "llm_model": "gemini-3.8-flash", "llm_api_key": "test"}
    )
    assert isinstance(get_llm(s), GeminiLLM)

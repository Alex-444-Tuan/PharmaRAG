"""AnthropicLLM against a stub SDK client. No network."""

import json
import logging
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from pharmarag.app.settings import Settings
from pharmarag.llm.anthropic import AnthropicLLM
from pharmarag.llm.base import FakeLLM
from pharmarag.llm.errors import (
    TRUNCATION_MARKER,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
)
from pharmarag.llm.factory import get_llm

MSGS = [{"role": "user", "content": "hi"}]


def _response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


class StubMessages:
    def __init__(
        self,
        response=None,
        error: Exception | None = None,
        pieces=(),
        stream_stop: str = "end_turn",
    ):
        self.calls: list[dict] = []
        self._response = response
        self._error = error
        self._pieces = pieces
        self._stream_stop = stream_stop

    async def create(self, **kw):
        self.calls.append(kw)
        if self._error:
            raise self._error
        return self._response

    def stream(self, **kw):
        self.calls.append(kw)
        pieces, error, stop = self._pieces, self._error, self._stream_stop

        class _Ctx:
            async def __aenter__(self_inner):
                async def gen():
                    for p in pieces:
                        yield p
                    if error:
                        raise error

                async def final():
                    return SimpleNamespace(stop_reason=stop)

                return SimpleNamespace(text_stream=gen(), get_final_message=final)

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()


def _llm(model: str, **stub_kw) -> tuple[AnthropicLLM, StubMessages]:
    stub = StubMessages(**stub_kw)
    return AnthropicLLM(model, client=SimpleNamespace(messages=stub)), stub


class Out(BaseModel):
    intent: str
    n: int


async def test_complete_joins_text_blocks_and_sends_model_and_system():
    llm, stub = _llm("claude-sonnet-5-5", response=_response("hello"))
    out = await llm.complete(MSGS, system="sys", max_tokens=77)
    assert out == "hello"
    kw = stub.calls[0]
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["system"] == "sys"
    assert kw["max_tokens"] == 77
    assert kw["messages"] == MSGS


async def test_sonnet_5_5_gets_no_temperature_and_lowest_thinking():
    llm, stub = _llm("claude-sonnet-5-5", response=_response("x"))
    await llm.complete(MSGS, temperature=0.0)
    kw = stub.calls[0]
    assert "temperature" not in kw
    assert "extra_body" not in kw
    assert kw["thinking"] == {"type": "between_tools"}
    assert llm.sampling_supported is False


async def test_older_model_gets_temperature_via_extra_body():
    llm, stub = _llm("claude-haiku-4-5", response=_response("x"))
    await llm.complete(MSGS, temperature=0.0)
    kw = stub.calls[0]
    assert kw["extra_body"] == {"temperature": 0.0}
    assert "thinking" not in kw


async def test_structured_uses_json_schema_output_config_not_free_text_parsing():
    llm, stub = _llm("claude-sonnet-5-5", response=_response(json.dumps({"intent": "a", "n": 2})))
    out = await llm.structured(MSGS, Out)
    assert out == Out(intent="a", n=2)
    fmt = stub.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"intent", "n"}
    assert "tools" not in stub.calls[0]


async def test_structured_invalid_json_raises_output_error():
    llm, _ = _llm("claude-sonnet-5-5", response=_response("not json"))
    with pytest.raises(LLMOutputError):
        await llm.structured(MSGS, Out)


async def test_structured_truncated_raises_output_error():
    llm, _ = _llm("claude-sonnet-5-5", response=_response('{"intent": "a"', "max_tokens"))
    with pytest.raises(LLMOutputError):
        await llm.structured(MSGS, Out)


async def test_refusal_raises_refusal_error_in_complete_structured_and_stream():
    llm, _ = _llm("claude-sonnet-5-5", response=_response("", "refusal"))
    with pytest.raises(LLMRefusalError):
        await llm.complete(MSGS)
    with pytest.raises(LLMRefusalError):
        await llm.structured(MSGS, Out)
    llm, _ = _llm("claude-sonnet-5-5", pieces=["a"], stream_stop="refusal")
    with pytest.raises(LLMRefusalError):
        [p async for p in llm.stream(MSGS)]
    assert issubclass(LLMRefusalError, LLMOutputError)


async def test_complete_marks_truncation_and_logs_warning(caplog):
    llm, _ = _llm("claude-sonnet-5-5", response=_response("A. [c1]\nB is cut", "max_tokens"))
    with caplog.at_level(logging.WARNING, logger="pharmarag.llm.anthropic"):
        out = await llm.complete(MSGS, max_tokens=12)
    assert out == "A. [c1]\nB is cut" + TRUNCATION_MARKER
    assert "truncated at max_tokens" in caplog.text
    llm, _ = _llm("claude-sonnet-5-5", response=_response("fine", "end_turn"))
    assert await llm.complete(MSGS) == "fine"


async def test_stream_marks_truncation():
    llm, _ = _llm("claude-sonnet-5-5", pieces=["a ", "b"], stream_stop="max_tokens")
    assert "".join([p async for p in llm.stream(MSGS)]) == "a b" + TRUNCATION_MARKER


async def test_stream_maps_5xx_429_and_connection_errors_to_unavailable():
    req = httpx2.Request("POST", "https://x")
    errors = [
        anthropic.InternalServerError("x", response=httpx2.Response(503, request=req), body=None),
        anthropic.RateLimitError("x", response=httpx2.Response(429, request=req), body=None),
        anthropic.APIConnectionError(request=req),
    ]
    for err in errors:
        llm, _ = _llm("claude-sonnet-5-5", pieces=["a "], error=err)
        with pytest.raises(LLMUnavailableError):
            [p async for p in llm.stream(MSGS)]


async def test_stream_4xx_is_not_mapped():
    req = httpx2.Request("POST", "https://x")
    err = anthropic.BadRequestError("x", response=httpx2.Response(400, request=req), body=None)
    llm, _ = _llm("claude-sonnet-5-5", error=err)
    with pytest.raises(anthropic.BadRequestError):
        [p async for p in llm.stream(MSGS)]


async def test_connection_error_maps_to_unavailable():
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    llm, _ = _llm("claude-sonnet-5-5", error=err)
    with pytest.raises(LLMUnavailableError):
        await llm.complete(MSGS)


async def test_5xx_maps_to_unavailable_but_4xx_does_not():
    req = httpx2.Request("POST", "https://x")
    r503 = httpx2.Response(503, request=req)
    llm, _ = _llm(
        "claude-sonnet-5-5", error=anthropic.InternalServerError("x", response=r503, body=None)
    )
    with pytest.raises(LLMUnavailableError):
        await llm.complete(MSGS)
    r400 = httpx2.Response(400, request=req)
    llm, _ = _llm(
        "claude-sonnet-5-5", error=anthropic.BadRequestError("x", response=r400, body=None)
    )
    with pytest.raises(anthropic.BadRequestError):
        await llm.complete(MSGS)


async def test_stream_yields_text_pieces():
    llm, stub = _llm("claude-sonnet-5-5", pieces=["a ", "b"])
    assert [p async for p in llm.stream(MSGS)] == ["a ", "b"]
    assert stub.calls[0]["thinking"] == {"type": "between_tools"}


def test_empty_model_id_rejected():
    with pytest.raises(ValueError):
        AnthropicLLM("", client=object())


def test_factory_fake_and_anthropic():
    fake = get_llm(Settings(_env_file=None, llm_provider="fake"))
    assert isinstance(fake, FakeLLM)
    real = get_llm(
        Settings(
            _env_file=None, llm_provider="anthropic", llm_model="claude-sonnet-5-5", llm_api_key="k"
        )
    )
    assert isinstance(real, AnthropicLLM)
    assert real.model_id == "claude-sonnet-5-5"


@pytest.mark.parametrize("model", ["", "claude-sonnet-latest"])
def test_factory_rejects_unpinned_model(model):
    with pytest.raises(ValueError):
        get_llm(Settings(_env_file=None, llm_provider="anthropic", llm_model=model))

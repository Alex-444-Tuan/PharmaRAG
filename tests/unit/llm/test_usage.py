"""AnthropicLLM keeps running token totals, which the latency recording reports."""

from types import SimpleNamespace

from pydantic import BaseModel

from pharmarag.llm.anthropic import AnthropicLLM
from tests.unit.llm.test_anthropic_llm import MSGS, StubMessages


class Out(BaseModel):
    answer: str


def usage(i: int, o: int):
    return SimpleNamespace(input_tokens=i, output_tokens=o)


async def test_usage_sums_input_and_output_tokens_over_complete_structured_and_stream():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text='{"answer": "x"}')],
        usage=usage(100, 20),
    )
    stub = StubMessages(response=response, pieces=["a", "b"])
    llm = AnthropicLLM("claude-sonnet-5-5", client=SimpleNamespace(messages=stub))

    await llm.complete(MSGS)
    await llm.structured(MSGS, Out)
    assert llm.usage == {"calls": 2, "input_tokens": 200, "output_tokens": 40}

    # A stream reports its usage on the final message.
    async def final():
        return SimpleNamespace(stop_reason="end_turn", usage=usage(300, 50))

    stub.stream = lambda **kw: _Stream(["a", "b"], final)
    assert [p async for p in llm.stream(MSGS)] == ["a", "b"]
    assert llm.usage == {"calls": 3, "input_tokens": 500, "output_tokens": 90}


async def test_a_response_without_usage_still_counts_the_call():
    response = SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text="x")]
    )
    llm = AnthropicLLM(
        "claude-sonnet-5-5", client=SimpleNamespace(messages=StubMessages(response=response))
    )
    await llm.complete(MSGS)
    assert llm.usage == {"calls": 1, "input_tokens": 0, "output_tokens": 0}


class _Stream:
    def __init__(self, pieces, final):
        self._pieces, self._final = pieces, final

    async def __aenter__(self):
        async def gen():
            for p in self._pieces:
                yield p

        return SimpleNamespace(text_stream=gen(), get_final_message=self._final)

    async def __aexit__(self, *a):
        return False

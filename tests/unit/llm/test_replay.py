"""ReplayLLM keeps the recorded text and timing. RecordingLLM captures them. No network."""

import time

import pytest
from pydantic import BaseModel

from pharmarag.llm.base import FakeLLM
from pharmarag.llm.replay import (
    NotRecordedError,
    Recording,
    RecordingLLM,
    ReplayLLM,
    load_recordings,
    question_of,
)


class Analysis(BaseModel):
    intent: str
    mentions: list[str]


def prompt(question: str) -> list[dict]:
    return [{"role": "user", "content": f"<question>\n{question}\n</question>\n\n<evidence>x"}]


def recording(question="What dose?", pieces=None, analysis_ms=0.0) -> Recording:
    return Recording(
        id="q1",
        question=question,
        analysis='{"intent": "label_qa", "mentions": ["hydroxyzine"]}',
        analysis_ms=analysis_ms,
        pieces=pieces or [("A. [c1]\n", 40.0), ("B. [c1]", 120.0), ("\nC.", 200.0)],
    )


async def arrivals(llm: ReplayLLM, question: str) -> list[tuple[str, float]]:
    t0 = time.perf_counter()
    return [(p, (time.perf_counter() - t0) * 1000) async for p in llm.stream(prompt(question))]


async def test_pieces_arrive_at_their_recorded_offsets():
    got = await arrivals(ReplayLLM([recording()]), "What dose?")
    assert [p for p, _ in got] == ["A. [c1]\n", "B. [c1]", "\nC."]
    for (_, actual), (_, wanted) in zip(got, recording().pieces, strict=True):
        assert wanted - 2 <= actual <= wanted + 150, (wanted, actual)  # never early, not far behind


async def test_scheduling_error_does_not_accumulate_over_many_tokens():
    # 40 tokens, 10 ms apart. Sleeping 10 ms after each one would drift by the sum of all the
    # overshoots; offsets measured from the start must not.
    pieces = [(f"t{i} ", 10.0 * (i + 1)) for i in range(40)]
    got = await arrivals(ReplayLLM([recording(pieces=pieces)]), "What dose?")
    assert got[-1][1] < 400 + 150


async def test_speed_scales_the_recorded_gaps():
    rec = recording(pieces=[("a", 100.0), ("b", 300.0)])
    got = await arrivals(ReplayLLM([rec], speed=4.0), "What dose?")
    assert got[0][1] < 120 and 70 <= got[1][1] < 250


async def test_complete_returns_the_recorded_text_after_the_recorded_time():
    llm = ReplayLLM([recording()])
    t0 = time.perf_counter()
    text = await llm.complete(prompt("What dose?"))
    assert text == "A. [c1]\nB. [c1]\nC." and time.perf_counter() - t0 >= 0.19


async def test_structured_returns_the_recorded_analysis_after_its_latency():
    llm = ReplayLLM([recording(analysis_ms=120.0)])
    t0 = time.perf_counter()
    out = await llm.structured(prompt("What dose?"), Analysis)
    assert out == Analysis(intent="label_qa", mentions=["hydroxyzine"])
    assert time.perf_counter() - t0 >= 0.115
    instant = ReplayLLM([recording(analysis_ms=500.0)], replay_analysis_delay=False)
    t0 = time.perf_counter()
    await instant.structured(prompt("What dose?"), Analysis)
    assert time.perf_counter() - t0 < 0.1


async def test_recordings_are_found_by_question_so_call_order_does_not_matter():
    llm = ReplayLLM(
        [recording("Q one?", [("one", 1.0)]), recording("Q two?", [("two", 1.0)])],
        replay_analysis_delay=False,
    )
    assert await llm.complete(prompt("Q two?")) == "two"
    assert await llm.complete(prompt("Q one?")) == "one"
    assert await llm.complete(prompt("Q two?")) == "two"  # a recording can be replayed forever


async def test_an_unrecorded_question_or_schema_raises_instead_of_inventing_text():
    llm = ReplayLLM([recording()])
    with pytest.raises(NotRecordedError):
        await llm.complete(prompt("Something else?"))

    class Rewrite(BaseModel):
        search_query: str

    with pytest.raises(NotRecordedError):
        await llm.structured(prompt("What dose?"), Rewrite)


def test_question_is_read_from_the_prompt_block():
    assert question_of(prompt("What dose?\nOf both?")) == "What dose?\nOf both?"
    assert question_of([{"role": "user", "content": "no block"}]) is None


def test_a_recording_survives_json_round_trip(tmp_path):
    rec = recording()
    path = tmp_path / "r.jsonl"
    path.write_text("# comment\n" + rec.to_json() + "\n\n")
    assert load_recordings(path) == [rec]


async def test_recording_llm_captures_analysis_and_pieces_with_arrival_times():
    inner = FakeLLM(
        ['{"intent": "label_qa", "mentions": ["hydroxyzine"]}', "A. [c1]\nB. [c1]"],
        token_delay_s=0.03,
    )
    llm = RecordingLLM(inner)
    msgs = prompt("What dose?")
    result = await llm.structured(msgs, Analysis)
    text = "".join([p async for p in llm.stream(msgs)])
    assert result.intent == "label_qa" and text == "A. [c1]\nB. [c1]"

    rec = llm.recording_for("q1", "What dose?")
    assert rec is not None and rec.text == text and rec.analysis_ms >= 0
    times = [t for _, t in rec.pieces]
    assert times == sorted(times) and times[0] >= 25  # first token after the first delay
    assert llm.recording_for("q1", "other") is None  # nothing recorded for it


async def test_a_recorded_run_replays_to_the_same_text():
    inner = FakeLLM(['{"intent": "label_qa", "mentions": []}', "A. [c1]\nB. [c1]"])
    rec_llm = RecordingLLM(inner)
    msgs = prompt("What dose?")
    await rec_llm.structured(msgs, Analysis)
    original = "".join([p async for p in rec_llm.stream(msgs)])
    rec = rec_llm.recording_for("q1", "What dose?")
    assert rec is not None
    replayed = "".join(
        [p async for p in ReplayLLM([rec], replay_analysis_delay=False).stream(msgs)]
    )
    assert replayed == original

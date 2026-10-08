"""The latency eval end to end on the real graph: recorded LLM traffic replayed through the ASGI
app in three VERIFY_MODEs, with the fake verifier. PostgreSQL, no network."""

import pytest

from pharmarag.app.runtime import Runtime
from pharmarag.eval import latency as lat
from pharmarag.eval.latency import SleepVerifier
from pharmarag.graph.build import build_graph
from pharmarag.llm.replay import Recording, ReplayLLM
from tests.graph.fakes import DOSE_TEXT, HYDROXYZIN_Q, FakeResolver, analysis, resolved

pytestmark = pytest.mark.pg

Q = HYDROXYZIN_Q


def token_stream(lines: list[str], step_ms: float = 25.0) -> list[tuple[str, float]]:
    """One piece per word, `step_ms` apart, newline after each line: how a real LLM streams."""
    pieces, t = [], 0.0
    for line in lines:
        words = line.split(" ")
        for k, word in enumerate(words):
            t += step_ms
            last = k == len(words) - 1
            pieces.append((word + ("\n" if last else " "), t))
    return pieces


LINES = [
    "Hydroxyzine dosage for anxiety is 50 to 100 mg four times daily. [c1]",
    "It is contraindicated in early pregnancy and in QT prolongation. [c1]",
    "Avoid it with CYP3A4 inhibitors that raise plasma levels. [c1]",
]
THREE_LINES = token_stream(LINES)
STREAM_END_MS = THREE_LINES[-1][1]


@pytest.fixture
def apps(make_deps, graph_thresholds, settings, corpus):
    rec = Recording(
        id="lq01",
        question=Q,
        analysis=analysis(mentions=["hydroxyzin"], search_query=DOSE_TEXT),
        analysis_ms=0.0,
        pieces=THREE_LINES,
    )
    llm = ReplayLLM([rec], replay_analysis_delay=False)
    verifier = SleepVerifier(per_sentence_ms=(150, 150), prepare_ms=0)
    deps = make_deps(
        llm,
        resolver=FakeResolver({"hydroxyzin": resolved("hydroxyzin", "1", "hydroxyzine")}),
        verifier=verifier,
    )
    deps.settings = deps.settings.model_copy(update={"eval_mode": True})
    base = Runtime(deps=deps, graph=build_graph(deps))
    return {m: lat.app_for(lat.runtime_for_mode(base, m), deps.settings) for m in lat.MODES}


async def test_each_mode_runs_with_its_own_verify_mode(apps):
    seen = {}
    for mode, app in apps.items():
        seen[mode] = app.state.runtime.deps.settings.verify_mode
    assert seen == {m: m for m in lat.MODES}


async def test_replayed_modes_return_the_same_answer_and_stream_beats_sequential(apps):
    samples = await lat.run_measurements(apps, [{"id": "lq01", "question": Q}], reps=2, sse=True)
    assert all(s.usable for s in samples), [s.error for s in samples if not s.usable]
    by = {(s.rep, s.mode, s.transport): s for s in samples}

    # Three sentences, no regeneration, in every mode (the verifier accepts them all).
    for s in samples:
        assert s.sentences == 3 and s.generation_attempts == 1 and s.status == "answered"

    for rep in (0, 1):
        off, seq, stream = (by[(rep, m, "json")].total_ms for m in lat.MODES)
        assert (
            off >= STREAM_END_MS - 5
        )  # the recording's last token: replay is not faster than recorded
        # Sequential pays for all three checks after the last token (3 x 150 ms). Stream only
        # for the last one, because the first two ran while the LLM was still writing.
        assert seq - off >= 3 * 150 - 60
        assert stream - off < 150 + 100
        assert seq - stream > 150

    # Over SSE the first sentence leaves the server before the LLM has finished writing.
    first = by[(1, "stream", "sse")].first_sentence_ms
    assert first is not None and first < STREAM_END_MS
    assert by[(1, "stream", "sse")].total_ms > first


async def test_statistics_over_a_replay_run(apps):
    samples = await lat.run_measurements(apps, [{"id": "lq01", "question": Q}], reps=3, sse=False)
    stats = lat.build_stats(samples, n_boot=100, seed=0)
    h = stats["headline"]
    assert h["pairs"] == 2 and stats["errors"] == []
    assert h["sequential_minus_off_p50_ms"] > h["stream_minus_off"]["p50"]["stream_minus_off_ms"]
    assert stats["modes"]["stream"]["answer_node"]["n"] == 2
    assert stats["modes"]["stream"]["regeneration"]["count"] == 0

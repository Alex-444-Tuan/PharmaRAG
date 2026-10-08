"""Latency eval: the report math, the timing client, the fake verifier. No network, no database."""

import asyncio
import json

import pytest

from pharmarag.eval import latency as lat
from pharmarag.eval.latency import (
    Sample,
    SleepVerifier,
    asgi_post,
    at3_check,
    bootstrap_ci,
    build_stats,
    diff_stat,
    percentile,
    render_markdown,
    summarize,
)
from pharmarag.graph.state import Sentence

# --- percentiles --------------------------------------------------------------------------------


def test_percentile_interpolates_like_numpy():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    assert percentile([7], 95) == 7
    assert percentile([5, 1, 3], 0) == 1 and percentile([5, 1, 3], 100) == 5
    assert percentile([10, 20], 25) == 12.5


def test_percentile_matches_numpy_on_random_data():
    import random

    import numpy as np

    rng = random.Random(3)
    xs = [rng.uniform(0, 1000) for _ in range(77)]
    for q in (5, 50, 90, 95, 99):
        assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))


def test_percentile_of_nothing_is_an_error_not_zero():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize():
    s = summarize([100.0, 200.0, 300.0, 400.0])
    assert s == {"n": 4, "p50": 250.0, "p95": 385.0, "mean": 250.0, "min": 100.0, "max": 400.0}
    assert summarize([])["n"] == 0 and summarize([])["p50"] is None


# --- AT-3 pass check ----------------------------------------------------------------------------


def test_at3_passes_inside_both_limits_and_fails_outside_either():
    off = [1000.0] * 20
    ok = at3_check(off, [1049.0] * 20)
    assert ok["pass"] and ok["p50_pass"] and ok["p95_pass"]
    assert ok["p50_ratio_minus_1"] == pytest.approx(0.049, abs=1e-4)

    assert not at3_check(off, [1051.0] * 20)["p50_pass"]  # 5.1% > 5%
    # p95 limit is 10%: 19 requests as fast as off, the slowest 20% 12% slower.
    stream = [1000.0] * 16 + [1120.0] * 4
    r = at3_check(off, stream)
    assert r["p50_pass"] and not r["p95_pass"] and not r["pass"]
    assert at3_check(off, [1099.0] * 20)["p95_pass"]
    assert not at3_check(off, [1101.0] * 20)["p95_pass"]


def test_at3_does_not_penalise_stream_being_faster():
    r = at3_check([1000.0] * 10, [700.0] * 10)
    assert r["pass"] and r["p50_ratio_minus_1"] == pytest.approx(-0.3)


# --- bootstrap ---------------------------------------------------------------------------------


def clusters_with_shift(shift: float, n_questions: int = 20, reps: int = 4):
    out = []
    for q in range(n_questions):
        base = 2000 + 37 * q
        out.append([(base + 11 * r, base + 11 * r + shift) for r in range(reps)])
    return out


def test_bootstrap_of_a_constant_difference_has_a_zero_width_interval():
    lo, hi = bootstrap_ci(clusters_with_shift(25.0), diff_stat(50, relative=False), n_boot=300)
    assert lo == pytest.approx(25.0) and hi == pytest.approx(25.0)
    lo, hi = bootstrap_ci(clusters_with_shift(0.0), diff_stat(95, relative=False), n_boot=300)
    assert (lo, hi) == (0.0, 0.0)


def test_bootstrap_interval_covers_the_point_estimate_and_widens_with_noise():
    import random

    rng = random.Random(1)
    clusters = [
        [(base := 2000 + rng.gauss(0, 300), base + 10 + rng.gauss(0, 40)) for _ in range(4)]
        for _ in range(30)
    ]
    stat = diff_stat(50, relative=False)
    a = [x for c in clusters for x, _ in c]
    b = [y for c in clusters for _, y in c]
    point = stat(a, b)
    lo, hi = bootstrap_ci(clusters, stat, n_boot=2000, seed=5)
    assert lo < point < hi and hi - lo > 1


def test_bootstrap_is_reproducible_by_seed_and_differs_across_seeds():
    clusters = [[(100.0 + q, 100.0 + q + (q * 7 % 11) * 1.5)] * 2 for q in range(15)]
    stat = diff_stat(50, relative=False)
    assert bootstrap_ci(clusters, stat, n_boot=200, seed=1) == bootstrap_ci(
        clusters, stat, n_boot=200, seed=1
    )
    assert bootstrap_ci(clusters, stat, n_boot=200, seed=1) != bootstrap_ci(
        clusters, stat, n_boot=200, seed=2
    )


def test_bootstrap_resamples_questions_not_single_pairs():
    # Two questions, each with identical reps. Resampling single pairs would give a spread of
    # many different values; resampling questions can only pick question A, B, or a mix.
    clusters = [[(100.0, 110.0)] * 4, [(100.0, 150.0)] * 4]
    lo, hi = bootstrap_ci(clusters, diff_stat(50, relative=False), n_boot=500, seed=0)
    assert lo == 10.0 and hi == 50.0


def test_relative_difference_is_a_fraction_of_off():
    stat = diff_stat(50, relative=True)
    assert stat([100.0, 200.0, 300.0], [110.0, 220.0, 330.0]) == pytest.approx(0.1)


# --- build_stats on hand-made samples ----------------------------------------------------------


def sample(qid, rep, mode, total, *, gen=1, transport="json", first=None, error=None, answer=10.0):
    return Sample(
        qid=qid,
        rep=rep,
        mode=mode,
        transport=transport,
        total_ms=total,
        first_sentence_ms=first,
        generation_attempts=gen,
        status="answered",
        node_ms={"answer": answer, "analyze": 5.0},
        error=error,
    )


def grid(off=1000.0, seq=1400.0, stream=1010.0, questions=6, reps=3):
    out = []
    for q in range(questions):
        for r in range(reps):
            jitter = q * 3 + r
            out.append(sample(f"q{q}", r, "off", off + jitter))
            out.append(sample(f"q{q}", r, "sequential", seq + jitter))
            out.append(sample(f"q{q}", r, "stream", stream + jitter))
            out.append(
                sample(f"q{q}", r, "stream", stream + 700 + jitter, transport="sse", first=400.0)
            )
    return out


def test_the_first_rep_is_discarded_and_modes_are_compared_pairwise():
    stats = build_stats(grid(), n_boot=200, seed=0)
    h = stats["headline"]
    assert h["pairs"] == 6 * 2  # rep 0 dropped
    assert stats["discarded_reps"] == 1
    assert h["off"]["n"] == h["stream"]["n"] == h["sequential"]["n"] == 12
    assert h["stream_minus_off"]["p50"]["stream_minus_off_ms"] == pytest.approx(10.0)
    assert (
        h["stream_minus_off"]["p50"]["ci95_ms"][0]
        <= 10.0
        <= h["stream_minus_off"]["p50"]["ci95_ms"][1]
    )
    assert h["sequential_minus_off_p50_ms"] == pytest.approx(400.0)
    assert h["at3"]["pass"] is True  # +1% at p50 and p95
    assert stats["errors"] == []


def test_a_slow_stream_mode_fails_at3():
    stats = build_stats(grid(stream=1200.0), n_boot=100, seed=0)
    assert stats["headline"]["at3"]["pass"] is False


def test_regenerated_requests_are_reported_separately_and_leave_the_headline():
    samples = grid()
    # Question q0, rep 1: the stream request regenerated (a second LLM call, much slower).
    for s in samples:
        if s.qid == "q0" and s.rep == 1 and s.mode == "stream" and s.transport == "json":
            s.generation_attempts, s.total_ms = 2, 9000.0
    stats = build_stats(samples, n_boot=100, seed=0)
    # The pair is dropped for all three modes, so the comparison stays like-for-like.
    assert stats["headline"]["pairs"] == 6 * 2 - 1
    assert stats["headline"]["stream"]["max"] < 2000
    m = stats["modes"]["stream"]
    assert m["regeneration"]["count"] == 1 and m["regeneration"]["rate"] == pytest.approx(
        1 / 12, abs=1e-4
    )
    assert m["regeneration"]["latency"]["p50"] == 9000.0
    assert stats["modes"]["off"]["regeneration"]["count"] == 0


def test_failed_requests_are_listed_and_excluded():
    samples = grid()
    samples[4].error = "HTTP 500"  # q0 rep 1 off? index 4 is rep 1 'sequential'
    stats = build_stats(samples, n_boot=100, seed=0)
    assert len(stats["errors"]) == 1 and stats["errors"][0]["error"] == "HTTP 500"
    assert stats["headline"]["pairs"] == 6 * 2 - 1


def test_sse_first_sentence_is_reported_but_never_enters_the_headline():
    stats = build_stats(grid(), n_boot=100, seed=0)
    assert stats["sse"]["time_to_first_sentence"]["p50"] == 400.0
    assert stats["sse"]["time_to_first_sentence"]["n"] == 12
    assert stats["headline"]["stream"]["p50"] < 1100  # the 1700 ms SSE totals are not in it


def test_per_node_timings_are_summarised_per_mode():
    m = build_stats(grid(), n_boot=100, seed=0)["modes"]["stream"]
    assert m["answer_node"]["p50"] == 10.0 and m["node_p50_ms"] == {"analyze": 5.0, "answer": 10.0}


def test_no_complete_pairs_gives_an_empty_headline_not_a_crash():
    only_off = [sample("q0", r, "off", 1000.0) for r in range(3)]
    stats = build_stats(only_off, n_boot=50, seed=0)
    assert stats["headline"] == {"pairs": 0}
    report = {
        "split": "replay_fake",
        "timestamp": "2026-01-01T00:00:00Z",
        "corpus_version": "c",
        "model_id": "m",
        "runs": 3,
        "config": {"verifier": "fake", "questions": 1, "note": "n"},
        "stats": stats,
    }
    assert "No complete pairs" in render_markdown(report)


def test_markdown_states_the_verdict_and_the_conditions():
    stats = build_stats(grid(), n_boot=100, seed=0)
    report = {
        "split": "replay_fake",
        "timestamp": "2026-01-01T00:00:00Z",
        "corpus_version": "abc123",
        "model_id": "claude-sonnet-5-5",
        "thresholds_commit": "deadbeef",
        "thresholds_dirty": False,
        "git_head": "cafe",
        "runs": 3,
        "config": {"verifier": SleepVerifier().describe(), "questions": 6, "note": "replay"},
        "stats": stats,
    }
    md = render_markdown(report)
    assert "AT-3 PASS" in md and "abc123" in md and "claude-sonnet-5-5" in md
    assert "30-80 ms" in md and "Time to first visible sentence" in md
    assert "| stream |" in md and "| sequential |" in md and "| off |" in md


# --- the fake verifier --------------------------------------------------------------------------


async def test_sleep_verifier_cost_is_in_range_and_depends_only_on_the_sentence():
    v = SleepVerifier()
    delays = {v.delay_ms(f"sentence number {i}") for i in range(200)}
    assert all(30.0 <= d <= 80.0 for d in delays) and len(delays) > 50
    assert v.delay_ms("same text") == v.delay_ms("same text")  # every mode pays the same


async def test_sleep_verifier_returns_supported_with_a_span_and_leaves_marked_sentences():
    v = SleepVerifier(per_sentence_ms=(1, 2), prepare_ms=1)
    ok = Sentence(text="A.", citations=["c2"])
    out = await v.verify([ok, Sentence(text="B.", citations=[], verdict="unsupported")], [], [])
    assert out[0].verdict == "supported" and out[0].span == "c2:0"
    assert out[1].verdict == "unsupported"


async def test_sleep_verifier_really_waits():
    v = SleepVerifier(per_sentence_ms=(60, 60), prepare_ms=40)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    prepared = await v.prepare([], [])
    t1 = loop.time()
    await v.verify_sentence(Sentence(text="A.", citations=["c1"]), prepared)
    t2 = loop.time()
    assert t1 - t0 >= 0.039 and t2 - t1 >= 0.059


# --- the timing ASGI client ---------------------------------------------------------------------


async def test_asgi_post_times_each_chunk_and_reads_the_request_body():
    seen = {}

    async def app(scope, receive, send):
        request = await receive()
        seen["body"] = json.loads(request["body"])
        seen["path"] = scope["path"]
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-a", b"b")]})
        await send(
            {"type": "http.response.body", "body": b"event: drugs\ndata: []\n\n", "more_body": True}
        )
        await asyncio.sleep(0.08)
        await send(
            {
                "type": "http.response.body",
                "body": b'event: done\ndata: {"ok": 1}\n\n',
                "more_body": False,
            }
        )

    r = await asgi_post(app, "/v1/ask", {"question": "q"})
    assert seen == {"body": {"question": "q"}, "path": "/v1/ask"}
    assert r.status == 200 and r.headers["x-a"] == "b"
    events = r.sse_events()
    assert [n for _, n, _ in events] == ["drugs", "done"] and events[1][2] == {"ok": 1}
    assert events[0][0] < 60 and 75 <= events[1][0] < 400  # the gap between chunks is visible
    assert r.last_byte_ms == events[1][0]


async def test_asgi_post_keeps_the_client_connected_until_the_response_is_complete():
    disconnected = []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})

        async def watch():
            msg = await receive()
            disconnected.append(msg["type"])

        task = asyncio.create_task(watch())
        await asyncio.sleep(0.05)  # a Starlette StreamingResponse does exactly this
        assert disconnected == []  # no premature disconnect that would cancel the response
        await send({"type": "http.response.body", "body": b"x", "more_body": False})
        await task

    r = await asgi_post(app, "/p", {})
    assert r.body == b"x" and disconnected == ["http.disconnect"]


# --- measurement loop ---------------------------------------------------------------------------


async def test_measurements_rotate_the_mode_order_and_cover_every_cell():
    order: list[tuple[str, str, bool]] = []

    def stub(mode):
        async def app(scope, receive, send):
            request = await receive()
            body = json.loads(request["body"])
            order.append((mode, body["question"], body["stream"]))
            payload = {
                "status": "answered",
                "answer": [],
                "meta": {"generation_attempts": 1},
                "debug": {"timings_ms": {"answer": 1.0}},
            }
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if body["stream"]:
                data = b"event: done\ndata: " + json.dumps(payload).encode() + b"\n\n"
            else:
                data = json.dumps(payload).encode()
            await send({"type": "http.response.body", "body": data, "more_body": False})

        return app

    apps = {m: stub(m) for m in lat.MODES}
    qs = [{"id": "a", "question": "Qa"}, {"id": "b", "question": "Qb"}]
    samples = await lat.run_measurements(apps, qs, reps=2, sse=True)

    assert len(samples) == 2 * 2 * 4  # reps x questions x (3 modes + 1 SSE)
    assert {(s.qid, s.rep, s.mode, s.transport) for s in samples} == {
        (q, r, m, "json") for q in "ab" for r in (0, 1) for m in lat.MODES
    } | {(q, r, "stream", "sse") for q in "ab" for r in (0, 1)}
    firsts = {order[i][0] for i in range(0, len(order), 4)}
    assert firsts == set(lat.MODES)  # no mode always goes first
    assert all(s.usable for s in samples)

    no_sse = await lat.run_measurements(apps, qs, reps=2, sse=False)
    assert len(no_sse) == 2 * 2 * 3


async def test_a_non_200_response_becomes_an_error_sample():
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 503, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    s = await lat.measure(app, "q", sse=False, qid="a", rep=0, mode="off")
    assert not s.usable and "503" in s.error


async def test_an_sse_stream_whose_sentences_differ_from_done_is_flagged():
    payload = {"status": "answered", "answer": [{"text": "A."}], "meta": {"generation_attempts": 1}}
    frames = [
        b'event: sentence\ndata: {"index": 0, "text": "SHOWN BUT NOT FINAL."}\n\n',
        b"event: done\ndata: " + json.dumps(payload).encode() + b"\n\n",
    ]

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for i, f in enumerate(frames):
            await send({"type": "http.response.body", "body": f, "more_body": i < len(frames) - 1})

    s = await lat.measure(app, "q", sse=True, qid="a", rep=0, mode="stream")
    assert s.error and "differ" in s.error and s.first_sentence_ms is not None


# --- inputs -------------------------------------------------------------------------------------


def test_the_question_set_is_thirty_unique_answerable_single_drug_questions():
    rows = lat.load_questions()
    assert len(rows) == 30
    assert len({r["id"] for r in rows}) == 30 and len({r["question"] for r in rows}) == 30
    for r in rows:
        assert r["drug"] in r["question"].lower()  # exactly one drug named, and it is the gold one


def test_cli_requires_a_mode_and_a_warmup_rep():
    with pytest.raises(SystemExit):
        lat.parse_args([])
    with pytest.raises(SystemExit):
        lat.parse_args(["--replay", "--reps", "1"])
    args = lat.parse_args(["--replay", "--reps", "5"])
    assert args.replay and not args.record and args.verifier == "fake" and args.reps == 5

"""Latency eval (spec section 11.3, AT-3): does in-stream verification add latency?

    uv run python -m pharmarag.eval.latency --record                 # ~30 real LLM calls, once
    uv run python -m pharmarag.eval.latency --replay --reps 5        # no LLM, no network
    uv run python -m pharmarag.eval.latency --replay --verifier real # after the verifier merged

`--record` sends each question of datasets/latency_questions.jsonl through the real graph with a
real streaming LLM call and saves the analyze reply and the writer's text with per-token arrival
times to datasets/latency_recordings.jsonl. `--replay` plays those back through the ASGI app in
the three VERIFY_MODEs (off, sequential, stream), interleaved per question, so LLM latency noise
cannot favour a mode: the token stream is the same recording in all three.

Measured: server time from request received to the last byte, through the ASGI app, and the
per-node timings the graph already records. Reported: p50 and p95 per mode, the stream minus off
difference with a bootstrap 95% interval (resampling questions, because the reps of one question
are not independent), and the pass check of AT-3 (stream p50 within 5% of off, p95 within 10%).
Requests that regenerated are reported separately and kept out of the headline. Time to first
visible sentence over SSE is reported too and is not part of AT-3.

`--verifier fake` (default) uses a verifier that sleeps 30 to 80 ms per sentence, so the
experiment can run before the NLI model exists and in CI. `--verifier real` uses the verifier
built by `build_verifier`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pharmarag.app.main import create_app
from pharmarag.app.runtime import Runtime, build_runtime
from pharmarag.app.settings import REPO_ROOT, Settings, Thresholds, get_settings, load_thresholds
from pharmarag.cache import InMemoryCache
from pharmarag.eval.report import base_meta, corpus_version, write_report
from pharmarag.graph.build import build_graph
from pharmarag.graph.run import run_ask
from pharmarag.graph.state import Chunk, Fact, Flags, Sentence
from pharmarag.grounding.base import PreparedEvidence, Verifier
from pharmarag.llm.errors import TRUNCATION_MARKER
from pharmarag.llm.factory import get_llm
from pharmarag.llm.replay import Recording, RecordingLLM, ReplayLLM, load_recordings

log = logging.getLogger("pharmarag.eval.latency")

DATASETS = Path(__file__).parent / "datasets"
QUESTIONS_FILE = DATASETS / "latency_questions.jsonl"
RECORDINGS_FILE = DATASETS / "latency_recordings.jsonl"
MODES = ("off", "sequential", "stream")

# AT-3 (spec section 2 and 11.3): stream p50 within 5% of off, and p95 within 10%.
P50_TOLERANCE = 0.05
P95_TOLERANCE = 0.10


# --- statistics ---------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100) by linear interpolation, numpy's default method."""
    if not values:
        raise ValueError("percentile of no values")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


Cluster = list[tuple[float, float]]  # one question's (off, stream) pairs, one per rep


def bootstrap_ci(
    clusters: Sequence[Cluster],
    stat: Callable[[list[float], list[float]], float],
    *,
    n_boot: int = 5000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of `stat(a, b)` over paired samples, resampling clusters.

    A cluster is all the pairs of one question. The reps of a question share its prompt and its
    recording, so they are correlated, and resampling single pairs would make the interval too
    narrow. Questions are drawn with replacement, all their pairs come along.
    """
    if not clusters:
        raise ValueError("bootstrap of no clusters")
    rng = random.Random(seed)
    k = len(clusters)
    stats: list[float] = []
    for _ in range(n_boot):
        a: list[float] = []
        b: list[float] = []
        for _ in range(k):
            for x, y in clusters[rng.randrange(k)]:
                a.append(x)
                b.append(y)
        stats.append(stat(a, b))
    stats.sort()
    tail = (1 - level) / 2 * 100
    return percentile(stats, tail), percentile(stats, 100 - tail)


def diff_stat(q: float, *, relative: bool) -> Callable[[list[float], list[float]], float]:
    """stat(off, stream) = the q-th percentile of stream minus that of off (ms, or a fraction)."""

    def stat(off: list[float], stream: list[float]) -> float:
        base, other = percentile(off, q), percentile(stream, q)
        return (other - base) / base if relative else other - base

    return stat


def at3_check(
    off: Sequence[float],
    stream: Sequence[float],
    ci_upper_relative: dict[str, float] | None = None,
) -> dict:
    """AT-3: stream p50 within 5% of off and p95 within 10% (spec section 11.3). Slower only:
    stream being faster than off is not a failure.

    The verdict is on the point estimates. `ci_upper_relative` (the upper end of the 95% interval
    of stream/off - 1, per percentile) adds the other half of the story: when that upper end is
    above the limit the data cannot rule out a failure, and the label says so. A PASS here is
    "the measured difference is inside the limit", never "the interval is inside the limit".
    """
    off50, off95 = percentile(off, 50), percentile(off, 95)
    st50, st95 = percentile(stream, 50), percentile(stream, 95)
    r50, r95 = st50 / off50 - 1, st95 / off95 - 1
    point_pass = r50 <= P50_TOLERANCE and r95 <= P95_TOLERANCE
    exceeds = {
        "p50": bool(ci_upper_relative) and ci_upper_relative["p50"] > P50_TOLERANCE,
        "p95": bool(ci_upper_relative) and ci_upper_relative["p95"] > P95_TOLERANCE,
    }
    label = "PASS (point estimate)" if point_pass else "FAIL"
    return {
        "p50_ratio_minus_1": round(r50, 4),
        "p95_ratio_minus_1": round(r95, 4),
        "p50_pass": r50 <= P50_TOLERANCE,
        "p95_pass": r95 <= P95_TOLERANCE,
        "pass": point_pass,
        "label": label,
        "ci_upper_relative": ci_upper_relative,
        "ci_upper_exceeds_limit": exceeds,
        "limits": {"p50": P50_TOLERANCE, "p95": P95_TOLERANCE},
    }


# --- an ASGI client that times every chunk -------------------------------------------------------


@dataclass
class AsgiResponse:
    status: int
    headers: dict[str, str]
    chunks: list[tuple[float, bytes]]  # (milliseconds since the request was sent, body chunk)

    @property
    def body(self) -> bytes:
        return b"".join(c for _, c in self.chunks)

    @property
    def last_byte_ms(self) -> float:
        return self.chunks[-1][0] if self.chunks else 0.0

    def json(self) -> dict:
        return json.loads(self.body)

    def sse_events(self) -> list[tuple[float, str, dict]]:
        """(arrival ms, event name, data) for each event. One event is one chunk here."""
        out = []
        for at, chunk in self.chunks:
            name, data = "", ""
            for line in chunk.decode().splitlines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    data = line[6:]
            if name:
                out.append((at, name, json.loads(data)))
        return out


async def asgi_post(app, path: str, payload: dict) -> AsgiResponse:
    """POST JSON to an ASGI app and time each body chunk. httpx's ASGI transport buffers the
    whole response, which would hide when the first SSE event left the server."""
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"latency"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("latency", 80),
    }
    sent_request = False
    finished = asyncio.Event()
    status = 0
    headers: dict[str, str] = {}
    chunks: list[tuple[float, bytes]] = []
    t0 = time.perf_counter()

    async def receive() -> dict:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        await finished.wait()  # a client that stays connected until the response is complete
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
            headers.update({k.decode().lower(): v.decode() for k, v in message["headers"]})
        elif message["type"] == "http.response.body":
            if message.get("body"):
                chunks.append(((time.perf_counter() - t0) * 1000, message["body"]))
            if not message.get("more_body", False):
                finished.set()

    await app(scope, receive, send)
    return AsgiResponse(status, headers, chunks)


# --- a verifier with a realistic cost -------------------------------------------------------------


class SleepVerifier:
    """Stands in for the NLI verifier: every sentence is `supported` after a sleep.

    `prepare` sleeps `prepare_ms` (embedding the evidence spans). `verify_sentence` sleeps
    `per_sentence_ms` (30 to 80 ms by default, the spec's budget is 150 ms p95), chosen by a hash
    of the sentence text so every mode and every rep pays the same cost for the same sentence.
    The real verifier runs its model in a worker thread, so the event loop stays free while it
    works; `asyncio.sleep` has that property too.
    """

    def __init__(
        self,
        *,
        per_sentence_ms: tuple[float, float] = (30.0, 80.0),
        prepare_ms: float = 150.0,
        batch_factor: float = 0.3,
        reject: Callable[[str], bool] | None = None,
    ) -> None:
        self.per_sentence_ms = per_sentence_ms
        self.prepare_ms = prepare_ms
        # A batch of k sentences costs the slowest one plus `batch_factor` of the others: one NLI
        # forward pass over k short pairs is cheaper than k passes, but not free. This is an
        # assumption, stated in the report, until the real verifier is measured.
        self.batch_factor = batch_factor
        self.reject = reject

    def delay_ms(self, text: str) -> float:
        lo, hi = self.per_sentence_ms
        return lo + (zlib.crc32(text.encode()) % 1000) / 999 * (hi - lo)

    async def prepare(self, evidence: list[Chunk], facts: list[Fact]) -> PreparedEvidence:
        await asyncio.sleep(self.prepare_ms / 1000)
        return object()  # type: ignore[return-value]

    async def verify_sentence(self, sentence: Sentence, prepared: PreparedEvidence) -> Sentence:
        if sentence.verdict != "unverified":
            return sentence
        await asyncio.sleep(self.delay_ms(sentence.text) / 1000)
        return self._verdict(sentence)

    def _verdict(self, sentence: Sentence) -> Sentence:
        if self.reject is not None and self.reject(sentence.text):
            return sentence.model_copy(update={"verdict": "unsupported", "score": 0.05})
        return sentence.model_copy(
            update={
                "verdict": "supported",
                "score": 0.99,
                "span": f"{sentence.citations[0]}:0" if sentence.citations else None,
            }
        )

    async def verify_many(
        self, sentences: list[Sentence], prepared: PreparedEvidence
    ) -> list[Sentence]:
        todo = [s for s in sentences if s.verdict == "unverified"]
        if todo:
            delays = [self.delay_ms(s.text) for s in todo]
            cost = max(delays) + self.batch_factor * (sum(delays) - max(delays))
            await asyncio.sleep(cost / 1000)
        return [self._verdict(s) if s.verdict == "unverified" else s for s in sentences]

    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        prepared = await self.prepare(evidence, facts)
        return [await self.verify_sentence(s, prepared) for s in sentences]

    def describe(self) -> str:
        lo, hi = self.per_sentence_ms
        return (
            f"fake: {lo:.0f}-{hi:.0f} ms per sentence, {self.prepare_ms:.0f} ms prepare, "
            f"a batch costs the slowest + {self.batch_factor:.0%} of the others"
        )


# --- the measurement ---------------------------------------------------------------------------


@dataclass
class Sample:
    qid: str
    rep: int
    mode: str
    transport: str  # "json" (the AT-3 measurement) or "sse"
    total_ms: float = 0.0
    first_sentence_ms: float | None = None  # sse only
    generation_attempts: int = 0
    status: str = ""
    node_ms: dict[str, float] = field(default_factory=dict)
    sentences: int = 0
    error: str | None = None
    group: str = "latency"  # "latency" (AT-3 set) or "regen" (grounding questions, section 11.3)

    @property
    def regenerated(self) -> bool:
        return self.generation_attempts > 1

    @property
    def usable(self) -> bool:
        return self.error is None


def runtime_for_mode(base: Runtime, mode: str) -> Runtime:
    """The same resources with VERIFY_MODE set to `mode`: a graph of its own, because a compiled
    graph binds its deps."""
    deps = dataclasses.replace(
        base.deps, settings=base.deps.settings.model_copy(update={"verify_mode": mode})
    )
    return Runtime(deps=deps, graph=build_graph(deps))


def app_for(runtime: Runtime, settings: Settings):
    app = create_app(settings)
    app.state.runtime = runtime  # the lifespan is not run: the runtime is already built
    return app


async def measure(app, question: str, *, sse: bool, qid: str, rep: int, mode: str) -> Sample:
    sample = Sample(qid=qid, rep=rep, mode=mode, transport="sse" if sse else "json")
    response = await asgi_post(app, "/v1/ask", {"question": question, "stream": sse, "debug": True})
    sample.total_ms = response.last_byte_ms
    if response.status != 200:
        sample.error = f"HTTP {response.status}: {response.body[:200]!r}"
        return sample
    if sse:
        events = response.sse_events()
        done = next((d for _, n, d in events if n == "done"), None)
        if done is None:
            sample.error = "SSE stream ended without a done event"
            return sample
        first = next((at for at, n, _ in events if n == "sentence"), None)
        sample.first_sentence_ms = first
        released = [d["text"] for _, n, d in events if n == "sentence"]
        if released != [a["text"] for a in done["answer"]]:
            sample.error = "released sentences differ from the done payload"
    else:
        done = response.json()
    sample.status = done["status"]
    sample.generation_attempts = done["meta"]["generation_attempts"]
    sample.sentences = len(done["answer"])
    sample.node_ms = done.get("debug", {}).get("timings_ms", {})
    return sample


def _rotate(items: Sequence[str], k: int) -> list[str]:
    k %= len(items)
    return [*items[k:], *items[:k]]


async def run_measurements(
    apps: dict[str, object],
    questions: Sequence[dict],
    *,
    reps: int,
    sse: bool,
    progress: Callable[[str], None] = lambda _: None,
    group: str = "latency",
) -> list[Sample]:
    """Every question, every rep, the three modes back to back in a rotating order.

    The order of the modes rotates with the question and the rep, so no mode always goes first
    (a first request pays the cold costs), and LLM or machine drift lands on all three. SSE
    requests for the `stream` mode follow each group.
    """
    samples: list[Sample] = []
    for rep in range(reps):
        for qi, q in enumerate(questions):
            for mode in _rotate(MODES, rep + qi):
                sample = await measure(
                    apps[mode], q["question"], sse=False, qid=q["id"], rep=rep, mode=mode
                )
                sample.group = group
                samples.append(sample)
            if sse:
                sample = await measure(
                    apps["stream"], q["question"], sse=True, qid=q["id"], rep=rep, mode="stream"
                )
                sample.group = group
                samples.append(sample)
        progress(f"rep {rep + 1}/{reps} done")
    return samples


# --- the report ---------------------------------------------------------------------------------


def _headline(complete: dict[tuple[str, int], dict[str, Sample]], n_boot: int, seed: int) -> dict:
    """Percentiles per mode, the stream - off difference with its interval, and the AT-3 check,
    over paired requests (one per mode) that did not regenerate."""
    headline: dict = {"pairs": len(complete), "questions": len({k[0] for k in complete})}
    if not complete:
        return headline
    off = [g["off"].total_ms for g in complete.values()]
    stream = [g["stream"].total_ms for g in complete.values()]
    headline["off"] = summarize(off)
    headline["sequential"] = summarize([g["sequential"].total_ms for g in complete.values()])
    headline["stream"] = summarize(stream)
    by_question: dict[str, Cluster] = {}
    for (qid, _), g in sorted(complete.items()):
        by_question.setdefault(qid, []).append((g["off"].total_ms, g["stream"].total_ms))
    clusters = list(by_question.values())
    diff = {}
    for name, q in (("p50", 50), ("p95", 95)):
        lo, hi = bootstrap_ci(clusters, diff_stat(q, relative=False), n_boot=n_boot, seed=seed)
        rlo, rhi = bootstrap_ci(clusters, diff_stat(q, relative=True), n_boot=n_boot, seed=seed)
        point = percentile(stream, q) - percentile(off, q)
        diff[name] = {
            "stream_minus_off_ms": round(point, 2),
            "ci95_ms": [round(lo, 2), round(hi, 2)],
            "ci95_relative": [round(rlo, 4), round(rhi, 4)],
        }
    headline["stream_minus_off"] = diff
    seq = [g["sequential"].total_ms for g in complete.values()]
    headline["sequential_minus_off_p50_ms"] = round(percentile(seq, 50) - percentile(off, 50), 2)
    headline["at3"] = at3_check(
        off,
        stream,
        ci_upper_relative={n: diff[n]["ci95_relative"][1] for n in ("p50", "p95")},
    )
    return headline


def _regen_section(samples: Sequence[Sample]) -> dict:
    """Regeneration on the grounding questions (spec section 11.3): how often a request
    regenerated, and what a regenerated request cost compared with one that did not."""
    out: dict = {}
    for mode in MODES:
        mine = [s for s in samples if s.mode == mode]
        regen = [s for s in mine if s.regenerated]
        clean = [s for s in mine if not s.regenerated]
        out[mode] = {
            "requests": len(mine),
            "regenerated": len(regen),
            "rate": round(len(regen) / len(mine), 4) if mine else None,
            "latency_regenerated": summarize([s.total_ms for s in regen]),
            "latency_not_regenerated": summarize([s.total_ms for s in clean]),
        }
    return out


def build_stats(
    samples: Sequence[Sample],
    *,
    n_boot: int,
    seed: int,
    discard_reps: int = 1,
    truncated_qids: frozenset[str] = frozenset(),
) -> dict:
    """Turn raw samples into the report body. Pure: the tests feed it hand-made samples.

    `truncated_qids` are the questions whose recorded answer hit the output token cap. The
    headline is reported twice, with and without them: a truncated answer drops its last
    sentence, so it has less verification work left at the end of the stream, which flatters the
    stream mode.
    """
    measured = [s for s in samples if s.rep >= discard_reps]
    errors = [s for s in measured if not s.usable]
    ok = [s for s in measured if s.usable]
    latency_ok = [s for s in ok if s.group == "latency"]
    json_ok = [s for s in latency_ok if s.transport == "json"]

    by_mode: dict[str, dict] = {}
    for mode in MODES:
        mine = [s for s in json_ok if s.mode == mode]
        clean = [s for s in mine if not s.regenerated]
        regen = [s for s in mine if s.regenerated]
        nodes = sorted({n for s in clean for n in s.node_ms})
        by_mode[mode] = {
            "requests": len(mine),
            "no_regeneration": summarize([s.total_ms for s in clean]),
            "answer_node": summarize([s.node_ms["answer"] for s in clean if "answer" in s.node_ms]),
            "node_p50_ms": {
                n: round(percentile([s.node_ms[n] for s in clean if n in s.node_ms], 50), 2)
                for n in nodes
            },
            "regeneration": {
                "count": len(regen),
                "rate": round(len(regen) / len(mine), 4) if mine else None,
                "latency": summarize([s.total_ms for s in regen]),
            },
        }

    # Paired, per (question, rep): only where all three modes finished without regenerating,
    # so every mode is compared on the same requests.
    groups: dict[tuple[str, int], dict[str, Sample]] = {}
    for s in json_ok:
        groups.setdefault((s.qid, s.rep), {})[s.mode] = s
    complete = {
        key: g
        for key, g in groups.items()
        if set(g) == set(MODES) and not any(x.regenerated for x in g.values())
    }
    headline = _headline(complete, n_boot, seed)
    kept = {k: g for k, g in complete.items() if k[0] not in truncated_qids}
    headline_kept = _headline(kept, n_boot, seed)
    headline_kept["excluded_truncated_questions"] = sorted(
        {k[0] for k in complete if k[0] in truncated_qids}
    )

    sse_ok = [s for s in latency_ok if s.transport == "sse" and not s.regenerated]
    regen_set = [s for s in ok if s.group == "regen" and s.transport == "json"]
    return {
        "discarded_reps": discard_reps,
        "measured_requests": len(measured),
        "errors": [dataclasses.asdict(s) | {"regenerated": s.regenerated} for s in errors],
        "modes": by_mode,
        "headline": headline,
        "headline_excluding_truncated": headline_kept,
        "regeneration_set": _regen_section(regen_set) if regen_set else None,
        "sse": {
            "note": "Time to first visible sentence in `stream` mode over SSE. Not part of AT-3.",
            "time_to_first_sentence": summarize(
                [s.first_sentence_ms for s in sse_ok if s.first_sentence_ms is not None]
            ),
            "time_to_done": summarize([s.total_ms for s in sse_ok]),
        },
    }


def _headline_markdown(title: str, h: dict) -> list[str]:
    if not h.get("pairs"):
        return [f"## {title}", "", "**No complete pairs: nothing to compare.**", ""]
    d, a = h["stream_minus_off"], h["at3"]
    out = [
        f"## {title} ({h['pairs']} paired requests, {h['questions']} questions)",
        "",
        "| mode | n | p50 ms | p95 ms | mean ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        r = h[mode]
        out.append(f"| {mode} | {r['n']} | {r['p50']} | {r['p95']} | {r['mean']} |")
    out += [
        "",
        "Stream minus off, bootstrap 95% interval (questions resampled):",
        "",
        "| statistic | difference ms | 95% CI ms | 95% CI relative |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("p50", "p95"):
        out.append(
            f"| {name} | {d[name]['stream_minus_off_ms']} | {d[name]['ci95_ms']} "
            f"| {d[name]['ci95_relative']} |"
        )
    out += [
        "",
        f"Sequential minus off at p50: {h['sequential_minus_off_p50_ms']} ms "
        "(what the verifier would cost without the in-stream design).",
        "",
        f"**AT-3 {a['label']}**: stream p50 is {a['p50_ratio_minus_1']:+.1%} of off "
        f"(limit +{P50_TOLERANCE:.0%}), stream p95 is {a['p95_ratio_minus_1']:+.1%} "
        f"(limit +{P95_TOLERANCE:.0%}).",
    ]
    for name, limit in (("p50", P50_TOLERANCE), ("p95", P95_TOLERANCE)):
        if a["ci_upper_exceeds_limit"][name]:
            out.append(
                f"- **CI upper bound exceeds the limit at {name}**: "
                f"{a['ci_upper_relative'][name]:+.1%} > +{limit:.0%}. The data cannot rule out "
                "a failure at this percentile."
            )
    out.append("")
    return out


CPUINFO = Path("/proc/cpuinfo")


def cpu_model(cpuinfo: Path = CPUINFO) -> str:
    """The CPU model name of the machine that ran the measurement.

    Latency numbers are only comparable between runs on the same kind of hardware, and the CI
    gate compares against a recorded run, so the report must say what it ran on. Linux:
    `model name` in /proc/cpuinfo (some ARM kernels print `Model` or `Hardware` instead).
    macOS: `sysctl machdep.cpu.brand_string`. Anything else: `platform.processor()`.
    """
    try:
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() in ("model name", "model", "hardware") and value.strip():
                return value.strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def runner_info(cpuinfo: Path = CPUINFO) -> dict:
    return {
        "cpu_model": cpu_model(cpuinfo),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _runner_line(report: dict) -> list[str]:
    runner = report.get("runner")  # reports written before this field existed have none
    if not runner:
        return []
    return [
        f"- runner: {runner['cpu_model']}, {runner['cpu_count']} logical CPUs "
        f"({runner['platform']})"
    ]


def render_markdown(report: dict) -> str:
    st, h = report["stats"], report["stats"]["headline"]
    out = [
        f"# Latency eval: {report['split']} ({report['timestamp']})",
        "",
        f"- corpus_version `{report['corpus_version']}`, "
        f"recorded with model `{report['model_id']}`",
        f"- thresholds commit `{report.get('thresholds_commit')}` "
        f"(dirty: {report.get('thresholds_dirty')}), git HEAD `{report.get('git_head')}`",
        f"- verifier: {report['config']['verifier']}",
        *_runner_line(report),
        f"- {report['config']['questions']} questions x {report['runs']} reps x 3 modes, "
        f"first rep discarded, {st['measured_requests']} measured requests, "
        f"{len(st['errors'])} errors",
        f"- {report['config']['note']}",
        "",
    ]
    out += _headline_markdown("Headline: all recorded questions, no regeneration", h)
    hk = st["headline_excluding_truncated"]
    cut = hk.get("excluded_truncated_questions", [])
    out += _headline_markdown(
        f"Headline without answers cut at the output token cap ({len(cut)} questions excluded)", hk
    )
    out += [
        "Answers that hit the cap lose their last sentence, so they leave less to verify at the "
        "end of the stream, which flatters the stream mode. Both headlines are reported.",
        "",
        "## Answer node and regeneration (AT-3 question set)",
        "",
        "| mode | answer node p50 ms | answer node p95 ms | requests | regenerated | "
        "regeneration rate | regenerated p50 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        m = st["modes"][mode]
        out.append(
            f"| {mode} | {m['answer_node']['p50']} | {m['answer_node']['p95']} | {m['requests']} "
            f"| {m['regeneration']['count']} | {m['regeneration']['rate']} "
            f"| {m['regeneration']['latency']['p50']} |"
        )
    rs = st.get("regeneration_set")
    if rs:
        out += [
            "",
            "## Regeneration on the grounding questions (spec section 11.3)",
            "",
            report["config"].get("regen_note", ""),
            "",
            "| mode | requests | regenerated | rate | regenerated p50 ms | "
            "not regenerated p50 ms |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for mode in MODES:
            r = rs[mode]
            out.append(
                f"| {mode} | {r['requests']} | {r['regenerated']} | {r['rate']} "
                f"| {r['latency_regenerated']['p50']} | {r['latency_not_regenerated']['p50']} |"
            )
    s = st["sse"]
    out += [
        "",
        "## Time to first visible sentence (SSE, stream mode)",
        "",
        s["note"],
        "",
        "| metric | n | p50 ms | p95 ms |",
        "| --- | --- | --- | --- |",
        f"| first `sentence` event | {s['time_to_first_sentence']['n']} "
        f"| {s['time_to_first_sentence']['p50']} | {s['time_to_first_sentence']['p95']} |",
        f"| `done` event | {s['time_to_done']['n']} | {s['time_to_done']['p50']} "
        f"| {s['time_to_done']['p95']} |",
        "",
    ]
    if st["errors"]:
        out += ["## Errors", ""]
        out += [
            f"- {e['qid']} rep {e['rep']} {e['mode']}/{e['transport']}: {e['error']}"
            for e in st["errors"][:10]
        ]
        out.append("")
    return "\n".join(out)


# --- CLI ----------------------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_FILE, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m pharmarag.eval.latency")
    p.add_argument("--record", action="store_true", help="make real LLM calls and save them")
    p.add_argument("--replay", action="store_true", help="measure the three modes on the recording")
    p.add_argument("--reps", type=int, default=5, help="repetitions per question; first discarded")
    p.add_argument("--questions", type=int, default=None, help="use only the first N questions")
    p.add_argument("--verifier", choices=["fake", "real"], default="fake")
    p.add_argument("--no-sse", action="store_true", help="skip the time-to-first-sentence runs")
    p.add_argument(
        "--no-analysis-delay",
        action="store_true",
        help="replay the analyze reply instantly instead of after its recorded latency",
    )
    p.add_argument("--force", action="store_true", help="with --record: re-record every question")
    p.add_argument(
        "--regen-questions",
        type=Path,
        default=None,
        help="JSONL of grounding questions (id, question): the regeneration rate and the latency "
        "of regenerated requests are measured on these (spec section 11.3)",
    )
    p.add_argument("--recordings", type=Path, default=RECORDINGS_FILE)
    p.add_argument("--boot", type=int, default=5000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    args = p.parse_args(argv)
    if not (args.record or args.replay):
        p.error("give --record, --replay, or both")
    if args.reps < 2:
        p.error("--reps must be at least 2: the first repetition is discarded as warm-up")
    return args


def _write_recordings(path: Path, recs: Sequence[Recording]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(r.to_json() + "\n" for r in recs))
    tmp.replace(path)


async def record(args: argparse.Namespace, settings: Settings) -> None:
    """One real run per question through the real graph, writer only (verifier off)."""
    settings = settings.model_copy(update={"eval_mode": True})
    questions = [(q, "latency") for q in load_questions(QUESTIONS_FILE, args.questions)]
    if args.regen_questions:
        questions += [(q, "regen") for q in load_questions(args.regen_questions, args.questions)]
    ids = [q["id"] for q, _ in questions]
    if len(ids) != len(set(ids)):
        sys.exit("question ids must be unique across the latency and regeneration sets")
    existing = (
        {r.id: r for r in load_recordings(args.recordings)} if args.recordings.exists() else {}
    )
    real = get_llm(settings)
    llm = RecordingLLM(real)
    runtime = await build_runtime(settings, llm=llm, verifier=None)
    conn_version = "unknown"
    try:
        from pharmarag.db import connect

        conn = await connect(settings.database_url)
        try:
            conn_version = await corpus_version(conn)
        finally:
            await conn.close()
        for q, group in questions:
            if q["id"] in existing and not args.force:
                continue
            out = await run_ask(
                runtime.graph, runtime.deps, question=q["question"], flags=Flags(verifier=False)
            )
            rec = llm.recording_for(q["id"], q["question"])
            # The AT-3 set must be answerable in one pass. The grounding questions include
            # unanswerable ones (the writer says INSUFFICIENT), which are still worth replaying;
            # what cannot be replayed is a request whose retrieval was retried (a second,
            # unrecorded structured call) or that never reached the writer.
            usable = (
                rec is not None
                and out["meta"]["retrieval_attempts"] == 1
                and out["meta"]["generation_attempts"] == 1
                and (group == "regen" or out["status"] == "answered")
            )
            print(f"{q['id']}: {out['status']} {'recorded' if usable else 'SKIPPED'}", flush=True)
            if usable:
                assert rec is not None
                existing[q["id"]] = dataclasses.replace(
                    rec,
                    group=group,
                    model_id=llm.model_id,
                    corpus_version=conn_version,
                    recorded_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            _write_recordings(args.recordings, list(existing.values()))
        print("usage:", json.dumps(getattr(real, "usage", None)), flush=True)
    finally:
        from pharmarag.app.runtime import close_runtime

        await close_runtime(runtime)


async def replay(args: argparse.Namespace, settings: Settings) -> dict:
    from pharmarag.db import connect
    from pharmarag.resolver.cascade import CascadeResolver
    from pharmarag.resolver.local import PgLocalStore

    settings = settings.model_copy(update={"eval_mode": True, "verify_mode": "stream"})
    thresholds: Thresholds = load_thresholds(settings.thresholds_path)
    questions = load_questions(QUESTIONS_FILE, args.questions)
    regen_questions = (
        load_questions(args.regen_questions, args.questions) if args.regen_questions else []
    )
    recordings = {r.id: r for r in load_recordings(args.recordings)}
    missing = [q["id"] for q in questions if q["id"] not in recordings]
    if missing:
        sys.exit(f"no recording for {missing}; run with --record first")
    # A grounding question that could not be recorded (it never reached the writer, or its
    # retrieval was retried) is left out and named in the report.
    regen_skipped = [q["id"] for q in regen_questions if q["id"] not in recordings]
    regen_questions = [q for q in regen_questions if q["id"] in recordings]
    model_id = recordings[questions[0]["id"]].model_id or "unknown"
    truncated_qids = frozenset(
        q["id"] for q in questions if TRUNCATION_MARKER in recordings[q["id"]].text
    )

    llm = ReplayLLM(
        [recordings[q["id"]] for q in [*questions, *regen_questions]],
        model_id=model_id,
        replay_analysis_delay=not args.no_analysis_delay,
    )
    verifier: Verifier | None = SleepVerifier() if args.verifier == "fake" else None
    kwargs = {"verifier": verifier} if args.verifier == "fake" else {}
    # No RxNav: the resolver runs on the local alias table and facts are skipped, so the replay
    # needs no network. The RxNorm cache is "warm" in the spec's sense: that step costs nothing.
    base = await build_runtime(settings, llm=llm, resolver=None, rx_facts=None, **kwargs)
    try:
        base.deps.resolver = CascadeResolver(
            PgLocalStore(base.deps.pool), None, thresholds.resolver, InMemoryCache()
        )
        if base.deps.verifier is None:
            sys.exit("--verifier real: this build has no verifier (build_verifier not implemented)")
        apps = {m: app_for(runtime_for_mode(base, m), settings) for m in MODES}
        t0 = time.perf_counter()
        samples = await run_measurements(
            apps,
            questions,
            reps=args.reps,
            sse=not args.no_sse,
            progress=lambda msg: print(f"{msg} ({time.perf_counter() - t0:.0f}s)", flush=True),
        )
        if regen_questions:
            samples += await run_measurements(
                apps,
                regen_questions,
                reps=args.reps,
                sse=False,
                group="regen",
                progress=lambda msg: print(
                    f"regen set: {msg} ({time.perf_counter() - t0:.0f}s)", flush=True
                ),
            )
        conn = await connect(settings.database_url)
        try:
            corpus = await corpus_version(conn)
        finally:
            await conn.close()
    finally:
        from pharmarag.app.runtime import close_runtime

        await close_runtime(base)

    report = base_meta(
        eval_name="latency",
        split=f"replay_{args.verifier}",
        mode="replay",
        runs=args.reps,
        model_id=model_id,
        corpus=corpus,
        thresholds=thresholds,
        dataset=QUESTIONS_FILE,
    )
    report["runner"] = runner_info()
    report["config"] = {
        "questions": len(questions),
        "verifier": SleepVerifier().describe() if args.verifier == "fake" else "real (NLI)",
        "recordings_sha256": __import__("hashlib").sha256(args.recordings.read_bytes()).hexdigest(),
        "analysis_latency_replayed": not args.no_analysis_delay,
        "bootstrap": {"resamples": args.boot, "seed": args.seed, "unit": "question"},
        "note": (
            "Replay: recorded analyze reply and writer tokens with recorded timing; resolver on "
            "the local alias table (no RxNav); rx_facts=None, so the RxNorm facts branch returns "
            "no facts and the writer prompt has no [r] blocks; no answer cache exists in this "
            "build."
        ),
        "rx_facts": None,
        "truncated_recordings": sorted(truncated_qids),
        "regen_questions": [q["id"] for q in regen_questions],
        "regen_questions_not_recorded": regen_skipped,
        "regen_note": (
            "Grounding questions replayed in the three modes. A regenerated request replays the "
            "first recording a second time (the second attempt's prompt differs, its text is not "
            "recorded), so its latency is the cost of a second writer call, not a measurement "
            "of the strict prompt."
            + (
                " The fake verifier accepts every sentence, so no request regenerates: "
                "the rate is 0 by construction; run with --verifier real."
                if args.verifier == "fake"
                else ""
            )
        ),
    }
    report["stats"] = build_stats(
        samples, n_boot=args.boot, seed=args.seed, truncated_qids=truncated_qids
    )
    report["samples"] = [dataclasses.asdict(s) for s in samples]
    return report


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.record:
        await record(args, settings)
    if args.replay:
        report = await replay(args, settings)
        json_path, md_path = write_report(report, args.out_dir, render_markdown(report))
        print(render_markdown(report))
        print(f"wrote {json_path} and {md_path}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()

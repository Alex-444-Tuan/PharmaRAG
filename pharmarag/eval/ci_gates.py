"""Pass/fail rules for the evals that run in CI (spec section 13, "Eval in CI").

    python -m pharmarag.eval.run lookalike --split test --mode system --resolver-only --offline --ci
    python -m pharmarag.eval.run grounding --score --ci
    python -m pharmarag.eval.run latency --ci

Each exits non-zero when its gate fails. The rules are pure functions of a report dict, so tests
feed them hand-made reports. CI has no DailyMed, RxNav or LLM: the look-alike gate runs the
resolver offline on the committed CI fixture, the grounding gate scores the frozen claims, and the
latency gate replays the committed recordings (Lane B's `pharmarag.eval.latency`, called as a
subprocess so its own asyncio loop, pool and ASGI apps are torn down with the process).

What each gate is, and is not:

* look-alike: ZERO wrong-drug resolutions on the frozen test set. That is the safety property
  (never act on another drug), not the AT-1 percentage, which needs the LLM arm.
* grounding: `summary.at2_valid` of the frozen-claims score, i.e. AT-2 met AND the AT-2b guard
  held AND the thresholds file is committed.
* latency: AT-3 on the POINT estimate (stream p50 within 5% and p95 within 10% of off). The 95%
  bootstrap interval is printed, not gated: with 10 questions it is wide, and the spec's pass
  rule is on the point estimates. Wall-clock noise on a shared CI runner can move a p95 by a few
  percent; the gate does not retry, because a retry would hide a regression half the time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# A latency headline over fewer pairs than this is not evidence of anything. 10 questions with
# 2 measured repetitions give 20 pairs; a few regenerated requests are dropped from the pairs.
MIN_LATENCY_PAIRS = 10


def lookalike_failures(report: dict, *, dataset_rows: int | None = None) -> list[str]:
    """Reasons the look-alike gate fails (empty list: pass). Reads every run, not the last."""
    if not report.get("resolver_only") or report.get("rxnav") != "offline":
        return ["the gate runs with --resolver-only --offline (no RxNav, no LLM)"]
    system = report.get("system")
    if not system or not system.get("runs"):
        return ["no system arm in the report: run with --mode system or both"]
    failures: list[str] = []
    for i, run in enumerate(system["runs"], 1):
        counts = run["counts"]
        if counts["total"] == 0 or (dataset_rows is not None and counts["total"] != dataset_rows):
            failures.append(f"run {i}: classified {counts['total']} rows, expected {dataset_rows}")
        if counts["wrong_drug"]:
            wrong = [r for r in run["rows"] if r["outcome"] == "wrong_drug"]
            failures.append(
                f"run {i}: {counts['wrong_drug']} wrong-drug resolution(s): "
                + "; ".join(
                    f"{r['mention']!r} -> {r['resolved_name']} ({r['resolved_key']}), "
                    f"gold {r['gold_ingredient_key']}"
                    for r in wrong
                )
            )
    return failures


def grounding_failures(report: dict) -> list[str]:
    s = report["summary"]
    if s.get("at2_valid") is True:
        return []
    reason = s.get("invalid_reason")
    return [
        "AT-2 is not valid: "
        + (
            reason
            or f"reduction {s.get('reduction')} (target {s.get('at2_target')}), "
            f"false-reject rate {s.get('at2b_false_reject_rate')}"
        )
    ]


def latency_failures(report: dict) -> list[str]:
    stats = report.get("stats", {})
    failures: list[str] = []
    if stats.get("errors"):
        failures.append(f"{len(stats['errors'])} replayed request(s) failed")
    headline = stats.get("headline", {})
    at3 = headline.get("at3")
    if at3 is None:
        return [*failures, "no complete off/sequential/stream pairs: nothing to compare"]
    if headline["pairs"] < MIN_LATENCY_PAIRS:
        failures.append(f"only {headline['pairs']} complete pairs (need {MIN_LATENCY_PAIRS})")
    if not at3["pass"]:
        limits = at3["limits"]
        failures.append(
            f"AT-3 failed: stream/off - 1 is {at3['p50_ratio_minus_1']:+.1%} at p50 "
            f"(limit {limits['p50']:.0%}) and {at3['p95_ratio_minus_1']:+.1%} at p95 "
            f"(limit {limits['p95']:.0%})"
        )
    return failures


def run_latency_replay(
    out_dir: Path, *, questions: int, reps: int, verifier: str, extra: Sequence[str] = ()
) -> tuple[int, dict | None]:
    """Run `python -m pharmarag.eval.latency --replay ...` and return (exit code, report).

    The report is the one JSON file the run wrote into `out_dir`, which must be empty or new.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pharmarag.eval.latency",
        "--replay",
        "--verifier",
        verifier,
        "--questions",
        str(questions),
        "--reps",
        str(reps),
        "--out-dir",
        str(out_dir),
        # The time-to-first-sentence runs are not part of AT-3 and cost a quarter of the time.
        "--no-sse",
        *extra,
    ]
    code = subprocess.run(cmd, check=False).returncode
    reports = sorted(out_dir.glob("latency_*.json"))
    if code != 0 or not reports:
        return (code or 1), None
    return 0, json.loads(reports[-1].read_text())

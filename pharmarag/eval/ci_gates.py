"""Pass/fail rules for the evals that run in CI (spec section 13).

    python -m pharmarag.eval.run lookalike --split test --mode system --resolver-only --offline --ci
    python -m pharmarag.eval.run grounding --score --ci
    python -m pharmarag.eval.run latency --ci

Each exits non-zero when its gate fails. The rules are pure functions of a report dict, so tests
feed them hand-made reports. CI has no DailyMed, RxNav or LLM: the look-alike gate runs the
resolver offline on the committed CI fixture, the grounding gate scores the frozen claims, and the
latency gate replays the committed recordings (Lane B's `pharmarag.eval.latency`, called as a
subprocess so its own asyncio loop, pool and ASGI apps are torn down with the process).

What each gate is, and is not:

* look-alike: ZERO wrong-drug resolutions on the frozen test set, AND every row exactly as in the
  committed reference run (outcome, status, resolved key). The first is the safety property
  (never act on another drug); the second stops a resolver that resolves nothing from passing
  it. Not the AT-1 percentage, which needs the LLM arm.
* grounding: `summary.at2_valid` of the frozen-claims score (AT-2 met AND the AT-2b guard held
  AND the thresholds file is committed), AND no verdict change and no best-entailment drift above
  1e-3 against the committed per-claim reference scores.
* latency: AT-3 on the POINT estimate (stream p50 within 5% and p95 within 10% of off). On a PR it
  runs with the fixed-sleep verifier: a hosted 2-vCPU runner is not the machine the real-verifier
  numbers were measured on, so a PR checks that the overlap mechanism works, deterministically.
  The real NLI verifier at the full 30 questions x 5 reps runs nightly, same limits. The 95%
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

DATASETS = Path(__file__).parent / "datasets"
LOOKALIKE_EXPECTED = DATASETS / "lookalike_test_offline_expected.jsonl"
GROUNDING_REFERENCE = DATASETS / "grounding_reference_scores.jsonl"
SCORE_DRIFT_MAX = 1e-3


def load_lookalike_expected() -> list[dict]:
    return _jsonl(LOOKALIKE_EXPECTED) if LOOKALIKE_EXPECTED.exists() else []


def load_grounding_reference() -> list[dict]:
    return _jsonl(GROUNDING_REFERENCE) if GROUNDING_REFERENCE.exists() else []


def main(argv: list[str] | None = None) -> int:
    """Freeze a reference from a report: `python -m pharmarag.eval.ci_gates freeze-lookalike
    <report.json>` or `freeze-grounding <report.json>`. Do it once, from a run you reviewed,
    and commit the file; changing it later is a deliberate, reviewed act."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in ("freeze-lookalike", "freeze-grounding"):
        print(main.__doc__)
        return 2
    report = json.loads(Path(args[1]).read_text())
    if args[0] == "freeze-lookalike":
        _write_jsonl(LOOKALIKE_EXPECTED, lookalike_reference(report))
        print(f"wrote {LOOKALIKE_EXPECTED}")
    else:
        _write_jsonl(GROUNDING_REFERENCE, grounding_reference(report))
        print(f"wrote {GROUNDING_REFERENCE}")
    return 0


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def lookalike_reference(report: dict) -> list[dict]:
    """The per-row outcome of one run of the offline look-alike gate, as frozen in
    `datasets/lookalike_test_offline_expected.jsonl`: which mention ended how, and on which key."""
    rows = report["system"]["runs"][0]["rows"]
    return [
        {
            "mention": r["mention"],
            "outcome": r["outcome"],
            "status": r["status"],
            "resolved_key": r["resolved_key"],
        }
        for r in rows
    ]


def lookalike_failures(
    report: dict, *, dataset_rows: int | None = None, expected: Sequence[dict] | None = None
) -> list[str]:
    """Reasons the look-alike gate fails (empty list: pass). Reads every run, not the last.

    Two rules. No wrong-drug resolution (the safety property). And every row must match the
    frozen reference exactly (outcome, status and, for a resolved row, the key): "no wrong drug"
    alone is also satisfied by a resolver that stops resolving anything."""
    if not report.get("resolver_only") or report.get("rxnav") != "offline":
        return ["the gate runs with --resolver-only --offline (no RxNav, no LLM)"]
    system = report.get("system")
    if not system or not system.get("runs"):
        return ["no system arm in the report: run with --mode system or both"]
    if not expected:
        return ["no frozen per-row reference to compare with (freeze-lookalike was never run)"]
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
        got = [
            {k: r[k] for k in ("mention", "outcome", "status", "resolved_key")} for r in run["rows"]
        ]
        if len(got) != len(expected):
            failures.append(f"run {i}: {len(got)} rows, reference has {len(expected)}")
            continue
        for want, have in zip(expected, got, strict=True):
            if want != have:
                failures.append(
                    f"run {i}: {want['mention']!r} expected {want['outcome']}/{want['status']}/"
                    f"{want['resolved_key']}, got {have['outcome']}/{have['status']}/"
                    f"{have['resolved_key']} (mention {have['mention']!r})"
                )
    return failures


def grounding_reference(report: dict) -> list[dict]:
    """Per-claim verdict and best entailment of the reference run, frozen in
    `datasets/grounding_reference_scores.jsonl`."""
    return [
        {
            "claim_id": r["claim_id"],
            "label": r["label"],
            "verdict": r["verdict"],
            "score": None if r["score"] is None else round(r["score"], 6),
        }
        for r in report["rows"]
    ]


def grounding_failures(report: dict, *, reference: Sequence[dict] | None = None) -> list[str]:
    """AT-2 must be valid, and the scores must not have drifted from the reference run.

    The pass/fail margin on the frozen claims is wide (accepted claims score at least 0.98,
    rejected ones at most 0.61), so a different CPU build would not flip a verdict. The drift
    check is what notices a silent numeric change (a different torch build, a moved model
    revision) before it eats into that margin: any verdict change fails, and so does any best
    entailment that moved by more than SCORE_DRIFT_MAX."""
    failures: list[str] = []
    s = report["summary"]
    if s.get("at2_valid") is not True:
        reason = s.get("invalid_reason")
        failures.append(
            "AT-2 is not valid: "
            + (
                reason
                or f"reduction {s.get('reduction')} (target {s.get('at2_target')}), "
                f"false-reject rate {s.get('at2b_false_reject_rate')}"
            )
        )
    if not reference:
        return [*failures, "no per-claim reference scores (freeze-grounding was never run)"]
    have = {r["claim_id"]: r for r in report["rows"]}
    want = {r["claim_id"]: r for r in reference}
    if set(have) != set(want):
        failures.append(
            f"claim set differs from the reference: missing {sorted(set(want) - set(have))}, "
            f"unexpected {sorted(set(have) - set(want))}"
        )
    for cid in sorted(set(have) & set(want)):
        h, w = have[cid], want[cid]
        if h["verdict"] != w["verdict"]:
            failures.append(f"{cid}: verdict {w['verdict']} -> {h['verdict']}")
        if (h["score"] is None) != (w["score"] is None):
            failures.append(f"{cid}: score {w['score']} -> {h['score']}")
        elif h["score"] is not None and abs(h["score"] - w["score"]) > SCORE_DRIFT_MAX:
            failures.append(
                f"{cid}: best entailment drifted {w['score']:.6f} -> {h['score']:.6f} "
                f"(more than {SCORE_DRIFT_MAX})"
            )
    return failures


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


if __name__ == "__main__":
    sys.exit(main())

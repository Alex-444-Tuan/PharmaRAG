"""CI gates (spec section 13): the rules, the CLI wiring and the exit codes."""

import asyncio
import json
import subprocess

import pytest

from pharmarag.eval import ci_gates
from pharmarag.eval import grounding_cli as cli
from pharmarag.eval import report as report_mod
from pharmarag.eval import run as eval_run


def run(argv):
    return asyncio.run(eval_run.main_async(eval_run.parse_args(argv)))


# --- look-alike ---

EXPECTED = ci_gates.load_lookalike_expected()


def la_report(*per_run_changes, total=None, offline=True):
    """A report whose runs equal the frozen reference, except for `changes`: a dict
    {row index: field overrides} per run. `la_report({})` is one clean run."""
    runs = []
    for changes in per_run_changes:
        rows = []
        for i, e in enumerate(EXPECTED):
            row = {
                **e,
                "resolved_name": "x",
                "gold_ingredient_key": e["resolved_key"] or "gold",
                **changes.get(i, {}),
            }
            rows.append(row)
        wrong = sum(r["outcome"] == "wrong_drug" for r in rows)
        n = len(rows) if total is None else total
        runs.append({"counts": {"wrong_drug": wrong, "total": n}, "rows": rows})
    return {
        "resolver_only": True,
        "rxnav": "offline" if offline else "live",
        "system": {"runs": runs},
    }


def gate(report):
    return ci_gates.lookalike_failures(report, dataset_rows=15, expected=EXPECTED)


WRONG = {0: {"outcome": "wrong_drug", "status": "resolved", "resolved_key": "5470"}}
DEAD = {
    i: {"outcome": "not_recognised", "status": "unknown", "resolved_key": None} for i in range(15)
}


def test_the_frozen_lookalike_reference_is_12_correct_and_3_clarified():
    assert len(EXPECTED) == 15
    outcomes = [e["outcome"] for e in EXPECTED]
    assert outcomes.count("correct") == 12 and outcomes.count("clarified") == 3
    assert all(e["resolved_key"] for e in EXPECTED if e["outcome"] == "correct")
    dataset = eval_run.lookalike.load_dataset("test")
    assert [e["mention"] for e in EXPECTED] == [r.mention for r in dataset]


def test_lookalike_passes_when_every_row_matches_the_reference():
    assert gate(la_report({})) == []


def test_lookalike_fails_when_the_resolver_stops_resolving():
    # Zero wrong-drug resolutions, and still a failure: every row is not_recognised.
    report = la_report(DEAD)
    assert report["system"]["runs"][0]["counts"]["wrong_drug"] == 0
    failures = gate(report)
    assert len(failures) == 15 and "expected correct/resolved/4815" in " ".join(failures)


def test_lookalike_fails_when_a_clarification_becomes_a_resolution_or_the_key_changes():
    assert gate(
        la_report({0: {"outcome": "correct", "status": "resolved", "resolved_key": "5553"}})
    )
    assert gate(la_report({2: {"resolved_key": "9999"}}))  # same outcome, different key


def test_lookalike_fails_on_one_wrong_drug_and_names_it():
    failures = gate(la_report(WRONG))
    assert any("1 wrong-drug" in f and "hydroxazine" in f for f in failures)


def test_lookalike_checks_every_run_not_just_the_last():
    failures = gate(la_report(WRONG, {}))
    assert failures and all(f.startswith("run 1") for f in failures)


def test_lookalike_fails_when_rows_went_missing_or_the_reference_is_absent():
    assert gate(la_report({}, total=14))
    short = la_report({})
    short["system"]["runs"][0]["rows"].pop()
    assert any("14 rows" in f for f in gate(short))
    assert ci_gates.lookalike_failures(la_report({}), dataset_rows=15, expected=None)


def test_lookalike_fails_without_a_system_arm_or_when_not_offline():
    assert gate({"resolver_only": True, "rxnav": "offline"})
    assert gate(la_report({}, offline=False))


# --- grounding ---

REFERENCE = ci_gates.load_grounding_reference()


def gr_report(valid=True, drift=None, verdicts=None, drop=()):
    rows = [dict(r) for r in REFERENCE if r["claim_id"] not in drop]
    for r in rows:
        if r["score"] is not None:  # two perturbed claims never reach the NLI model
            r["score"] += (drift or {}).get(r["claim_id"], 0.0)
        r["verdict"] = (verdicts or {}).get(r["claim_id"], r["verdict"])
    return {"summary": {"at2_valid": valid}, "rows": rows}


def test_the_frozen_grounding_reference_covers_the_40_claims():
    assert len(REFERENCE) == 40
    scores = [r["score"] for r in REFERENCE if r["score"] is not None]
    # The nearest claims to entail_min (0.7) are 0.740 (g13, a hallucination that passes) and
    # 0.614: far outside the 1e-3 the drift check allows, so a drift alarm comes long before a flip.
    assert min(abs(x - 0.7) for x in scores) > 0.03
    assert ci_gates.SCORE_DRIFT_MAX == 1e-3


def test_grounding_gate_passes_on_the_reference_itself():
    assert ci_gates.grounding_failures(gr_report(), reference=REFERENCE) == []


def test_grounding_gate_follows_at2_valid():
    invalid = {"at2_valid": False, "reduction": 0.5, "at2_target": 0.73}
    (msg,) = ci_gates.grounding_failures({**gr_report(), "summary": invalid}, reference=REFERENCE)
    assert "0.5" in msg
    dirty = {"at2_valid": False, "invalid_reason": "dirty thresholds"}
    (msg,) = ci_gates.grounding_failures({**gr_report(), "summary": dirty}, reference=REFERENCE)
    assert "dirty thresholds" in msg


def test_grounding_gate_tolerates_a_drift_up_to_1e_3_and_fails_beyond_it():
    cid = REFERENCE[0]["claim_id"]
    assert ci_gates.grounding_failures(gr_report(drift={cid: 9e-4}), reference=REFERENCE) == []
    (msg,) = ci_gates.grounding_failures(gr_report(drift={cid: -1.5e-3}), reference=REFERENCE)
    assert cid in msg and "drifted" in msg


def test_grounding_gate_fails_on_any_verdict_change_even_with_at2_valid():
    cid = REFERENCE[0]["claim_id"]
    (msg,) = ci_gates.grounding_failures(
        gr_report(verdicts={cid: "unsupported"}), reference=REFERENCE
    )
    assert f"{cid}: verdict supported -> unsupported" in msg


def test_grounding_gate_fails_on_a_missing_claim_or_reference():
    assert ci_gates.grounding_failures(gr_report(drop={"g10"}), reference=REFERENCE)
    assert ci_gates.grounding_failures(gr_report(), reference=None)


# --- latency ---


def lat_report(r50=0.01, r95=0.02, pairs=20, errors=()):
    at3 = {
        "p50_ratio_minus_1": r50,
        "p95_ratio_minus_1": r95,
        "pass": r50 <= 0.05 and r95 <= 0.10,
        "limits": {"p50": 0.05, "p95": 0.10},
    }
    return {"stats": {"errors": list(errors), "headline": {"pairs": pairs, "at3": at3}}}


def test_latency_passes_within_limits():
    assert ci_gates.latency_failures(lat_report()) == []


def test_latency_fails_at_each_percentile():
    (m50,) = ci_gates.latency_failures(lat_report(r50=0.06))
    assert "+6.0% at p50" in m50
    (m95,) = ci_gates.latency_failures(lat_report(r95=0.11))
    assert "+11.0% at p95" in m95


def test_latency_fails_with_no_pairs_few_pairs_or_errors():
    assert ci_gates.latency_failures({"stats": {"headline": {"pairs": 0}}})
    assert ci_gates.latency_failures(lat_report(pairs=ci_gates.MIN_LATENCY_PAIRS - 1))
    assert ci_gates.latency_failures(lat_report(errors=[{"qid": "lq01"}]))


def test_run_latency_replay_calls_the_latency_cli_and_reads_its_report(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, check):
        seen["cmd"] = cmd
        out = cmd[cmd.index("--out-dir") + 1]
        (tmp_path / "o").mkdir(exist_ok=True)
        (tmp_path / "o" / "latency_replay_real_x.json").write_text(json.dumps(lat_report()))
        assert out == str(tmp_path / "o")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ci_gates.subprocess, "run", fake_run)
    code, report = ci_gates.run_latency_replay(
        tmp_path / "o", questions=10, reps=3, verifier="real"
    )
    cmd = seen["cmd"]
    assert cmd[1:4] == ["-m", "pharmarag.eval.latency", "--replay"]
    for flag, value in (("--verifier", "real"), ("--questions", "10"), ("--reps", "3")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--no-sse" in cmd
    assert code == 0 and report == lat_report()


def test_run_latency_replay_reports_a_crash_and_a_missing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ci_gates.subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 3)
    )
    assert ci_gates.run_latency_replay(tmp_path, questions=1, reps=2, verifier="fake") == (3, None)
    monkeypatch.setattr(
        ci_gates.subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 0)
    )
    code, report = ci_gates.run_latency_replay(tmp_path, questions=1, reps=2, verifier="fake")
    assert code == 1 and report is None


# --- CLI wiring and exit codes ---


def test_lookalike_ci_requires_the_offline_resolver_only_test_split():
    ok = ["lookalike", "--split", "test", "--resolver-only", "--offline", "--ci"]
    assert eval_run.parse_args(ok).ci is True
    for bad in (
        ["lookalike", "--split", "test", "--resolver-only", "--ci"],
        ["lookalike", "--split", "test", "--ci"],
        ["lookalike", "--split", "dev", "--resolver-only", "--offline", "--ci"],
        [*ok, "--mode", "baseline"],
    ):
        with pytest.raises(SystemExit):
            eval_run.parse_args(bad)


def test_lookalike_ci_exit_code_follows_the_gate(tmp_path, monkeypatch, capsys):
    argv = ["lookalike", "--split", "test", "--mode", "system", "--resolver-only", "--offline"]
    argv += ["--ci", "--out-dir", str(tmp_path)]
    current = {}

    async def fake_resolver_only(args, settings):
        return current["report"]

    monkeypatch.setattr(eval_run, "_resolver_only", fake_resolver_only)
    monkeypatch.setattr(eval_run, "get_settings", lambda: None)
    monkeypatch.setattr(eval_run, "write_report", lambda r, d, md: ("j", "m"))
    monkeypatch.setattr(eval_run.lookalike_run, "render_markdown", lambda r: "")
    current["report"] = la_report({})
    assert run(argv) == 0
    current["report"] = la_report(WRONG)
    assert run(argv) == 1
    assert "CI GATE FAILED (lookalike)" in capsys.readouterr().out
    current["report"] = la_report(DEAD)  # no wrong drug, but nothing resolves
    assert run(argv) == 1


def test_without_ci_a_wrong_drug_does_not_change_the_exit_code(tmp_path, monkeypatch):
    async def fake_resolver_only(args, settings):
        return la_report(WRONG)

    monkeypatch.setattr(eval_run, "_resolver_only", fake_resolver_only)
    monkeypatch.setattr(eval_run, "get_settings", lambda: None)
    monkeypatch.setattr(eval_run, "write_report", lambda r, d, md: ("j", "m"))
    monkeypatch.setattr(eval_run.lookalike_run, "render_markdown", lambda r: "")
    argv = ["lookalike", "--split", "test", "--mode", "system", "--resolver-only", "--offline"]
    assert run(argv) == 0


def test_latency_ci_subcommand_defaults_and_exit_codes(tmp_path, monkeypatch, capsys):
    args = eval_run.parse_args(["latency", "--ci", "--out-dir", str(tmp_path)])
    assert (args.questions, args.reps, args.verifier) == (10, 3, "fake")  # the PR gate
    with pytest.raises(SystemExit):
        eval_run.parse_args(["latency"])  # --ci is required: it is only the gate

    calls = []

    def fake(out_dir, **kw):
        calls.append(kw)
        return 0, current["report"]

    current = {"report": lat_report()}
    monkeypatch.setattr(eval_run.ci_gates, "run_latency_replay", fake)
    assert run(["latency", "--ci", "--out-dir", str(tmp_path)]) == 0
    assert calls == [{"questions": 10, "reps": 3, "verifier": "fake"}]
    nightly = ["latency", "--ci", "--verifier", "real", "--questions", "30", "--reps", "5"]
    assert run([*nightly, "--out-dir", str(tmp_path)]) == 0
    assert calls[-1] == {"questions": 30, "reps": 5, "verifier": "real"}
    current["report"] = lat_report(r50=0.2)
    assert run(["latency", "--ci", "--out-dir", str(tmp_path)]) == 1
    assert "CI GATE FAILED (latency)" in capsys.readouterr().out
    current["report"] = None
    assert run(["latency", "--ci", "--out-dir", str(tmp_path)]) == 1


def test_grounding_ci_flag_needs_score(capsys):
    assert run(["grounding", "--worksheet", "--ci"]) == 2
    assert "--score only" in capsys.readouterr().out


def test_grounding_score_ci_exit_code(monkeypatch, tmp_path):
    """`--score --ci` returns 1 for an invalid AT-2 and 0 for a valid one; without --ci it is 0."""
    from pharmarag.eval import grounding_score

    clean = {"thresholds_commit": "abc", "thresholds_dirty": False, "git_head": "def"}
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: clean)
    monkeypatch.setattr(cli, "_load_verifier", lambda *a, **k: object())

    class Conn:
        async def close(self):
            return None

    async def fake_connect(url):
        return Conn()

    monkeypatch.setattr(cli, "connect", fake_connect)
    monkeypatch.setattr(cli, "write_report", lambda r, d, md: ("j", "m"))
    current = {}

    async def fake_score(**kw):
        return current["report"], "md"

    monkeypatch.setattr(grounding_score, "run_score", fake_score)
    argv = ["grounding", "--score", "--out-dir", str(tmp_path)]
    current["report"] = {
        **gr_report(),
        "summary": {"at2_valid": False, "reduction": 0.4, "at2_target": 0.73},
    }
    assert run(argv) == 0
    assert run([*argv, "--ci"]) == 1
    current["report"] = gr_report()
    assert run([*argv, "--ci"]) == 0
    cid = REFERENCE[0]["claim_id"]
    current["report"] = gr_report(drift={cid: 0.01})  # AT-2 still valid, scores moved
    assert run([*argv, "--ci"]) == 1

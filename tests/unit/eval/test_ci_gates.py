"""CI gates (spec section 13, "Eval in CI"): the rules, the CLI wiring and the exit codes."""

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


def la_report(*per_run_wrong, total=15, offline=True):
    runs = []
    for wrong in per_run_wrong:
        rows = [
            {
                "mention": f"m{i}",
                "outcome": "wrong_drug",
                "resolved_name": "partner",
                "resolved_key": "1",
                "gold_ingredient_key": "2",
            }
            for i in range(wrong)
        ]
        runs.append({"counts": {"wrong_drug": wrong, "total": total}, "rows": rows})
    return {
        "resolver_only": True,
        "rxnav": "offline" if offline else "live",
        "system": {"runs": runs},
    }


def test_lookalike_passes_with_zero_wrong_drug_resolutions():
    assert ci_gates.lookalike_failures(la_report(0), dataset_rows=15) == []


def test_lookalike_fails_on_one_wrong_drug_and_names_it():
    (msg,) = ci_gates.lookalike_failures(la_report(1), dataset_rows=15)
    assert "1 wrong-drug" in msg and "'m0'" in msg and "partner" in msg


def test_lookalike_checks_every_run_not_just_the_last():
    failures = ci_gates.lookalike_failures(la_report(1, 0), dataset_rows=15)
    assert len(failures) == 1 and failures[0].startswith("run 1")


def test_lookalike_fails_when_rows_went_missing():
    assert ci_gates.lookalike_failures(la_report(0, total=14), dataset_rows=15)
    assert ci_gates.lookalike_failures(la_report(0, total=0), dataset_rows=None)


def test_lookalike_fails_without_a_system_arm_or_when_not_offline():
    assert ci_gates.lookalike_failures({"resolver_only": True, "rxnav": "offline"})
    assert ci_gates.lookalike_failures(la_report(0, offline=False))


# --- grounding ---


def test_grounding_gate_follows_at2_valid():
    assert ci_gates.grounding_failures({"summary": {"at2_valid": True}}) == []
    (msg,) = ci_gates.grounding_failures(
        {"summary": {"at2_valid": False, "reduction": 0.5, "at2_target": 0.73}}
    )
    assert "0.5" in msg
    (msg,) = ci_gates.grounding_failures(
        {"summary": {"at2_valid": False, "invalid_reason": "dirty thresholds"}}
    )
    assert "dirty thresholds" in msg


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
    current["report"] = la_report(0)
    assert run(argv) == 0
    current["report"] = la_report(2)
    assert run(argv) == 1
    assert "CI GATE FAILED (lookalike)" in capsys.readouterr().out


def test_without_ci_a_wrong_drug_does_not_change_the_exit_code(tmp_path, monkeypatch):
    async def fake_resolver_only(args, settings):
        return la_report(3)

    monkeypatch.setattr(eval_run, "_resolver_only", fake_resolver_only)
    monkeypatch.setattr(eval_run, "get_settings", lambda: None)
    monkeypatch.setattr(eval_run, "write_report", lambda r, d, md: ("j", "m"))
    monkeypatch.setattr(eval_run.lookalike_run, "render_markdown", lambda r: "")
    argv = ["lookalike", "--split", "test", "--mode", "system", "--resolver-only", "--offline"]
    assert run(argv) == 0


def test_latency_ci_subcommand_defaults_and_exit_codes(tmp_path, monkeypatch, capsys):
    args = eval_run.parse_args(["latency", "--ci", "--out-dir", str(tmp_path)])
    assert (args.questions, args.reps, args.verifier) == (10, 3, "real")
    with pytest.raises(SystemExit):
        eval_run.parse_args(["latency"])  # --ci is required: it is only the gate

    calls = []

    def fake(out_dir, **kw):
        calls.append(kw)
        return 0, current["report"]

    current = {"report": lat_report()}
    monkeypatch.setattr(eval_run.ci_gates, "run_latency_replay", fake)
    assert run(["latency", "--ci", "--out-dir", str(tmp_path)]) == 0
    assert calls == [{"questions": 10, "reps": 3, "verifier": "real"}]
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
    current["report"] = {"summary": {"at2_valid": False, "reduction": 0.4, "at2_target": 0.73}}
    assert run(argv) == 0
    assert run([*argv, "--ci"]) == 1
    current["report"] = {"summary": {"at2_valid": True}}
    assert run([*argv, "--ci"]) == 0

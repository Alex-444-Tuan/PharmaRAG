"""Command line wiring for `python -m pharmarag.eval.run grounding ...`."""

import pytest

from pharmarag.eval import grounding_cli as cli
from pharmarag.eval import grounding_data as gd
from pharmarag.eval import grounding_labels as gl
from pharmarag.eval import run as eval_run
from tests.unit.eval.test_grounding_labels import CLAIMS, fill


@pytest.mark.parametrize("action", cli.ACTIONS)
def test_each_action_flag_selects_itself(action):
    args = eval_run.parse_args(["grounding", f"--{action}"])
    assert args.eval == "grounding" and args.action == action


def test_exactly_one_action_is_required(capsys):
    with pytest.raises(SystemExit):
        eval_run.parse_args(["grounding"])
    with pytest.raises(SystemExit):
        eval_run.parse_args(["grounding", "--score", "--generate"])


def test_options_parse():
    a = eval_run.parse_args(
        [
            "grounding",
            "--latency",
            "--nli-model",
            "x/y",
            "--context",
            "cascade",
            "--heading",
            "sub",
            "--candidates",
            "3",
            "--repeats",
            "5",
            "--torch-threads",
            "4",
        ]
    )
    assert (a.nli_model, a.context, a.heading, a.candidates, a.repeats, a.torch_threads) == (
        "x/y",
        "cascade",
        "sub",
        3,
        5,
        4,
    )
    with pytest.raises(SystemExit):
        eval_run.parse_args(["grounding", "--tune", "--context", "nonsense"])


def test_the_lookalike_command_still_parses_and_validates():
    a = eval_run.parse_args(["lookalike", "--split", "dev"])
    assert a.eval == "lookalike" and a.runs == 1
    with pytest.raises(SystemExit):
        eval_run.parse_args(["lookalike", "--split", "dev", "--runs", "0"])


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every dataset path at tmp_path and write the three test claims there."""
    for name in ("TEST_CLAIMS", "TEST_LABELS", "TEST_TOPUP"):
        monkeypatch.setattr(gd, name, tmp_path / f"{name}.jsonl")
    gd.write_jsonl(gd.TEST_CLAIMS, CLAIMS)
    return tmp_path


def run(argv):
    import asyncio

    args = eval_run.parse_args(argv)
    return asyncio.run(eval_run.main_async(args))


def test_worksheet_import_topup_flow(sandbox, capsys):
    out = ["--out-dir", str(sandbox)]
    assert run(["grounding", "--worksheet", *out]) == 0
    ws = sandbox / "grounding_label_worksheet.md"
    assert "LABEL:" in ws.read_text() and not gd.TEST_LABELS.exists()

    # importing the blank worksheet fails and writes nothing
    assert run(["grounding", "--import", *out]) == 2
    assert not gd.TEST_LABELS.exists()
    assert "LABEL is empty" in capsys.readouterr().out

    # the human fills it in
    ws.write_text(fill(ws.read_text(), {"g01": "S", "g02": "S", "g03": "H"}))
    assert run(["grounding", "--import", *out]) == 0
    assert gl.read_labels(gd.TEST_LABELS) == {"g01": "S", "g02": "S", "g03": "H"}
    assert "Fewer than 10 are H" in capsys.readouterr().out


def test_import_removes_a_stale_topup(sandbox):
    out = ["--out-dir", str(sandbox)]
    run(["grounding", "--worksheet", *out])
    ws = sandbox / "grounding_label_worksheet.md"
    ws.write_text(fill(ws.read_text(), {"g01": "S", "g02": "S", "g03": "S"}))
    gd.TEST_TOPUP.write_text("stale\n")
    assert run(["grounding", "--import", *out]) == 0
    assert not gd.TEST_TOPUP.exists()


def test_generate_refuses_to_overwrite_the_frozen_claims(sandbox, capsys):
    before = gd.TEST_CLAIMS.read_text()
    assert run(["grounding", "--generate", "--out-dir", str(sandbox)]) == 2
    assert "frozen" in capsys.readouterr().out
    assert gd.TEST_CLAIMS.read_text() == before

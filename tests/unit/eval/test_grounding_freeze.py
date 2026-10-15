"""Test-set scoring can never pick its own settings (spec review log round 5.9, section 11):
`--score` refuses every tuning override and a dirty thresholds file, and a result made with a
dirty file is not valid."""

import asyncio

import pytest

from pharmarag.eval import grounding_cli as cli
from pharmarag.eval import grounding_score as gs
from pharmarag.eval import report as report_mod
from pharmarag.eval import run as eval_run
from pharmarag.models.embedder import FakeEmbedder
from tests.unit.eval.test_grounding_score import rows
from tests.unit.grounding.test_verifier import TH, FakeNLI, verifier

CLEAN = {"thresholds_commit": "abc", "thresholds_dirty": False, "git_head": "def"}
DIRTY = {"thresholds_commit": "abc", "thresholds_dirty": True, "git_head": "def"}


def run(argv):
    return asyncio.run(eval_run.main_async(eval_run.parse_args(argv)))


@pytest.fixture
def no_models(monkeypatch):
    """Loading the verifier or opening the database is a failure: refusal must come first."""

    def boom(*a, **k):
        raise AssertionError("a model was loaded or a connection opened before the refusal")

    monkeypatch.setattr(cli, "_load_verifier", boom)
    monkeypatch.setattr(cli, "connect", boom)
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: CLEAN)


@pytest.mark.parametrize(
    "option",
    [
        ["--context", "neighbour"],
        ["--context", "cascade"],
        ["--heading", "none"],
        ["--heading", "full"],
        ["--candidates", "5"],
        ["--nli-model", "cross-encoder/nli-deberta-v3-xsmall"],
        ["--torch-threads", "4"],
    ],
)
def test_score_refuses_each_override(option, no_models, capsys):
    assert run(["grounding", "--score", *option]) == 2
    out = capsys.readouterr().out
    assert "refuses" in out and option[0] in out


def test_score_names_every_override_it_was_given(no_models, capsys):
    assert run(["grounding", "--score", "--context", "span", "--candidates", "9"]) == 2
    out = capsys.readouterr().out
    assert "--context" in out and "--candidates" in out


def test_overrides_are_still_allowed_for_tuning_and_latency():
    a = eval_run.parse_args(["grounding", "--tune", "--context", "span", "--candidates", "5"])
    assert cli.score_overrides(a) == ["--context", "--candidates"]  # parsed, just not for --score
    assert cli.score_overrides(eval_run.parse_args(["grounding", "--score"])) == []


def test_score_refuses_a_dirty_thresholds_file_before_loading_anything(
    no_models, monkeypatch, capsys
):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: DIRTY)
    assert run(["grounding", "--score"]) == 2
    assert "dirty" in capsys.readouterr().out


# --- the same rules inside run_score, for any other caller ---


def frozen(**kw):
    return verifier(FakeNLI(), th=TH, **kw)


def test_check_passes_for_the_frozen_verifier(thresholds, monkeypatch):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: CLEAN)
    from pharmarag.grounding.verifier import GroundingVerifier

    v = GroundingVerifier(FakeEmbedder(), FakeNLI(), thresholds.grounding)
    gs.check_scoring_is_frozen(v, thresholds)  # no exception


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"context": "span"}, "premise mode"),
        ({"heading": "full"}, "premise mode"),
    ],
)
def test_check_refuses_a_different_premise_mode(thresholds, monkeypatch, change, match):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: CLEAN)
    from pharmarag.grounding.verifier import GroundingVerifier

    v = GroundingVerifier(FakeEmbedder(), FakeNLI(), thresholds.grounding, **change)
    with pytest.raises(gs.ScoreRefused, match=match):
        gs.check_scoring_is_frozen(v, thresholds)


def test_check_refuses_different_thresholds_or_candidate_count(thresholds, monkeypatch):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: CLEAN)
    from pharmarag.grounding.verifier import GroundingVerifier

    g = thresholds.grounding
    for update in ({"candidate_spans": g.candidate_spans + 2}, {"entail_min": 0.5}):
        v = GroundingVerifier(FakeEmbedder(), FakeNLI(), g.model_copy(update=update))
        with pytest.raises(gs.ScoreRefused, match="differ"):
            gs.check_scoring_is_frozen(v, thresholds)


def test_check_refuses_a_dirty_thresholds_file(thresholds, monkeypatch):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: DIRTY)
    from pharmarag.grounding.verifier import GroundingVerifier

    v = GroundingVerifier(FakeEmbedder(), FakeNLI(), thresholds.grounding)
    with pytest.raises(gs.ScoreRefused, match="uncommitted"):
        gs.check_scoring_is_frozen(v, thresholds)


async def test_run_score_refuses_before_reading_any_claim(thresholds, monkeypatch):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: CLEAN)
    from pharmarag.grounding.verifier import GroundingVerifier

    v = GroundingVerifier(FakeEmbedder(), FakeNLI(), thresholds.grounding, context="neighbour")
    with pytest.raises(gs.ScoreRefused):
        await gs.run_score(verifier=v, thresholds=thresholds, conn=None, llm_model_id=None)


# --- a result made with a dirty file is not valid ---


def make_report(thresholds, tmp_path, monkeypatch, provenance, scored):
    monkeypatch.setattr(report_mod, "thresholds_provenance", lambda *a, **k: provenance)
    ds = tmp_path / "c.jsonl"
    ds.write_text("{}\n")
    return gs.build_report(
        split="test", rows=scored, thresholds=thresholds, verifier=verifier(FakeNLI()),
        llm_model_id=None, corpus="c", claims_path=ds, extra_datasets={}, note="n",
    )  # fmt: skip


GOOD = rows(n_h_pass=1, n_h_reject=11, n_s_pass=29, n_s_reject=0)  # 92% reduction, 0% false reject


def test_a_good_result_is_valid_with_a_clean_file(thresholds, tmp_path, monkeypatch):
    r = make_report(thresholds, tmp_path, monkeypatch, CLEAN, GOOD)
    assert r["summary"]["at2_met"] and r["summary"]["at2b_pass"]
    assert r["summary"]["at2_valid"] is True


def test_the_same_result_is_not_valid_with_a_dirty_file(thresholds, tmp_path, monkeypatch):
    r = make_report(thresholds, tmp_path, monkeypatch, DIRTY, GOOD)
    assert r["thresholds_dirty"] is True
    assert r["summary"]["at2_met"] and r["summary"]["at2b_pass"]  # the numbers are what they are
    assert r["summary"]["at2_valid"] is False
    assert "uncommitted" in r["summary"]["invalid_reason"]
    md = gs.render_markdown(r)
    assert "NOT VALID" in md and "uncommitted" in md


def test_unknown_dirtiness_does_not_by_itself_void_the_result(thresholds, tmp_path, monkeypatch):
    unknown = {"thresholds_commit": None, "thresholds_dirty": None, "git_head": None}
    r = make_report(thresholds, tmp_path, monkeypatch, unknown, GOOD)
    assert r["summary"]["at2_valid"] is True  # no git: reported as null, not as dirty
    assert r["thresholds_dirty"] is None

"""Threshold tuning on dev and the per-sentence latency harness (spec sections 9 and 11.2)."""

import pytest

from pharmarag.eval import grounding_bench as gb
from pharmarag.eval import grounding_tune as gt
from pharmarag.eval.grounding_data import Claim
from pharmarag.eval.grounding_score import LabelledClaim
from pharmarag.grounding.verifier import Analysis, GroundingVerifier
from pharmarag.models.embedder import FakeEmbedder
from tests.unit.grounding.test_verifier import TH, FakeNLI, make_chunk


def sweep_row(e, fr, red, recall):
    return {"entail_min": e, "false_reject_rate": fr, "reduction": red, "recall": recall}


# --- proposing thresholds ---


def test_current_value_is_kept_when_nothing_beats_it_by_more_than_a_claim():
    sweep = [sweep_row(e, 0.0, 0.95, 1.0) for e in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert gt.propose_entail(sweep, current=0.70) == 0.7


def test_value_moves_when_another_is_clearly_better():
    sweep = [
        sweep_row(0.5, 0.0, 0.60, 1.0),
        sweep_row(0.7, 0.0, 0.62, 1.0),
        sweep_row(0.9, 0.0, 0.95, 1.0),
    ]
    assert gt.propose_entail(sweep, current=0.70) == 0.9


def test_values_over_the_false_reject_budget_are_excluded():
    sweep = [
        sweep_row(0.5, 0.01, 0.80, 0.99),
        sweep_row(0.9, 0.30, 1.00, 0.70),  # better reduction, but rejects 30% of supported claims
    ]
    assert gt.propose_entail(sweep, current=0.7) == 0.5


def test_when_every_value_is_over_budget_all_are_considered():
    sweep = [sweep_row(0.5, 0.2, 0.9, 0.8), sweep_row(0.9, 0.3, 1.0, 0.7)]
    assert gt.propose_entail(sweep, current=0.7) in (0.5, 0.9)


def test_ties_go_to_the_higher_value():
    sweep = [sweep_row(0.6, 0.0, 0.9, 1.0), sweep_row(0.8, 0.0, 0.9, 1.0)]
    assert gt.propose_entail(sweep, current=0.7) == 0.8


def test_contradict_min_is_the_value_nearest_current_that_is_precise_enough():
    sweep = [
        {"contradict_min": 0.6, "contradicted": 10, "precision": 0.80},
        {"contradict_min": 0.8, "contradicted": 8, "precision": 0.90},
        {"contradict_min": 0.9, "contradicted": 6, "precision": 0.97},
        {"contradict_min": 0.99, "contradicted": 2, "precision": 1.0},
    ]
    assert gt.propose_contradict(sweep, current=0.8) == 0.9


def test_contradict_min_keeps_current_when_it_is_already_precise():
    sweep = [
        {"contradict_min": 0.7, "contradicted": 10, "precision": 1.0},
        {"contradict_min": 0.8, "contradicted": 9, "precision": 1.0},
    ]
    assert gt.propose_contradict(sweep, current=0.8) == 0.8


# --- sweeps over saved analyses ---


def labelled(label, cid):
    c = Claim(
        id=cid,
        sentence="s",
        citations=["c1"],
        evidence=[make_chunk("c1", ["Take 10 mg."])],
        label=label,
    )
    return LabelledClaim(c, label, "synthetic")


def analysis(entail, contra):
    return Analysis(["c1:0"], [], entail, "c1:0", contra, "c1:0")


def test_entail_sweep_moves_with_the_threshold():
    lc = [labelled("S", "a"), labelled("S", "b"), labelled("H", "c"), labelled("H", "d")]
    an = [analysis(0.95, 0), analysis(0.60, 0), analysis(0.55, 0), analysis(0.02, 0)]
    by = {r["entail_min"]: r for r in gt.sweep_entail(lc, an, TH)}
    assert by[0.5]["false_reject"].startswith("0/2")
    assert by[0.5]["reduction"] == pytest.approx(0.5)  # d is caught, c (0.55) slips through
    assert by[0.65]["false_reject"].startswith("1/2")
    assert by[0.65]["reduction"] == pytest.approx(1.0)


def test_contradict_sweep_reports_precision_of_the_contradicted_verdict():
    lc = [labelled("H", "a"), labelled("H", "b"), labelled("S", "c")]
    an = [analysis(0.1, 0.95), analysis(0.1, 0.85), analysis(0.1, 0.82)]  # c is a wrong S reject
    by = {r["contradict_min"]: r for r in gt.sweep_contradict(lc, an, TH)}
    assert by[0.8]["contradicted"] == 3 and by[0.8]["contradicted_on_supported"] == 1
    assert by[0.9]["contradicted"] == 1 and by[0.9]["precision"] == 1.0
    assert by[0.99]["contradicted"] == 0 and by[0.99]["precision"] is None


# --- end to end with fake models ---


def make_verifier(rule) -> GroundingVerifier:
    return GroundingVerifier(FakeEmbedder(), FakeNLI(rule), TH.model_copy(), context="neighbour")


async def test_run_tune_reports_every_mode_and_checks_the_cascade(thresholds):
    claims = [
        Claim(
            id="a",
            sentence="Take 10 mg once daily.",
            citations=["c1"],
            label="S",
            evidence=[make_chunk("c1", ["Take 10 mg once daily.", "Avoid alcohol."])],
        ),
        Claim(
            id="b",
            sentence="Take 40 mg once daily.",
            citations=["c1"],
            label="H",
            evidence=[make_chunk("c1", ["Take 10 mg once daily.", "Avoid alcohol."])],
        ),
        Claim(
            id="c",
            sentence="Drink alcohol freely.",
            citations=["c1"],
            label="H",
            evidence=[make_chunk("c1", ["Take 10 mg once daily.", "Avoid alcohol."])],
        ),
    ]

    def rule(p, h):
        return (0.95, 0.0) if h.startswith("Take 10") else (0.05, 0.9)

    v = make_verifier(rule)
    report = await gt.run_tune(v, claims, thresholds)
    assert set(report["modes"]) == {"neighbour", "span", "both"}
    assert report["claims"] == {"S": 1, "H": 2}
    both = report["modes"]["both"]["proposed_summary"]
    assert both["reduction"] == 1.0 and both["at2b_false_reject_rate"] == 0.0
    assert report["cascade_check"]["same_verdict"] == 3
    assert v.context == "neighbour"  # the verifier is left as it was found
    md = gt.render_markdown(report)
    assert "Premise: both" in md and "Cascade check" in md and "synthetic" in md


async def test_run_tune_ignores_unlabelled_claims(thresholds):
    c = Claim(id="a", sentence="Take 10 mg.", citations=["c1"], evidence=[make_chunk("c1", ["x."])])
    report = await gt.run_tune(
        make_verifier(lambda p, h: (0.9, 0.0)), [c], thresholds, modes=("span",)
    )
    assert report["claims"] == {"S": 0, "H": 0}


# --- latency harness ---


def test_percentiles():
    p = gb.percentiles([float(i) for i in range(1, 101)])
    assert p["n"] == 100
    assert p["p50_ms"] == pytest.approx(50.5, abs=0.1)
    assert p["p95_ms"] == pytest.approx(95.05, abs=0.1)
    assert p["max_ms"] == 100.0


async def test_time_verifier_measures_each_sentence_after_a_warmup_pass():
    nli = FakeNLI(lambda p, h: (0.95, 0.0), delay_s=0.01)
    v = GroundingVerifier(FakeEmbedder(), nli, TH)
    claims = [
        Claim(
            id=f"c{i}",
            sentence="Take 10 mg once daily.",
            citations=["c1"],
            evidence=[make_chunk("c1", ["Take 10 mg once daily."])],
        )
        for i in range(3)
    ]
    stats = await gb.time_verifier(v, claims, repeats=2)
    assert stats["verify_sentence"]["n"] == 6  # 3 claims x 2 timed repeats
    assert len(nli.batches) == 3 * 3  # plus one untimed warm-up pass
    assert stats["verify_sentence"]["p50_ms"] >= 10
    assert stats["verify_sentence_supported"]["n"] == 6
    assert stats["prepare"]["n"] == 3


async def test_numeric_rejects_are_excluded_from_the_nli_path_numbers():
    v = GroundingVerifier(FakeEmbedder(), FakeNLI(lambda p, h: (0.9, 0.0)), TH)
    ev = [make_chunk("c1", ["Take 10 mg once daily."])]
    claims = [
        Claim(id="a", sentence="Take 10 mg once daily.", citations=["c1"], evidence=ev),
        Claim(id="b", sentence="Take 99 mg once daily.", citations=["c1"], evidence=ev),
    ]
    stats = await gb.time_verifier(v, claims, repeats=1)
    assert stats["verify_sentence"]["n"] == 2
    assert stats["verify_sentence_nli_path"]["n"] == 1


async def test_run_bench_reports_budget_per_variant(thresholds):
    v = GroundingVerifier(FakeEmbedder(), FakeNLI(lambda p, h: (0.9, 0.0)), TH)
    claims = [
        Claim(
            id="a",
            sentence="Take with food.",
            citations=["c1"],
            evidence=[make_chunk("c1", ["Take with food."])],
        )
    ]
    report = await gb.run_bench(
        v,
        claims,
        thresholds,
        variants={"a": {}, "b": {"context": "span", "candidate_spans": 3}},
        repeats=1,
        hardware="test",
    )
    assert set(report["variants"]) == {"a", "b"}
    assert report["variants"]["b"]["config"] == {"context": "span", "candidate_spans": 3}
    assert report["budget_p95_ms"] == 150.0
    assert isinstance(report["variants"]["a"]["meets_budget"], bool)
    md = gb.render_markdown(report)
    assert "| a |" in md and "150 ms p95" in md


def test_variant_shares_models_and_overrides_settings():
    v = GroundingVerifier(FakeEmbedder(), FakeNLI(), TH, context="neighbour", heading="none")
    w = gb.variant(v, context="cascade", heading="sub", candidate_spans=3)
    assert w.nli is v.nli and w.embedder is v.embedder
    assert (w.context, w.heading, w.thresholds.candidate_spans) == ("cascade", "sub", 3)
    assert (v.context, v.heading, v.thresholds.candidate_spans) == ("neighbour", "none", 5)

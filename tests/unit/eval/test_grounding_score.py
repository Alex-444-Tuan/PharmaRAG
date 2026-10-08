"""The scoring math (spec section 11.2): B, S, reduction (B - S) / B, and the AT-2b guard."""

import pytest

from pharmarag.eval import grounding_score as gs
from pharmarag.eval.grounding_data import Claim
from pharmarag.grounding.verifier import Analysis
from tests.unit.grounding.test_verifier import TH, FakeNLI, make_chunk, verifier


def row(label, verdict, kind="x", origin="natural", reason=None, cid="g"):
    return gs.Scored(
        claim_id=cid,
        label=label,
        origin=origin,
        kind=kind,
        verdict=verdict,
        reason=reason,
        score=0.0,
        span=None,
        best_contra=None,
        sentence="s",
    )


def rows(n_h_pass, n_h_reject, n_s_pass, n_s_reject):
    return (
        [row("H", "supported") for _ in range(n_h_pass)]
        + [row("H", "unsupported") for _ in range(n_h_reject)]
        + [row("S", "supported") for _ in range(n_s_pass)]
        + [row("S", "unsupported") for _ in range(n_s_reject)]
    )


# --- reduction ---


def test_reduction_is_b_minus_s_over_b_with_counts():
    s = gs.summarise(rows(n_h_pass=3, n_h_reject=8, n_s_pass=28, n_s_reject=1))
    assert s["B_hallucinated"] == 11
    assert s["S_hallucinated_passed"] == 3
    assert s["reduction"] == pytest.approx(8 / 11)  # the spec's example: 11 falling to 3 = 73%
    assert s["reduction_text"] == "8/11 = 72.7%"


def test_spec_example_fifteen_falling_to_four():
    s = gs.summarise(rows(4, 11, 25, 0))
    assert s["reduction"] == pytest.approx(11 / 15)


def test_no_hallucinated_claims_means_no_reduction_not_a_divide_by_zero():
    s = gs.summarise(rows(0, 0, 10, 0))
    assert s["reduction"] is None
    assert s["at2_met"] is False and s["at2_valid"] is False


def test_verifier_that_catches_everything():
    s = gs.summarise(rows(0, 12, 28, 0))
    assert s["reduction"] == 1.0
    assert s["S_hallucinated_passed"] == 0


# --- AT-2b guard ---


def test_false_reject_rate_counts_every_non_supported_verdict_on_supported_claims():
    r = (
        [row("S", "supported")] * 17
        + [row("S", "unsupported")] * 1
        + [row("S", "contradicted")] * 1
        + [row("S", "unverified")] * 1
        + [row("H", "unsupported")] * 5
    )
    s = gs.summarise(r)
    assert s["supported_total"] == 20
    assert s["supported_false_rejected"] == 3
    assert s["at2b_false_reject_rate"] == pytest.approx(0.15)


def test_guard_passes_at_exactly_ten_percent_and_fails_above():
    ok = gs.summarise(rows(0, 10, 18, 2))  # 2 of 20 = 10%
    assert ok["at2b_false_reject_rate"] == pytest.approx(0.10)
    assert ok["at2b_pass"] is True
    bad = gs.summarise(rows(0, 10, 17, 3))  # 15%
    assert bad["at2b_pass"] is False


def test_a_verifier_that_rejects_everything_scores_100_percent_but_is_not_valid():
    s = gs.summarise(rows(0, 12, 0, 28))
    assert s["reduction"] == 1.0 and s["at2_met"] is True
    assert s["at2b_false_reject_rate"] == 1.0
    assert s["at2b_pass"] is False
    assert s["at2_valid"] is False  # the guard voids the reduction


def test_valid_needs_both_target_and_guard():
    assert gs.summarise(rows(2, 9, 28, 1))["at2_valid"] is True  # 82% and 3.4%
    assert gs.summarise(rows(5, 6, 28, 1))["at2_valid"] is False  # 54%: guard ok, target missed
    assert gs.summarise(rows(2, 9, 20, 8))["at2_valid"] is False  # 82% but 29% false rejects


def test_target_edge_is_inclusive():
    s = gs.summarise(rows(27, 73, 30, 0))  # exactly 73%
    assert s["reduction"] == pytest.approx(0.73)
    assert s["at2_met"] is True


def test_precision_and_recall_of_supported():
    s = gs.summarise(rows(n_h_pass=2, n_h_reject=8, n_s_pass=18, n_s_reject=2))
    assert s["precision_supported"] == pytest.approx(18 / 20)
    assert s["recall_supported"] == pytest.approx(18 / 20)


def test_breakdown_by_kind_and_reason_and_origin():
    r = [
        row("H", "unsupported", kind="numeric", origin="perturbed", reason="numeric_mismatch"),
        row("H", "supported", kind="negation"),
        row("S", "supported", kind="natural"),
    ]
    s = gs.summarise(r)
    assert s["by_kind"]["H:numeric"] == {"n": 1, "passed": 0}
    assert s["by_kind"]["H:negation"] == {"n": 1, "passed": 1}
    assert s["rejection_reasons"] == {"numeric_mismatch": 1}
    assert s["origin"] == {"H:perturbed": 1, "H:natural": 1, "S:natural": 1}


def test_pct_helper_shows_counts_beside_the_percentage():
    assert gs.pct(1, 8) == "1/8 = 12.5%"
    assert gs.pct(0, 0) == "0/0 = n/a"


# --- assembling the test set ---


def claim(cid, sentence="Take 10 mg.", label=None, **kw):
    text = "Take 10 mg once daily."
    return Claim(
        id=cid,
        sentence=sentence,
        citations=["c1"],
        evidence=[make_chunk("c1", [text])],
        label=label,
        **kw,
    )


def test_load_test_set_pairs_claims_with_human_labels():
    claims = [claim("g01"), claim("g02")]
    out = gs.load_test_set(claims, {"g01": "S", "g02": "H"})
    assert [(x.claim.id, x.label, x.origin) for x in out] == [
        ("g01", "S", "natural"),
        ("g02", "H", "natural"),
    ]


def test_load_test_set_requires_a_label_for_every_claim():
    with pytest.raises(ValueError, match="g02 has no label"):
        gs.load_test_set([claim("g01"), claim("g02")], {"g01": "S"})


def test_topup_replaces_the_named_claims_and_marks_them_perturbed():
    claims = [claim("g01"), claim("g02"), claim("g03")]
    labels = {"g01": "S", "g02": "S", "g03": "H"}
    topup = [claim("g02p", label="H", kind="numeric", meta={"replaces": "g02"})]
    out = gs.load_test_set(claims, labels, topup)
    assert [(x.claim.id, x.label, x.origin) for x in out] == [
        ("g01", "S", "natural"),
        ("g03", "H", "natural"),
        ("g02p", "H", "perturbed"),
    ]


def test_topup_planned_against_older_labels_is_refused():
    claims = [claim("g01"), claim("g02")]
    topup = [claim("g02p", label="H", kind="numeric", meta={"replaces": "g02"})]
    with pytest.raises(ValueError, match="not labelled S"):
        gs.load_test_set(claims, {"g01": "S", "g02": "H"}, topup)


def test_topup_must_be_labelled_h_and_point_at_a_real_claim():
    with pytest.raises(ValueError, match="must be labelled H"):
        gs.load_test_set(
            [claim("g01")], {"g01": "S"}, [claim("g01p", label="S", meta={"replaces": "g01"})]
        )
    with pytest.raises(ValueError, match="unknown claims"):
        gs.load_test_set(
            [claim("g01")], {"g01": "S"}, [claim("x", label="H", meta={"replaces": "g77"})]
        )


# --- running the real verifier object on claims (fake models) ---


async def test_analyse_all_shares_prepare_between_claims_with_the_same_evidence():
    from tests.unit.grounding.test_verifier import SpyEmbedder

    emb = SpyEmbedder()
    v = verifier(FakeNLI(lambda p, h: (0.9, 0.0)), embedder=emb)
    claims = [claim("g01"), claim("g02"), claim("g03")]
    analyses = await gs.analyse_all(v, claims)
    assert len(analyses) == 3
    span_batches = [c for c in emb.calls if c == ["Take 10 mg once daily."]]
    assert len(span_batches) == 1  # the one evidence set was embedded once


def test_score_analyses_applies_the_thresholds_to_saved_analyses():
    labelled = [
        gs.LabelledClaim(claim("g01"), "S", "natural"),
        gs.LabelledClaim(claim("g02"), "H", "natural"),
    ]
    analyses = [
        Analysis(
            candidates=["c1:0"],
            missing=[],
            best_entail=0.75,
            best_entail_span="c1:0",
            best_contra=0.1,
            best_contra_span="c1:0",
        ),
        Analysis(
            candidates=["c1:0"],
            missing=[],
            best_entail=0.75,
            best_entail_span="c1:0",
            best_contra=0.1,
            best_contra_span="c1:0",
        ),
    ]
    low = gs.score_analyses(labelled, analyses, TH)
    assert [r.verdict for r in low] == ["supported", "supported"]
    high = gs.score_analyses(labelled, analyses, TH.model_copy(update={"entail_min": 0.8}))
    assert [r.verdict for r in high] == ["unsupported", "unsupported"]
    assert gs.summarise(low)["S_hallucinated_passed"] == 1
    assert gs.summarise(high)["S_hallucinated_passed"] == 0


async def test_end_to_end_scoring_with_a_scripted_verifier():
    # S claim restates the label; H claim has a wrong dose; H claim is a flipped negation that the
    # scripted NLI contradicts.
    def nli(p, h):
        if "not" in h:
            return (0.02, 0.95)
        return (0.95, 0.01)

    v = verifier(FakeNLI(nli))
    labelled = [
        gs.LabelledClaim(claim("g01", "Take 10 mg once daily."), "S", "natural"),
        gs.LabelledClaim(claim("g02", "Take 40 mg once daily."), "H", "natural"),
        gs.LabelledClaim(claim("g03", "Do not take once daily."), "H", "natural"),
    ]
    analyses = await gs.analyse_all(v, [lc.claim for lc in labelled])
    rows_ = gs.score_analyses(labelled, analyses, TH)
    assert [r.verdict for r in rows_] == ["supported", "unsupported", "contradicted"]
    s = gs.summarise(rows_)
    assert (s["B_hallucinated"], s["S_hallucinated_passed"], s["reduction"]) == (2, 0, 1.0)
    assert s["at2b_false_reject_rate"] == 0.0 and s["at2_valid"] is True


def test_build_report_records_models_thresholds_dataset_and_corpus(thresholds, tmp_path):
    ds = tmp_path / "claims.jsonl"
    ds.write_text("{}\n")
    v = verifier(FakeNLI())
    report = gs.build_report(
        split="test",
        rows=rows(1, 9, 28, 2),
        thresholds=thresholds,
        verifier=v,
        llm_model_id="claude-sonnet-5-5",
        corpus="abc123",
        claims_path=ds,
        extra_datasets={"labels.jsonl": "deadbeef"},
        note="n",
    )
    assert report["eval"] == "grounding" and report["split"] == "test"
    assert report["model_id"] == "claude-sonnet-5-5"
    assert report["nli_model"] == "fake-nli"
    assert report["embedding_model"] == "fake-embedder"
    assert report["corpus_version"] == "abc123"
    assert report["verifier_config"] == {
        "context": "neighbour",
        "heading": "none",
        "candidate_spans": 5,
        "nearest_spans": 3,
        "entail_min": 0.7,
        "contradict_min": 0.8,
    }
    assert "thresholds_commit" in report and "thresholds_dirty" in report
    assert report["thresholds"]["grounding"]["entail_min"] == thresholds.grounding.entail_min
    assert len(report["dataset"]["sha256"]) == 64
    assert report["other_datasets_sha256"] == {"labels.jsonl": "deadbeef"}
    assert report["runs"] == 1
    assert report["summary"]["B_hallucinated"] == 10
    md = gs.render_markdown(report)
    for needle in ("AT-2b", "reduction", "abc123", "claude-sonnet-5-5"):
        assert needle in md
    assert "90.0%" in md and "B=10, S=1" in md  # counts beside the percentage
    assert "2 of 30" in md


def test_markdown_flags_an_invalid_result(thresholds, tmp_path):
    ds = tmp_path / "c.jsonl"
    ds.write_text("{}\n")
    report = gs.build_report(
        split="test",
        rows=rows(0, 12, 0, 28),
        thresholds=thresholds,
        verifier=verifier(FakeNLI()),
        llm_model_id=None,
        corpus="c",
        claims_path=ds,
        extra_datasets={},
        note="n",
    )
    md = gs.render_markdown(report)
    assert "AT-2b guard: FAIL" in md
    assert "NOT VALID" in md

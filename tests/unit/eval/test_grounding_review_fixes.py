"""Cycle 3 review fixes on the eval side: verify_many timing, the harder dev kinds."""

import random

import pytest

from pharmarag.eval import grounding_bench as gb
from pharmarag.eval import grounding_dev as dev
from pharmarag.eval.grounding_data import Claim
from pharmarag.graph.state import Chunk, Span
from pharmarag.grounding.numeric import missing_quantities
from pharmarag.grounding.verifier import GroundingVerifier
from pharmarag.models.embedder import FakeEmbedder
from tests.unit.grounding.test_verifier import TH, FakeNLI, make_chunk

# --- verify_many latency harness ---


def answer(n_sentences: int, tag: str) -> list[Claim]:
    ev = [make_chunk("c1", ["Take with food.", "Avoid alcohol.", "Store dry."])]
    return [
        Claim(id=f"{tag}{i}", sentence=f"Take with food {i}.", citations=["c1"], evidence=ev)
        for i in range(n_sentences)
    ]


def test_evidence_groups_collect_the_sentences_of_one_answer():
    claims = answer(2, "a") + [
        Claim(
            id="b0",
            sentence="x",
            citations=["c1"],
            evidence=[make_chunk("c1", ["Other."]).model_copy(update={"chunk_id": 99})],
        )
    ]
    groups = gb.evidence_groups(claims)
    assert [len(g) for g in groups] == [2, 1]


async def test_verify_many_timing_covers_batches_of_one_to_four():
    nli = FakeNLI(lambda p, h: (0.9, 0.0))
    v = GroundingVerifier(FakeEmbedder(), nli, TH)
    stats = await gb.time_verify_many(v, answer(5, "a") + answer(3, "b")[:0], repeats=2)
    assert list(stats) == ["1", "2", "3", "4"]
    assert stats["4"]["answers"] == 1 and stats["4"]["n"] == 2  # 1 answer x 2 repeats
    assert stats["4"]["per_sentence_p50_ms"] == pytest.approx(stats["4"]["p50_ms"] / 4, abs=0.06)


async def test_verify_many_timing_makes_one_nli_batch_per_timed_call():
    nli = FakeNLI(lambda p, h: (0.9, 0.0))
    v = GroundingVerifier(FakeEmbedder(), nli, TH, context="neighbour")
    await gb.time_verify_many(v, answer(4, "a"), sizes=(4,), repeats=3)
    assert len(nli.batches) == 4  # 1 untimed warm-up + 3 timed, each ONE batch of 4 sentences


async def test_sizes_larger_than_any_answer_are_skipped():
    v = GroundingVerifier(FakeEmbedder(), FakeNLI(lambda p, h: (0.9, 0.0)), TH)
    stats = await gb.time_verify_many(v, answer(2, "a"), repeats=1)
    assert list(stats) == ["1", "2"]


async def test_run_bench_includes_the_verify_many_table(thresholds):
    v = GroundingVerifier(FakeEmbedder(), FakeNLI(lambda p, h: (0.9, 0.0)), TH)
    report = await gb.run_bench(
        v, answer(4, "a"), thresholds, variants={"a": {}}, repeats=1, hardware="t"
    )
    assert set(report["verify_many"]) == {"1", "2", "3", "4"}
    assert report["verify_many_config"]["candidate_spans"] == 5
    md = gb.render_markdown(report)
    assert "verify_many" in md and "per-sentence p95" in md


# --- the harder dev kinds ---

TABLE_ROW = (
    "Renal Function Creatinine Clearance (mL/min): >30 to 59; "
    "Total Daily Dose Range (mg/day): 400 to 1400; Dose Regimen (mg): 200 BID"
)
ADVERSE_ROW = "Adverse Reaction: Arthralgia; % Placebo N=7,311: 6.5; % 10 mg N=3,908: 8.9"


def test_row_sentence_restates_a_unit_column():
    rng = random.Random(1)
    out = {dev.row_sentence(TABLE_ROW, random.Random(s)) for s in range(20)}
    assert any("400 to 1400 mg/day" in o for o in out if o)
    assert all(
        o.startswith("For renal function creatinine clearance >30 to 59 mL/min,") for o in out if o
    )
    del rng


def test_row_sentence_restates_a_percent_column():
    out = {dev.row_sentence(ADVERSE_ROW, random.Random(s)) for s in range(20)}
    assert "Arthralgia occurred in 8.9% of patients in the 10 mg group." in out
    assert "Arthralgia occurred in 6.5% of patients in the Placebo group." in out


def test_row_sentences_are_supported_by_their_row_under_the_number_check():
    # the row restatement must pass the check (header units carried), or the S label is a lie
    for row in (TABLE_ROW, ADVERSE_ROW):
        for s in range(10):
            sentence = dev.row_sentence(row, random.Random(s))
            if sentence:
                assert missing_quantities(sentence, [row]) == [], sentence


def test_row_sentence_declines_a_row_without_a_numeric_column():
    assert dev.row_sentence("Drug: Warfarin; Effect: bleeding", random.Random(1)) is None


def test_row_spans_picks_only_table_rows():
    text = "Intro sentence here. " + ADVERSE_ROW
    start = text.index("Adverse")
    chunk = Chunk(
        id="c1", chunk_id=1, setid="00000000-0000-0000-0000-000000000001", spl_version=1,
        ingredient_key="1", section_loinc="34084-4", section_title="s", text=text,
        spans=[Span(i=0, start=0, end=20), Span(i=1, start=start, end=len(text))], fused_score=0,
    )  # fmt: skip
    assert [sp.i for sp, _ in dev.row_spans(chunk)] == [1]


def test_tallman_name_and_plain_name():
    t = "The initial dosage of PredniSONE tablets may vary."
    assert dev.tallman_name(t) == "PredniSONE"
    assert dev.plain_name(t, "PredniSONE") == "The initial dosage of prednisone tablets may vary."
    assert dev.plain_name("HydrALAZINE is a vasodilator.", "HydrALAZINE").startswith("Hydralazine")
    assert dev.tallman_name("Use eGFR to guide dosing.") is None  # not a drug name
    assert dev.tallman_name("Plain prednisone.") is None

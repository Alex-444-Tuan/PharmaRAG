"""Cycle 3 review fixes for the verifier: tall-man casing, verify_many, numbers checked against
the premise that entails, and the cross-drug number pass (spec review log round 5)."""

import pytest

from pharmarag.graph.state import Sentence
from pharmarag.grounding.spans import SpanRecord, eligible_for_nearest
from pharmarag.grounding.verifier import (
    REASON_NUMERIC,
    CrossEncoderNLI,
    fold_tallman,
)
from tests.unit.grounding.test_factory import _cached_nli_path
from tests.unit.grounding.test_verifier import (
    TH,
    FakeNLI,
    SpyEmbedder,
    make_chunk,
    sent,
    verifier,
)

# --- tall-man lettering ---


@pytest.mark.parametrize(
    ("text", "folded"),
    [
        ("PredniSONE tablets", "prednisone tablets"),
        ("PARoxetine and FLUoxetine", "paroxetine and fluoxetine"),
        ("hydrALAZINE vs hydrOXYzine", "hydralazine vs hydroxyzine"),
        ("buPROPion", "bupropion"),
        ("glipiZIDE", "glipizide"),
    ],
)
def test_tallman_names_are_case_folded(text, folded):
    assert fold_tallman(text) == folded


@pytest.mark.parametrize(
    "text",
    [
        "HIV and CYP2D6 and NSAID",  # all capitals: acronyms stay
        "Prednisone is a Steroid",  # ordinary capitalisation stays
        "take 5 mg of metformin",
        "mL and mcg",  # a single capital in a unit
    ],
)
def test_ordinary_text_is_not_changed(text):
    assert fold_tallman(text) == text


async def test_the_nli_model_sees_folded_text_in_premise_and_hypothesis():
    nli = FakeNLI(lambda p, h: (0.9, 0.0))
    ev = [make_chunk("c1", ["PredniSONE is started at 5 mg daily."])]
    v = verifier(nli, context="span")
    prepared = await v.prepare(ev, [])
    await v.verify_sentence(sent("PredniSONE is started at 5 mg daily."), prepared)
    premise, hypothesis = nli.batches[0][0]
    assert premise == "prednisone is started at 5 mg daily."
    assert hypothesis == "prednisone is started at 5 mg daily."


async def test_the_output_sentence_text_keeps_its_original_casing():
    nli = FakeNLI(lambda p, h: (0.9, 0.0))
    ev = [make_chunk("c1", ["PredniSONE is started at 5 mg daily."])]
    v = verifier(nli)
    out = (await v.verify([sent("PredniSONE is started at 5 mg daily.")], ev, []))[0]
    assert out.text == "PredniSONE is started at 5 mg daily."


@pytest.fixture(scope="module")
def real_nli():
    path = _cached_nli_path()
    if path is None:
        pytest.skip("cross-encoder/nli-deberta-v3-small is not in the local cache")
    return CrossEncoderNLI(path)


PRED_PREMISE = (
    "The initial dosage of PredniSONE tablets may vary from 5 mg to 60 mg per day, "
    "depending on the specific disease entity being treated."
)
PRED_CLAIM = PRED_PREMISE.replace("PredniSONE", "prednisone")


def test_real_model_mistakes_tallman_casing_for_a_contradiction(real_nli):
    # This is the bug: a verbatim restatement scores as a contradiction (0.97 on this label).
    e, c = real_nli.score([(PRED_PREMISE, PRED_CLAIM)])
    assert c[0] > 0.8 and e[0] < 0.1


def test_real_model_agrees_once_the_casing_is_folded(real_nli):
    e, c = real_nli.score([(fold_tallman(PRED_PREMISE), fold_tallman(PRED_CLAIM))])
    assert e[0] > 0.9 and c[0] < 0.1


async def test_real_model_verifier_supports_the_restatement(real_nli):
    from pharmarag.models.embedder import FakeEmbedder
    from tests.unit.grounding.test_verifier import GroundingVerifier

    v = GroundingVerifier(FakeEmbedder(), real_nli, TH, context="span")
    ev = [make_chunk("c1", [PRED_PREMISE])]
    out = (await v.verify([sent(PRED_CLAIM)], ev, []))[0]
    assert out.verdict == "supported", out


# --- verify_many ---

EV = [
    make_chunk("c1", ["The usual starting dose is 10 mg once daily.", "Take with food."]),
    make_chunk("c2", ["Avoid use in severe renal impairment.", "Store at room temperature."]),
]


def entail_if_food(p, h):
    return (0.95, 0.0) if "food" in h.lower() else (0.1, 0.1)


MIXED = [
    sent("Take with food."),  # supported
    sent("The usual starting dose is 40 mg once daily."),  # number not in any candidate span
    sent("Take at night."),  # NLI says no
    Sentence(text="Anything.", citations=[], verdict="unsupported", reason="no_citation"),
    sent("Store with food.", ("c2",)),
    Sentence(text="Uncited but unverified.", citations=[]),  # defence in depth: unsupported
]


@pytest.mark.parametrize("context", ["neighbour", "span", "both", "cascade"])
async def test_verify_many_equals_verify_sentence_on_each(context):
    v1 = verifier(FakeNLI(entail_if_food), context=context)
    p1 = await v1.prepare(EV, [])
    one_by_one = [await v1.verify_sentence(s, p1) for s in MIXED]
    v2 = verifier(FakeNLI(entail_if_food), context=context)
    p2 = await v2.prepare(EV, [])
    many = await v2.verify_many(list(MIXED), p2)
    assert many == one_by_one
    assert [m.text for m in many] == [s.text for s in MIXED]  # order kept


async def test_verify_many_uses_one_nli_batch_for_all_sentences():
    nli = FakeNLI(entail_if_food)
    v = verifier(nli, context="neighbour")
    prepared = await v.prepare(EV, [])
    out = await v.verify_many(
        [sent("Take with food."), sent("Take at night."), sent("Eat with food.")], prepared
    )
    assert len(nli.batches) == 1
    per_sentence = [
        len(a.candidates)
        for a in await verifier(FakeNLI()).analyse_many(
            [sent("Take with food."), sent("Take at night."), sent("Eat with food.")], prepared
        )
    ]
    assert len(nli.batches[0]) == sum(per_sentence)  # every sentence's pairs, together
    assert [o.verdict for o in out] == ["supported", "unsupported", "supported"]


async def test_verify_many_batch_holds_every_sentences_pairs():
    nli = FakeNLI(entail_if_food)
    v = verifier(nli, context="neighbour")
    prepared = await v.prepare(EV, [])
    await v.verify_many([sent("Take with food."), sent("Take at night.")], prepared)
    hypotheses = [h for _, h in nli.batches[0]]
    assert set(hypotheses) == {"Take with food.", "Take at night."}
    assert hypotheses.count("Take with food.") == hypotheses.count("Take at night.") > 0


async def test_cascade_verify_many_runs_two_batches_at_most_and_only_for_the_unsettled():
    # span-only premises entail; the neighbour premise (contains two sentences) does not
    nli = FakeNLI(lambda p, h: (0.95, 0.0) if p.count(".") == 1 else (0.05, 0.05))
    v = verifier(nli, context="cascade")
    prepared = await v.prepare(EV, [])
    out = await v.verify_many([sent("Take with food."), sent("Store dry.")], prepared)
    assert len(nli.batches) == 2  # not 2 per sentence
    assert len(nli.batches[1]) == len(nli.batches[0])  # both needed the second stage
    assert all(o.verdict == "supported" for o in out)


async def test_cascade_verify_many_skips_the_second_batch_when_all_are_settled():
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    v = verifier(nli, context="cascade")
    prepared = await v.prepare(EV, [])
    await v.verify_many([sent("Take with food."), sent("Store dry.")], prepared)
    assert len(nli.batches) == 1


async def test_verify_many_with_nothing_to_check_makes_no_model_call():
    nli = FakeNLI()
    emb = SpyEmbedder()
    v = verifier(nli, embedder=emb)
    prepared = await v.prepare(EV, [])
    n_embeds = len(emb.calls)
    out = await v.verify_many(
        [Sentence(text="x", citations=[], verdict="unsupported", reason="no_citation")], prepared
    )
    assert nli.batches == [] and len(emb.calls) == n_embeds
    assert out[0].reason == "no_citation"
    assert await v.verify_many([], prepared) == []


async def test_verify_many_embeds_all_sentences_in_one_call():
    emb = SpyEmbedder()
    v = verifier(FakeNLI(entail_if_food), embedder=emb)
    prepared = await v.prepare(EV, [])
    emb.calls.clear()
    await v.verify_many(
        [sent("Take with food."), sent("Take at night."), sent("Eat food.")], prepared
    )
    assert len(emb.calls) == 1 and len(emb.calls[0]) == 3


# --- numbers must be in the premise that entails (round 5.5) ---


async def test_dose_in_one_span_cannot_be_borrowed_for_a_claim_entailed_by_another():
    # c1 has two separate spans. The sentence asks for 10 mg and 40 mg. NLI is satisfied by the
    # first span alone (it "entails" because the scripted model says so), but 40 mg is only in
    # the second span, so no single premise states both numbers.
    ev = [
        make_chunk(
            "c1", ["The usual dose is 10 mg daily.", "Other facts.", "The maximum is 40 mg."]
        )
    ]
    nli = FakeNLI(lambda p, h: (0.95, 0.0) if "usual dose" in p else (0.05, 0.05))
    v = verifier(nli, context="span")
    out = await run(v, sent("The usual dose is 10 mg daily and the maximum is 40 mg."), ev)
    assert (out.verdict, out.reason) == ("unsupported", REASON_NUMERIC)


async def test_a_premise_that_holds_both_numbers_supports_the_claim():
    ev = [make_chunk("c1", ["The usual dose is 10 mg daily.", "The maximum is 40 mg."])]
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    v = verifier(nli, context="neighbour")  # the neighbour sentence is in the premise
    out = await run(v, sent("The usual dose is 10 mg daily and the maximum is 40 mg."), ev)
    assert out.verdict == "supported"


async def test_a_second_premise_can_rescue_a_numbers_failure():
    ev = [make_chunk("c1", ["The usual dose is 10 mg daily.", "Unrelated sentence here."])]
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    v = verifier(nli, context="both")
    out = await run(v, sent("The usual dose is 10 mg daily."), ev)
    assert out.verdict == "supported"


async def run(v, s, ev):
    prepared = await v.prepare(ev, [])
    return await v.verify_sentence(s, prepared)


def two_drugs():
    paroxetine = make_chunk("c1", ["The maximum paroxetine dose is 50 mg daily."])
    sertraline = make_chunk("c2", ["The maximum sertraline dose is 200 mg daily."])
    return [
        paroxetine.model_copy(update={"ingredient_key": "32937"}),
        sertraline.model_copy(update={"ingredient_key": "36437"}),
    ]


async def test_paroxetine_claim_cannot_borrow_sertralines_200_mg():
    # The writer cites paroxetine but states sertraline's dose. NLI is scripted to say yes to
    # anything, as a generic NLI model can; the number check must be what stops it.
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    v = verifier(nli)
    out = await run(v, sent("The maximum paroxetine dose is 200 mg daily.", ("c1",)), two_drugs())
    assert (out.verdict, out.reason) == ("unsupported", REASON_NUMERIC)
    # the other drug's span was never offered to the NLI model either
    assert not any("sertraline" in p for batch in nli.batches for p, _ in batch)


async def test_the_same_sentence_about_the_right_drug_passes():
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    out = await run(
        verifier(nli),
        sent("The maximum sertraline dose is 200 mg daily.", ("c2",)),
        two_drugs(),
    )
    assert out.verdict == "supported"


async def test_a_two_drug_sentence_that_cites_both_may_use_both():
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    out = await run(
        verifier(nli),
        sent("The maximum doses are 50 mg for paroxetine and 200 mg for sertraline.", ("c1", "c2")),
        two_drugs(),
    )
    # Both quantities exist in the pool, but each premise holds only one of them, so no single
    # premise states both. The check is strict here (a known false reject for a sentence that
    # merges facts from two chunks); the alternative is the cross-drug pass this fixes.
    assert (out.verdict, out.reason) == ("unsupported", REASON_NUMERIC)


# --- eligibility of nearest-k spans ---


def rec(rid, source, key="", rx=""):
    return SpanRecord(id=rid, source=source, text="t", context="t", ingredient_key=key, rxcui=rx)


def test_only_the_cited_drugs_spans_are_eligible_for_nearest():
    recs = [rec("c1:0", "c1", "A"), rec("c1:1", "c1", "A"), rec("c2:0", "c2", "B")]
    assert eligible_for_nearest(recs, ["c1"]) == [True, True, False]
    assert eligible_for_nearest(recs, ["c2"]) == [False, False, True]
    assert eligible_for_nearest(recs, ["c1", "c2"]) == [True, True, True]


def test_combination_drug_chunks_share_ingredients():
    recs = [rec("c1:0", "c1", "A+B"), rec("c2:0", "c2", "B"), rec("c3:0", "c3", "C")]
    assert eligible_for_nearest(recs, ["c2"]) == [True, True, False]


def test_facts_follow_the_cited_chunks_ingredients():
    recs = [rec("c1:0", "c1", "111"), rec("r1", "r1", rx="111"), rec("r2", "r2", rx="222")]
    assert eligible_for_nearest(recs, ["c1"]) == [True, True, False]
    # a cited fact makes its own ingredient eligible
    assert eligible_for_nearest(recs, ["r2"]) == [False, False, True]


def test_no_known_citation_means_no_restriction():
    recs = [rec("c1:0", "c1", "A"), rec("c2:0", "c2", "B")]
    assert eligible_for_nearest(recs, []) == [True, True]
    assert eligible_for_nearest(recs, ["c9"]) == [True, True]


# --- the union premise: a sentence that merges two spans ---


def both_facts_rule(p, h):
    """Entailed only by a premise that holds BOTH facts, as a conjunctive claim needs."""
    return (0.95, 0.0) if ("food" in p and "alcohol" in p) else (0.05, 0.05)


async def test_union_premise_supports_a_sentence_that_merges_two_spans():
    ev = [make_chunk("c1", ["Take with food.", "Filler one.", "Filler two.", "Avoid alcohol."])]
    claim = sent("Take with food, and avoid alcohol.")
    for context in ("cascade", "both"):
        nli = FakeNLI(both_facts_rule)
        out = await run(verifier(nli, context=context), claim, ev)
        assert out.verdict == "supported", context
        assert "+" in out.span  # the union premise is named by the spans it joins


async def test_without_the_union_the_merged_sentence_is_rejected():
    ev = [make_chunk("c1", ["Take with food.", "Filler one.", "Filler two.", "Avoid alcohol."])]
    for context in ("neighbour", "span"):
        out = await run(
            verifier(FakeNLI(both_facts_rule), context=context),
            sent("Take with food, and avoid alcohol."),
            ev,
        )
        assert out.verdict == "unsupported", context


async def test_union_is_tried_last_and_only_when_needed():
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    await run(verifier(nli, context="cascade"), sent("Take with food."), EV)
    assert len(nli.batches) == 1  # settled by the first premise
    nli2 = FakeNLI(both_facts_rule)
    ev = [make_chunk("c1", ["Take with food.", "Filler one.", "Filler two.", "Avoid alcohol."])]
    await run(verifier(nli2, context="cascade"), sent("Take with food, and avoid alcohol."), ev)
    assert len(nli2.batches) == 3  # neighbour, span, union
    assert len(nli2.batches[2]) == 1  # one extra pair per sentence


async def test_union_numbers_must_all_be_in_the_union_text():
    ev = [make_chunk("c1", ["Dose is 10 mg.", "Filler one.", "Filler two.", "Maximum is 40 mg."])]
    nli = FakeNLI(lambda p, h: (0.95, 0.0) if "10 mg" in p and "40 mg" in p else (0.05, 0.05))
    ok = await run(verifier(nli, context="cascade"), sent("Dose is 10 mg, maximum is 40 mg."), ev)
    assert ok.verdict == "supported"
    bad = await run(verifier(nli, context="cascade"), sent("Dose is 10 mg, maximum is 80 mg."), ev)
    assert (bad.verdict, bad.reason) == ("unsupported", REASON_NUMERIC)

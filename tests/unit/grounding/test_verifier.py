"""Grounding verifier (spec section 9, "Verifier, per sentence"). No real models: a scripted NLI
scorer and the FakeEmbedder stand in, so these tests pin down the logic around the models."""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pharmarag.app.settings import GroundingThresholds
from pharmarag.graph.state import Chunk, Fact, Sentence, Span
from pharmarag.grounding.verifier import (
    REASON_CONTRADICTED,
    REASON_LOW_ENTAILMENT,
    REASON_NO_CITATION,
    REASON_NO_EVIDENCE,
    REASON_NUMERIC,
    Analysis,
    CrossEncoderNLI,
    GroundingVerifier,
    decide,
    resolve_label_indices,
    softmax,
)
from pharmarag.models.embedder import FakeEmbedder

SETID = "00000000-0000-0000-0000-000000000001"
TH = GroundingThresholds(
    entail_min=0.70, contradict_min=0.80, support_ratio_min=0.8, candidate_spans=5, nearest_spans=3
)


def make_chunk(cid: str, sentences: list[str], subsection: str | None = None) -> Chunk:
    text, spans = "", []
    for i, s in enumerate(sentences):
        start = len(text) + (1 if text else 0)
        text = f"{text} {s}" if text else s
        spans.append(Span(i=i, start=start, end=start + len(s)))
    return Chunk(
        id=cid,
        chunk_id=int(cid[1:]),
        setid=SETID,
        spl_version=1,
        ingredient_key="1",
        section_loinc="34068-7",
        section_title="2 DOSAGE AND ADMINISTRATION",
        subsection_title=subsection,
        text=text,
        spans=spans,
        fused_score=0.5,
    )


class FakeNLI:
    """Scripted scorer. `rule(premise, hypothesis) -> (p_entail, p_contra)`.

    Records every batch, the thread it ran in, and whether two batches ever overlapped."""

    model_name = "fake-nli"

    def __init__(self, rule=None, delay_s: float = 0.0):
        self.rule = rule or (lambda p, h: (0.0, 0.0))
        self.delay_s = delay_s
        self.batches: list[list[tuple[str, str]]] = []
        self.threads: set[int] = set()
        self.max_active = 0
        self._active = 0
        self._guard = threading.Lock()

    def score(self, pairs):
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            self.threads.add(threading.get_ident())
            self.batches.append(list(pairs))
            if self.delay_s:
                time.sleep(self.delay_s)
            out = [self.rule(p, h) for p, h in pairs]
            return np.array([o[0] for o in out]), np.array([o[1] for o in out])
        finally:
            with self._guard:
                self._active -= 1


class SpyEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return super().embed_documents(texts)


def verifier(nli=None, embedder=None, th=TH, **kw) -> GroundingVerifier:
    return GroundingVerifier(embedder or FakeEmbedder(), nli or FakeNLI(), th, **kw)


EVIDENCE = [
    make_chunk("c1", ["The usual starting dose is 10 mg once daily.", "Take with food."]),
    make_chunk("c2", ["Avoid use in severe renal impairment.", "Store at room temperature."]),
]


def sent(text: str, cites=("c1",), **kw) -> Sentence:
    return Sentence(text=text, citations=list(cites), **kw)


# --- verdicts and their edges (spec step 4) ---


def an(entail=None, contra=None, missing=(), cands=("c1:0",)) -> Analysis:
    return Analysis(
        candidates=list(cands),
        missing=list(missing),
        best_entail=entail,
        best_entail_span="c1:0",
        best_contra=contra,
        best_contra_span="c1:1",
    )


def test_supported_at_exactly_entail_min():
    assert decide(an(0.70, 0.0), TH) == ("supported", None)


def test_just_below_entail_min_is_not_supported():
    assert decide(an(0.6999, 0.0), TH)[0] == "unsupported"


def test_contradicted_at_exactly_contradict_min_when_not_supported():
    assert decide(an(0.10, 0.80), TH) == ("contradicted", REASON_CONTRADICTED)


def test_just_below_contradict_min_is_unsupported():
    assert decide(an(0.10, 0.7999), TH) == ("unsupported", REASON_LOW_ENTAILMENT)


def test_entailment_wins_over_contradiction():
    # both high is possible across different spans; supported takes precedence (spec step 4)
    assert decide(an(0.90, 0.95), TH)[0] == "supported"


def test_numeric_miss_is_unsupported_even_with_high_scores():
    from decimal import Decimal

    from pharmarag.grounding.numeric import Quantity

    miss = [Quantity(Decimal(20), "mg", "20 mg")]
    assert decide(an(0.99, 0.0, missing=miss), TH) == ("unsupported", REASON_NUMERIC)


def test_no_candidates_is_unsupported_no_evidence():
    assert decide(Analysis(candidates=[], missing=[]), TH) == ("unsupported", REASON_NO_EVIDENCE)


@pytest.mark.parametrize("entail_min", [0.0, 0.5, 0.99])
def test_thresholds_come_from_the_object_not_constants(entail_min):
    th = TH.model_copy(update={"entail_min": entail_min})
    assert decide(an(0.5, 0.0), th)[0] == ("supported" if entail_min <= 0.5 else "unsupported")


async def run_one(v, s, evidence=EVIDENCE, facts=()):
    prepared = await v.prepare(list(evidence), list(facts))
    return await v.verify_sentence(s, prepared)


async def test_supported_sentence_gets_score_span_and_no_reason():
    nli = FakeNLI(lambda p, h: (0.95, 0.01) if "10 mg once daily" in p else (0.01, 0.01))
    out = await run_one(verifier(nli), sent("The starting dose is 10 mg once daily."))
    assert out.verdict == "supported"
    assert out.score == pytest.approx(0.95)
    assert out.span == "c1:0"
    assert out.reason is None


async def test_contradicted_sentence_points_at_the_contradicting_span():
    nli = FakeNLI(lambda p, h: (0.02, 0.97) if "renal" in p else (0.02, 0.05))
    out = await run_one(verifier(nli), sent("Use freely in severe renal impairment.", ("c2",)))
    assert out.verdict == "contradicted"
    assert out.reason == REASON_CONTRADICTED
    assert out.span == "c2:0"


async def test_unsupported_sentence_keeps_best_entailment_as_score():
    nli = FakeNLI(lambda p, h: (0.40, 0.10))
    out = await run_one(verifier(nli), sent("Take with milk."))
    assert (out.verdict, out.reason) == ("unsupported", REASON_LOW_ENTAILMENT)
    assert out.score == pytest.approx(0.40)


# --- number check (spec step 2) ---


async def test_wrong_dose_is_rejected_without_calling_nli():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))  # an NLI model that would pass anything
    out = await run_one(verifier(nli), sent("The usual starting dose is 20 mg once daily."))
    assert (out.verdict, out.reason) == ("unsupported", REASON_NUMERIC)
    assert nli.batches == []  # the deterministic check settles it; no model time spent


async def test_correct_dose_passes_the_check_and_reaches_nli():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    out = await run_one(verifier(nli), sent("Start with 10mg once daily."))
    assert out.verdict == "supported"
    assert nli.batches


async def test_number_can_be_found_in_a_nearest_span_of_an_uncited_chunk():
    # sentence cites c2, but its dose appears only in c1; the nearest-span search finds it
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    out = await run_one(
        verifier(nli), sent("The usual starting dose is 10 mg once daily.", ("c2",))
    )
    assert out.reason != REASON_NUMERIC


# --- already-rejected sentences, uncited sentences ---


async def test_unsupported_sentence_is_returned_unchanged():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    s = Sentence(text="Anything.", citations=[], verdict="unsupported", reason=REASON_NO_CITATION)
    out = await run_one(verifier(nli), s)
    assert out == s
    assert nli.batches == []


async def test_uncited_sentence_is_never_supported_even_if_it_slips_through():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    out = await run_one(verifier(nli), Sentence(text="Take with food.", citations=[]))
    assert (out.verdict, out.reason) == ("unsupported", REASON_NO_CITATION)
    assert nli.batches == []


async def test_no_evidence_means_unsupported():
    out = await run_one(verifier(), sent("Take with food."), evidence=[])
    assert (out.verdict, out.reason) == ("unsupported", REASON_NO_EVIDENCE)


async def test_rxnorm_fact_is_a_citable_span():
    facts = [Fact(id="r1", rxcui="1", text="Zestril 10 mg oral tablet", retrieval_attempt=1)]
    nli = FakeNLI(lambda p, h: (0.95, 0.0) if "Zestril" in p else (0.0, 0.0))
    out = await run_one(
        verifier(nli), sent("Zestril comes as a 10 mg tablet.", ("r1",)), facts=facts
    )
    assert out.verdict == "supported" and out.span == "r1"


# --- NLI batching and premise (spec step 3) ---


async def test_all_candidate_pairs_go_in_one_batch():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    await run_one(verifier(nli, context="neighbour"), sent("Take with food.", ("c1", "c2")))
    assert len(nli.batches) == 1
    assert len(nli.batches[0]) == 4  # c1 and c2 each have 2 spans, all within 5 candidates
    assert all(h == "Take with food." for _, h in nli.batches[0])


async def test_candidate_spans_threshold_caps_the_batch():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    th = TH.model_copy(update={"candidate_spans": 2})
    await run_one(verifier(nli, th=th, context="neighbour"), sent("Take with food.", ("c1", "c2")))
    assert len(nli.batches[0]) == 2


async def test_neighbour_premise_includes_the_adjacent_sentence():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    await run_one(verifier(nli, context="neighbour"), sent("Take with food.", ("c1",)))
    premises = {p for p, _ in nli.batches[0]}
    assert "The usual starting dose is 10 mg once daily. Take with food." in premises


async def test_span_premise_is_the_span_alone():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    await run_one(verifier(nli, context="span"), sent("Take with food.", ("c1",)))
    premises = {p for p, _ in nli.batches[0]}
    all_spans = {
        "The usual starting dose is 10 mg once daily.",
        "Take with food.",
        "Avoid use in severe renal impairment.",
        "Store at room temperature.",
    }
    assert {"The usual starting dose is 10 mg once daily.", "Take with food."} <= premises
    assert premises <= all_spans  # single spans only: no joined neighbour text


async def test_heading_leads_the_premise_when_enabled():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    ev = [make_chunk("c1", ["Start at 10 mg."], subsection="2.1 Adult Dosage")]
    await run_one(verifier(nli, context="span", heading="sub"), sent("Adults start at 10 mg."), ev)
    assert nli.batches[0][0][0] == "2.1 Adult Dosage: Start at 10 mg."
    nli2 = FakeNLI(lambda p, h: (0.99, 0.0))
    await run_one(
        verifier(nli2, context="span", heading="full"), sent("Adults start at 10 mg."), ev
    )
    assert (
        nli2.batches[0][0][0] == "2 DOSAGE AND ADMINISTRATION > 2.1 Adult Dosage: Start at 10 mg."
    )
    nli3 = FakeNLI(lambda p, h: (0.99, 0.0))
    await run_one(
        verifier(nli3, context="span", heading="none"), sent("Adults start at 10 mg."), ev
    )
    assert nli3.batches[0][0][0] == "Start at 10 mg."


async def test_heading_falls_back_to_the_section_title():
    nli = FakeNLI(lambda p, h: (0.99, 0.0))
    ev = [make_chunk("c1", ["Start at 10 mg."])]
    await run_one(verifier(nli, context="span", heading="sub"), sent("Start at 10 mg."), ev)
    assert nli.batches[0][0][0] == "2 DOSAGE AND ADMINISTRATION: Start at 10 mg."


# --- cascade must equal "both premises scored" ---


def context_only_rule(p, h):
    """Entailed only when the premise has two sentences (the neighbour is needed)."""
    return (0.95, 0.0) if p.count(".") >= 2 else (0.05, 0.05)


def span_only_rule(p, h):
    return (0.95, 0.0) if p.count(".") == 1 else (0.05, 0.05)


@pytest.mark.parametrize("rule", [context_only_rule, span_only_rule, lambda p, h: (0.1, 0.9)])
async def test_cascade_gives_the_same_verdict_as_both(rule):
    s = sent("Take with food.", ("c1",))
    both = await run_one(verifier(FakeNLI(rule), context="both"), s)
    cascade = await run_one(verifier(FakeNLI(rule), context="cascade"), s)
    assert cascade.verdict == both.verdict
    assert cascade.reason == both.reason


async def test_cascade_runs_one_batch_when_the_first_premise_entails():
    nli = FakeNLI(lambda p, h: (0.95, 0.0))
    out = await run_one(verifier(nli, context="cascade"), sent("Take with food."))
    assert out.verdict == "supported"
    assert len(nli.batches) == 1


async def test_cascade_rescues_with_the_second_premise_in_a_second_batch():
    # the neighbour premise is long; only the span alone is entailing
    nli = FakeNLI(span_only_rule)
    out = await run_one(verifier(nli, context="cascade"), sent("Take with food."))
    assert out.verdict == "supported"
    assert len(nli.batches) == 2
    assert all(p.count(".") == 1 for p, _ in nli.batches[1])  # second stage = span alone


# --- prepare, ordering, concurrency (spec "Latency design") ---


async def test_prepare_embeds_all_spans_once_and_verify_reuses_it():
    emb = SpyEmbedder()
    v = verifier(FakeNLI(lambda p, h: (0.99, 0.0)), embedder=emb)
    out = await v.verify(
        [sent("Take with food."), sent("Avoid renal impairment.", ("c2",)), sent("Store dry.")],
        EVIDENCE,
        [],
    )
    assert len(out) == 3
    span_embeddings = [c for c in emb.calls if len(c) > 1]
    assert len(span_embeddings) == 1  # one prepare for the whole evidence set
    assert len(span_embeddings[0]) == 4
    single = [c for c in emb.calls if len(c) == 1]
    assert len(single) == 3  # then one tiny embedding per sentence


async def test_verify_keeps_order_and_mixes_verdicts():
    nli = FakeNLI(lambda p, h: (0.9, 0.0) if "food" in h else (0.1, 0.1))
    out = await verifier(nli).verify(
        [sent("Take with food."), sent("Use 99 mg."), sent("Take at night.")], EVIDENCE, []
    )
    assert [o.text for o in out] == ["Take with food.", "Use 99 mg.", "Take at night."]
    assert [o.verdict for o in out] == ["supported", "unsupported", "unsupported"]
    assert out[1].reason == REASON_NUMERIC


async def test_verify_does_not_mutate_the_input_sentences():
    s = sent("Take with food.")
    await verifier(FakeNLI(lambda p, h: (0.9, 0.0))).verify([s], EVIDENCE, [])
    assert s.verdict == "unverified" and s.score is None


async def test_model_work_runs_off_the_event_loop_thread():
    nli = FakeNLI(lambda p, h: (0.9, 0.0))
    await run_one(verifier(nli), sent("Take with food."))
    assert nli.threads and threading.get_ident() not in nli.threads


async def test_event_loop_stays_responsive_during_verification():
    nli = FakeNLI(lambda p, h: (0.9, 0.0), delay_s=0.3)
    v = verifier(nli)
    prepared = await v.prepare(EVIDENCE, [])
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    await v.verify_sentence(sent("Take with food."), prepared)
    t.cancel()
    assert ticks >= 10  # the loop kept running while the 0.3 s model call was in flight


async def test_concurrent_verify_sentence_is_safe_and_gives_sequential_results():
    # The scorer below fails if two batches ever overlap, like a tokenizer that is not thread
    # safe. The NLI wrapper must serialise calls; the fake here is the raw model, so the
    # CrossEncoderNLI lock is what keeps it safe.
    class Raw:
        def __init__(self):
            self.active = 0
            self.overlap = False

        def predict(self, pairs, **kw):
            self.active += 1
            if self.active > 1:
                self.overlap = True
            time.sleep(0.01)
            self.active -= 1
            return np.array(
                [[-3.0, 4.0, 0.0] if "food" in h else [0.0, -4.0, 2.0] for _, h in pairs]
            )

    raw = Raw()
    model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        ),
        predict=raw.predict,
    )
    v = verifier(CrossEncoderNLI("fake", model=model), context="neighbour")
    prepared = await v.prepare(EVIDENCE, [])
    texts = [f"Take {i} with food." if i % 2 else f"Take {i} at night." for i in range(24)]
    # numbers without units pass the number check
    sentences = [sent(t) for t in texts]
    seq = [await v.verify_sentence(s, prepared) for s in sentences]
    conc = await asyncio.gather(*(v.verify_sentence(s, prepared) for s in sentences))
    assert not raw.overlap
    assert [c.verdict for c in conc] == [s.verdict for s in seq]
    assert [c.span for c in conc] == [s.span for s in seq]
    assert {c.verdict for c in conc} == {"supported", "unsupported"}


# --- NLI label mapping (spec step 3: read the order from the model config) ---


def test_label_order_is_read_from_the_config_not_assumed():
    assert resolve_label_indices({0: "contradiction", 1: "entailment", 2: "neutral"}) == (1, 0)
    assert resolve_label_indices({0: "entailment", 1: "neutral", 2: "contradiction"}) == (0, 2)
    assert resolve_label_indices({"0": "ENTAILMENT", "1": "Neutral", "2": "Contradiction"}) == (
        0,
        2,
    )


@pytest.mark.parametrize(
    "labels",
    [{0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, {0: "positive", 1: "negative"}, {}],
)
def test_unknown_label_names_are_an_error_not_a_guess(labels):
    with pytest.raises(ValueError, match="entailment"):
        resolve_label_indices(labels)


def fake_cross_encoder(id2label, logits):
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(id2label=id2label)),
        predict=lambda pairs, **kw: np.array([logits for _ in pairs]),
    )


def test_cross_encoder_maps_probabilities_by_label_name_contradiction_first():
    m = fake_cross_encoder({0: "contradiction", 1: "entailment", 2: "neutral"}, [-5.0, 5.0, 0.0])
    entail, contra = CrossEncoderNLI("m", model=m).score([("p", "h")])
    assert entail[0] > 0.99 and contra[0] < 0.01


def test_cross_encoder_maps_probabilities_by_label_name_entailment_first():
    # same logits, different label order: the probabilities swap roles
    m = fake_cross_encoder({0: "entailment", 1: "neutral", 2: "contradiction"}, [-5.0, 5.0, 0.0])
    entail, contra = CrossEncoderNLI("m", model=m).score([("p", "h")])
    assert entail[0] < 0.01 and contra[0] < 0.01
    m2 = fake_cross_encoder({0: "entailment", 1: "neutral", 2: "contradiction"}, [5.0, 0.0, -5.0])
    entail, contra = CrossEncoderNLI("m", model=m2).score([("p", "h")])
    assert entail[0] > 0.99 and contra[0] < 0.01


def test_cross_encoder_scores_one_batch_of_the_whole_list():
    calls = []

    def predict(pairs, **kw):
        calls.append((len(pairs), kw.get("batch_size")))
        return np.zeros((len(pairs), 3))

    m = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        ),
        predict=predict,
    )
    CrossEncoderNLI("m", model=m).score([("a", "b")] * 5)
    assert calls == [(5, 5)]


def test_cross_encoder_empty_input():
    m = fake_cross_encoder({0: "contradiction", 1: "entailment", 2: "neutral"}, [0, 0, 0])
    e, c = CrossEncoderNLI("m", model=m).score([])
    assert e.size == 0 and c.size == 0


def test_probabilities_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [100.0, 0.0, -100.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert not np.isnan(p).any()

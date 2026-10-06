import uuid

import pytest

from pharmarag.graph.state import Hit
from pharmarag.retrieval.fusion import fuse

SETID = uuid.UUID(int=1)


def hit(chunk_id, source, rank, *, score=0.1, attempt=1, key="1", setid=SETID):
    return Hit(
        chunk_id=chunk_id,
        setid=setid,
        spl_version=1,
        ingredient_key=key,
        section_loinc="34068-7",
        section_title="Dosage and administration",
        text=f"chunk {chunk_id}",
        spans=[],
        source=source,
        rank=rank,
        score=score,
        retrieval_attempt=attempt,
    )


def test_rrf_math_hand_computed(thresholds):
    # dense: A(1) B(2) C(3); lexical: B(1) D(2) A(3); k = 60
    A, B, C, D = 10, 20, 30, 40
    hits = [
        hit(A, "dense", 1),
        hit(B, "dense", 2),
        hit(C, "dense", 3),
        hit(B, "lexical", 1),
        hit(D, "lexical", 2),
        hit(A, "lexical", 3),
    ]
    evidence, _ = fuse(hits, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert [c.chunk_id for c in evidence] == [B, A, D, C]
    assert [c.id for c in evidence] == ["c1", "c2", "c3", "c4"]
    scores = {c.chunk_id: c.fused_score for c in evidence}
    assert scores[B] == pytest.approx(1 / 62 + 1 / 61)  # 0.032522
    assert scores[A] == pytest.approx(1 / 61 + 1 / 63)  # 0.032266
    assert scores[D] == pytest.approx(1 / 62)
    assert scores[C] == pytest.approx(1 / 63)


def test_rrf_uses_configured_k(thresholds):
    t = thresholds.retrieval.model_copy(update={"rrf_k": 10})
    evidence, _ = fuse([hit(1, "dense", 1)], attempt=1, ingredient_keys=[], thresholds=t)
    assert evidence[0].fused_score == pytest.approx(1 / 11)


def test_stale_attempt_hits_are_ignored(thresholds):
    hits = [
        hit(1, "dense", 1, attempt=1, score=0.9),
        hit(2, "dense", 1, attempt=2, score=0.5),
    ]
    evidence, ok = fuse(hits, attempt=2, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert [c.chunk_id for c in evidence] == [2]
    assert ok  # 0.5 >= 0.45; the stale 0.9 is irrelevant
    evidence, ok = fuse(hits, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert [c.chunk_id for c in evidence] == [1]


def test_stale_hits_cannot_make_evidence_ok(thresholds):
    hits = [hit(1, "dense", 1, attempt=1, score=0.95), hit(2, "dense", 1, attempt=2, score=0.1)]
    evidence, ok = fuse(hits, attempt=2, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert [c.chunk_id for c in evidence] == [2]
    assert not ok


def test_evidence_ok_via_dense_score(thresholds):
    _, ok = fuse(
        [hit(1, "dense", 1, score=0.45)],
        attempt=1,
        ingredient_keys=[],
        thresholds=thresholds.retrieval,
    )
    assert ok
    _, ok = fuse(
        [hit(1, "dense", 1, score=0.449)],
        attempt=1,
        ingredient_keys=[],
        thresholds=thresholds.retrieval,
    )
    assert not ok


def test_evidence_ok_via_two_chunks_in_both_lists(thresholds):
    both = [
        hit(1, "dense", 1, score=0.1),
        hit(2, "dense", 2, score=0.1),
        hit(1, "lexical", 1),
        hit(2, "lexical", 2),
    ]
    _, ok = fuse(both, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert ok
    one = [hit(1, "dense", 1, score=0.1), hit(1, "lexical", 1), hit(2, "dense", 2, score=0.1)]
    _, ok = fuse(one, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert not ok


def test_evidence_not_ok_when_empty(thresholds):
    evidence, ok = fuse([], attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert evidence == [] and not ok


def test_single_list_without_strong_dense_score_is_not_ok(thresholds):
    hits = [hit(i, "lexical", i) for i in range(1, 4)]
    _, ok = fuse(hits, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert not ok


def test_top_k_cap(thresholds):
    hits = [hit(i, "dense", i, score=0.9) for i in range(1, 21)]
    evidence, _ = fuse(hits, attempt=1, ingredient_keys=["1"], thresholds=thresholds.retrieval)
    assert [c.chunk_id for c in evidence] == list(range(1, 9))


def test_per_drug_top_k_guarantees_both_drugs(thresholds):
    # Drug "1" owns the top 7 dense ranks, drug "2" ranks 8-10. A global top-8 would starve "2".
    hits = [hit(i, "dense", i, score=0.9, key="1") for i in range(1, 8)]
    hits += [hit(100 + i, "dense", 7 + i, score=0.8, key="2") for i in range(1, 4)]
    global_ev, _ = fuse(hits, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert sum(c.ingredient_key == "2" for c in global_ev) == 1
    evidence, ok = fuse(
        hits, attempt=1, ingredient_keys=["1", "2"], thresholds=thresholds.retrieval
    )
    keys = [c.ingredient_key for c in evidence]
    assert keys.count("1") == 5 and keys.count("2") == 3
    assert keys == ["1"] * 5 + ["2"] * 3  # grouped in ingredient_keys order
    assert [c.id for c in evidence] == [f"c{i}" for i in range(1, 9)]
    assert ok


def test_multi_drug_evidence_not_ok_when_one_drug_has_no_hits(thresholds):
    strong_a = [hit(i, "dense", i, score=0.9, key="1") for i in range(1, 4)]
    evidence, ok = fuse(
        strong_a, attempt=1, ingredient_keys=["1", "2"], thresholds=thresholds.retrieval
    )
    assert {c.ingredient_key for c in evidence} == {"1"}
    assert not ok  # drug "2" retrieved nothing, drug "1" cannot vouch for it
    _, ok = fuse(strong_a, attempt=1, ingredient_keys=["1"], thresholds=thresholds.retrieval)
    assert ok


def test_multi_drug_each_drug_must_pass_the_score_or_overlap_rule_itself(thresholds):
    a = [hit(i, "dense", i, score=0.9, key="1") for i in range(1, 3)]
    weak_b = [hit(100, "dense", 1, score=0.2, key="2")]
    keys = ["1", "2"]
    _, ok = fuse(a + weak_b, attempt=1, ingredient_keys=keys, thresholds=thresholds.retrieval)
    assert not ok  # B has a chunk but a weak dense score and no overlap
    overlap_b = [
        hit(100, "dense", 1, score=0.2, key="2"),
        hit(101, "dense", 2, score=0.2, key="2"),
        hit(100, "lexical", 1, key="2"),
        hit(101, "lexical", 2, key="2"),
    ]
    _, ok = fuse(a + overlap_b, attempt=1, ingredient_keys=keys, thresholds=thresholds.retrieval)
    assert ok  # B passes through the overlap rule
    strong_b = [hit(100, "dense", 1, score=0.6, key="2")]
    _, ok = fuse(a + strong_b, attempt=1, ingredient_keys=keys, thresholds=thresholds.retrieval)
    assert ok


def test_per_drug_cap_is_five(thresholds):
    hits = [hit(i, "dense", i, key="2") for i in range(1, 9)]
    evidence, _ = fuse(hits, attempt=1, ingredient_keys=["1", "2"], thresholds=thresholds.retrieval)
    assert len(evidence) == 5


def test_product_hint_breaks_ties_only(thresholds):
    hint_setid = uuid.UUID(int=2)
    # Chunks 1 and 2 tie exactly (dense rank 1 vs lexical rank 1). 2 belongs to the hint label.
    hits = [
        hit(1, "dense", 1),
        hit(2, "lexical", 1, setid=hint_setid),
        hit(3, "dense", 2, setid=hint_setid),
    ]
    plain, _ = fuse(hits, attempt=1, ingredient_keys=[], thresholds=thresholds.retrieval)
    assert [c.chunk_id for c in plain][:2] == [1, 2]
    hinted, _ = fuse(
        hits,
        attempt=1,
        ingredient_keys=[],
        thresholds=thresholds.retrieval,
        preferred_setids={hint_setid},
    )
    assert [c.chunk_id for c in hinted][:2] == [2, 1]
    # The hint never overrides a real score difference: chunk 3 (rank 2) stays behind chunk 1.
    assert [c.chunk_id for c in hinted].index(3) == 2

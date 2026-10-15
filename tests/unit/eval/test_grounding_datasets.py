"""The frozen grounding datasets (spec section 11.2): pinned by SHA, well formed, and honest.

If a SHA test fails, a frozen file changed. Test claims and questions must never change once a
test-set run exists; the dev file is pinned because the proposed thresholds were tuned on it.
"""

import json
from collections import Counter

import pytest

from pharmarag.eval import grounding_data as gd
from pharmarag.eval import grounding_dev
from pharmarag.grounding.numeric import missing_quantities

TEST_CLAIMS_SHA = "d9780f73f936220f2710667781f83710f4ab1fd1ca6612d03c2c5b3c28f3d28a"
DEV_CLAIMS_SHA = "2cd8d438c8232c227576c814a24cd506d170dfa0c7b639cee6b29270dba14125"
QUESTIONS_SHA = "4de7db84d8a49c6b8511e03db0afd8a4af728a2a5aac97cfa7f72cec81795105"


def test_frozen_test_claims_are_unchanged():
    assert gd.sha256_file(gd.TEST_CLAIMS) == TEST_CLAIMS_SHA


def test_dev_claims_are_unchanged():
    assert gd.sha256_file(gd.DEV_CLAIMS) == DEV_CLAIMS_SHA


def test_questions_are_unchanged():
    assert gd.sha256_file(gd.QUESTIONS) == QUESTIONS_SHA


# --- questions ---


def test_fourteen_questions_nine_answerable_five_not():
    qs = gd.load_questions()
    assert len(qs) == 14
    assert sum(q.answerable for q in qs) == 9
    assert sum(not q.answerable for q in qs) == 5
    assert len({q.id for q in qs}) == 14


# --- test claims ---


@pytest.fixture(scope="module")
def test_claims():
    return gd.load_claims(gd.TEST_CLAIMS)


def test_forty_natural_claims_with_no_labels_in_the_file(test_claims):
    assert len(test_claims) == 40
    assert len({c.id for c in test_claims}) == 40
    # human labels live in their own file; none may be baked in
    assert all(c.label is None for c in test_claims)
    assert all(c.kind == "natural" for c in test_claims)


def test_claims_come_from_the_fixed_questions_and_the_pinned_model(test_claims):
    ids = {q.id: q for q in gd.load_questions()}
    for c in test_claims:
        assert c.meta["question_id"] in ids
        assert c.question == ids[c.meta["question_id"]].question
        assert c.meta["llm_model_id"] == "claude-sonnet-5-5"


def test_every_citation_resolves_to_evidence_the_writer_saw(test_claims):
    for c in test_claims:
        known = {e.id for e in c.evidence} | {f.id for f in c.facts}
        assert c.citations and set(c.citations) <= known, c.id


def test_evidence_spans_slice_to_text(test_claims):
    for c in test_claims:
        for e in c.evidence:
            for sp in e.spans:
                assert e.text[sp.start : sp.end].strip(), (c.id, e.id, sp)


def test_claims_cover_several_questions_and_none_dominates(test_claims):
    counts = Counter(c.meta["question_id"] for c in test_claims)
    assert len(counts) >= 8
    assert max(counts.values()) <= 6


# --- dev claims ---


@pytest.fixture(scope="module")
def dev_claims():
    return gd.load_claims(gd.DEV_CLAIMS)


def test_dev_has_at_least_sixty_claims_with_by_construction_labels(dev_claims):
    assert len(dev_claims) >= 60
    assert {c.label for c in dev_claims} == {"S", "H"}
    assert len({c.id for c in dev_claims}) == len(dev_claims)
    kinds = Counter(c.kind for c in dev_claims)
    for needed in (
        "verbatim",
        "rewrite",
        "numeric",
        "negation",
        "other_drug",
        "frequency",
        "population",
    ):
        assert kinds[needed] >= 5, needed
    # the harder supported kinds that mimic real writer output (review round 5)
    for harder in ("tallman", "combine", "row_restate", "heading", "join"):
        assert kinds[harder] >= 6, harder
    assert kinds["row_numeric"] >= 6


def test_tallman_claims_use_a_plain_name_over_a_tallman_label(dev_claims):
    for c in dev_claims:
        if c.kind == "tallman":
            assert c.meta["tallman"] in " ".join(e.text for e in c.evidence), c.id
            assert c.meta["tallman"] not in c.sentence, c.id  # the claim is typed plainly


def test_row_claims_restate_a_real_table_row(dev_claims):
    for c in dev_claims:
        if c.kind == "row_restate":
            row = c.meta["source_sentence"]
            assert "; " in row and ": " in row, c.id
            assert missing_quantities(c.sentence, [row]) == [], c.id  # supported by that row
        if c.kind == "row_numeric":
            assert missing_quantities(c.sentence, [e.text for e in c.evidence]), c.id


def test_dev_citations_resolve(dev_claims):
    for c in dev_claims:
        assert set(c.citations) <= {e.id for e in c.evidence}, c.id


def test_dev_label_matches_kind(dev_claims):
    for c in dev_claims:
        assert c.label == grounding_dev.LABEL_OF[c.kind]


def test_no_hallucinated_dev_claim_is_stated_verbatim_in_its_evidence(dev_claims):
    for c in dev_claims:
        if c.label == "H":
            texts = " ".join(" ".join(e.text.split()) for e in c.evidence)
            assert " ".join(c.sentence.split()) not in texts, c.id


def test_numeric_hallucinations_really_have_a_number_the_evidence_lacks(dev_claims):
    nums = [c for c in dev_claims if c.kind == "numeric"]
    assert nums
    for c in nums:
        assert missing_quantities(c.sentence, [e.text for e in c.evidence]), c.id


def test_perturbed_claims_record_their_source_sentence(dev_claims):
    for c in dev_claims:
        if c.kind in ("numeric", "negation", "frequency", "population", "overclaim"):
            assert c.meta["source_sentence"] != c.sentence, c.id


def test_other_drug_claims_come_from_a_different_label(dev_claims):
    moved = [c for c in dev_claims if c.kind == "other_drug"]
    assert moved
    for c in moved:
        assert c.meta["moved_from"] != c.meta["drug"], c.id


def test_dev_and_test_claims_do_not_overlap(dev_claims, test_claims):
    assert not ({c.sentence for c in dev_claims} & {c.sentence for c in test_claims})


def test_jsonl_files_are_valid_json_per_line():
    for path in (gd.DEV_CLAIMS, gd.TEST_CLAIMS, gd.QUESTIONS):
        for line in path.read_text().splitlines():
            json.loads(line)

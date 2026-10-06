"""Shape and honesty checks on the look-alike datasets (spec section 11.1).

The test file is frozen: the hash below must change only on purpose, in its own commit, and
never after a thresholds change is tuned against it.
"""

import csv
import json
from collections import Counter

import pytest
from rapidfuzz.distance import Levenshtein

from pharmarag.app.settings import REPO_ROOT
from pharmarag.eval.lookalike import KINDS, Row, dataset_path, load_dataset
from pharmarag.eval.report import sha256_of

# Frozen once written (cycle 2). Never tune on this file.
TEST_SET_SHA256 = "71f96645e017c5739ede6dfd1d16498c7501d4f24f1d885a34f63e47352351f0"

FIELDS = {
    "question",
    "mention",
    "gold_ingredient_key",
    "gold_name",
    "partner_name",
    "kind",
}


def seed_pairs() -> set[frozenset[str]]:
    groups: dict[str, set[str]] = {}
    with (REPO_ROOT / "data" / "seed_ingredients.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lookalike_group"].strip():
                groups.setdefault(row["lookalike_group"].strip(), set()).add(row["name"].strip())
    return {frozenset(g) for g in groups.values()}


@pytest.fixture(params=["dev", "test"])
def split(request):
    return request.param


def test_the_test_set_is_frozen():
    assert sha256_of(dataset_path("test")) == TEST_SET_SHA256


def test_sizes_and_mix():
    dev, test = load_dataset("dev"), load_dataset("test")
    assert (len(dev), len(test)) == (20, 15)
    assert Counter(r.kind for r in test) == {"misspelling": 8, "brand": 4, "correct_partner": 3}
    # The dev set uses a similar mix.
    mix = Counter(r.kind for r in dev)
    assert set(mix) == set(KINDS)
    assert mix["misspelling"] >= 8 and mix["brand"] >= 4 and mix["correct_partner"] >= 3


def test_every_row_has_exactly_the_documented_fields(split):
    for line in dataset_path(split).read_text().splitlines():
        assert set(json.loads(line)) == FIELDS


def test_the_mention_is_in_the_question_exactly_as_typed(split):
    for r in load_dataset(split):
        assert r.mention in r.question


def test_dev_and_test_share_no_question_and_no_mention():
    dev, test = load_dataset("dev"), load_dataset("test")
    assert not {r.question for r in dev} & {r.question for r in test}
    assert not {r.mention.lower() for r in dev} & {r.mention.lower() for r in test}


def test_questions_are_unique_within_a_split(split):
    questions = [r.question for r in load_dataset(split)]
    assert len(set(questions)) == len(questions)


def test_misspellings_are_real_misspellings_of_the_gold_drug_not_of_the_partner(split):
    for r in (r for r in load_dataset(split) if r.kind == "misspelling"):
        m = r.mention.lower()
        assert m != r.gold_name and m != r.partner_name, r.mention
        d_gold = Levenshtein.distance(m, r.gold_name)
        d_partner = Levenshtein.distance(m, r.partner_name)
        assert 1 <= d_gold <= 2, f"{r.mention}: {d_gold} edits from {r.gold_name}"
        assert d_gold < d_partner, f"{r.mention} is as close to the partner as to the gold drug"


def test_correct_partner_rows_spell_the_drug_correctly(split):
    for r in (r for r in load_dataset(split) if r.kind == "correct_partner"):
        assert r.mention == r.gold_name


def test_brand_rows_use_a_brand_not_a_generic_name(split):
    for r in (r for r in load_dataset(split) if r.kind == "brand"):
        assert r.mention.lower() not in (r.gold_name, r.partner_name)


def test_misspelling_and_correct_partner_rows_come_from_the_seed_lookalike_pairs(split):
    pairs = seed_pairs()
    assert len(pairs) == 20
    for r in (r for r in load_dataset(split) if r.kind != "brand"):
        assert frozenset({r.gold_name, r.partner_name}) in pairs, (r.gold_name, r.partner_name)


def test_rows_are_typed_rows(split):
    assert all(isinstance(r, Row) for r in load_dataset(split))


@pytest.mark.corpus
async def test_gold_keys_come_from_the_corpus_and_mentions_are_not_aliases_of_the_partner(
    corpus_conn, split
):
    names = {
        n: k
        for k, n in await (
            await corpus_conn.execute("SELECT rxcui, name FROM ingredient")
        ).fetchall()
    }
    aliases = {
        (a, k)
        for a, k in await (
            await corpus_conn.execute("SELECT alias, ingredient_key FROM drug_alias")
        ).fetchall()
    }
    for r in load_dataset(split):
        assert names[r.gold_name] == r.gold_ingredient_key
        cur = await corpus_conn.execute(
            "SELECT 1 FROM label WHERE ingredient_key = %s", (r.gold_ingredient_key,)
        )
        assert await cur.fetchone(), f"{r.gold_name} has no label in the corpus"
        partner_key = names.get(r.partner_name)
        assert partner_key is not None and partner_key != r.gold_ingredient_key
        if r.kind == "brand":
            # A brand row names a brand the corpus knows for the gold drug, typed exactly or
            # with one typo (a look-alike brand typo such as "Lamictel" is a realistic error).
            gold_brands = {a for a, k in aliases if k == r.gold_ingredient_key}
            m = r.mention.lower()
            assert any(Levenshtein.distance(m, b) <= 1 for b in gold_brands), r.mention
            assert (m, partner_key) not in aliases
        if r.kind == "misspelling":
            # Not a name RxNorm knows for either drug as far as the alias table can tell.
            assert (r.mention.lower(), r.gold_ingredient_key) not in aliases
            assert (r.mention.lower(), partner_key) not in aliases

"""The committed CI fixture (data/ci_fixture.sql.gz) holds what the CI gates need, and is small.

Read straight from the gzip: no database. The restore itself is exercised by
scripts/load_ci_fixture.sh (see the workflow); this test guards the content and the size.
"""

import gzip
import json

import pytest

from pharmarag.app.settings import REPO_ROOT
from pharmarag.eval import lookalike as la

FIXTURE = REPO_ROOT / "data" / "ci_fixture.sql.gz"
LATENCY_QUESTIONS = REPO_ROOT / "pharmarag" / "eval" / "datasets" / "latency_questions.jsonl"


def parse(path):
    """{table: (columns, rows)} from the COPY blocks, plus the text outside them."""
    tables, other, current = {}, [], None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if current is None and line.startswith("COPY "):
                head = line[len("COPY ") :]
                name, cols = head.split(" (", 1)
                current = (name, cols.split(")")[0].split(", "), [])
            elif current is not None and line == "\\.":
                tables[current[0]] = (current[1], current[2])
                current = None
            elif current is not None:
                current[2].append(line.split("\t"))
            else:
                other.append(line)
    return tables, other


@pytest.fixture(scope="module")
def fixture():
    return parse(FIXTURE)


def test_size_is_within_the_15_mb_budget():
    assert FIXTURE.stat().st_size <= 15 * 1024 * 1024


def test_has_the_tables_and_is_data_only(fixture):
    tables, other = fixture
    assert set(tables) == {"ingredient", "drug_alias", "label", "chunk"}
    text = "\n".join(other)
    assert "CREATE TABLE" not in text  # the schema comes from alembic
    assert text.count("BEGIN;") == 1 and text.count("COMMIT;") == 1
    assert "INSERT INTO corpus_meta" in text and "+ci'" in text  # a subset says so
    assert "setval(pg_get_serial_sequence('chunk', 'id')" in text


def test_every_row_has_one_value_per_column(fixture):
    tables, _ = fixture
    for name, (cols, rows) in tables.items():
        assert rows, name
        assert all(len(r) == len(cols) for r in rows), name


def test_alias_and_ingredient_tables_are_complete(fixture):
    tables, _ = fixture
    assert len(tables["drug_alias"][1]) >= 1000 and len(tables["ingredient"][1]) >= 200


def keys_with_labels(fixture) -> set[str]:
    cols, rows = fixture[0]["label"]
    i = cols.index("ingredient_key")
    return {r[i] for r in rows}


def test_lookalike_gold_drugs_and_partners_have_labels(fixture):
    have = keys_with_labels(fixture)
    ing_cols, ing_rows = fixture[0]["ingredient"]
    by_name = {r[1].lower(): r[0] for r in ing_rows}
    for row in la.load_dataset("test"):
        assert row.gold_ingredient_key in have, row.gold_name
        assert by_name[row.partner_name.lower()] in have, row.partner_name


def test_latency_drugs_have_labels_and_all_their_chunks(fixture):
    tables, _ = fixture
    alias_cols, alias_rows = tables["drug_alias"]
    key_of = {r[0]: r[3] for r in alias_rows}
    have = keys_with_labels(fixture)
    for line in LATENCY_QUESTIONS.read_text().splitlines():
        q = json.loads(line)
        assert key_of[q["drug"].lower()] in have, q["drug"]
    # Chunks belong to exactly the labels present: whole labels or nothing.
    c_cols, c_rows = tables["chunk"]
    assert {r[c_cols.index("ingredient_key")] for r in c_rows} <= have


def test_chunk_ids_are_unique(fixture):
    cols, rows = fixture[0]["chunk"]
    ids = [r[cols.index("id")] for r in rows]
    assert len(ids) == len(set(ids))

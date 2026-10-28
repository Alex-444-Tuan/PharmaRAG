"""Write the CI corpus fixture: a compressed, data-only SQL dump of the labels the CI evals need.

    uv run python scripts/export_ci_fixture.py                       # data/ci_fixture.sql.gz
    uv run python scripts/export_ci_fixture.py --latency-questions 10 --out /tmp/small.sql.gz

CI has no DailyMed, no RxNav and no ingested corpus (spec section 13). The three
gates need only a small slice of it:

* look-alike (test split): the gold drug of each row, its look-alike partner, and every
  ingredient the resolver's trigram step returns for the row's mention. The whole `drug_alias`
  and `ingredient` tables are copied (about 1,300 and 200 rows) so alias lookups, trigram
  neighbours and display names behave exactly as on the full corpus.
* grounding: the 40 frozen claims carry their own evidence, so they need no labels. The score
  run reads only `corpus_meta`.
* latency replay: the label of each drug in the first `--latency-questions` questions of
  `latency_questions.jsonl`, with its chunks. Retrieval for these questions is filtered by
  `ingredient_key`, so a chunk's rank depends only on the chunks of the same drug, and the
  subset reproduces the full corpus' results. Chunk ids are kept, because the recordings cite
  evidence by position and the reports link to chunk ids.

Only whole labels are copied (a label with all its chunks, ids unchanged), never part of one.
`corpus_meta.corpus_version` becomes `<source version>+ci`: the fixture is a subset, and a report
produced from it must not claim the full corpus' version.

Restore with scripts/load_ci_fixture.sh. The file is plain SQL (COPY blocks), so psql reads it.
The source database is only read.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pharmarag.app.settings import load_thresholds  # noqa: E402
from pharmarag.db import create_pool  # noqa: E402
from pharmarag.eval import lookalike as la  # noqa: E402
from pharmarag.resolver.local import PgLocalStore  # noqa: E402
from pharmarag.resolver.normalize import collapse, normalize  # noqa: E402

DEFAULT_SOURCE = os.environ.get(
    "CORPUS_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pharmarag_dev"
)
DEFAULT_OUT = REPO_ROOT / "data" / "ci_fixture.sql.gz"
LATENCY_QUESTIONS = REPO_ROOT / "pharmarag" / "eval" / "datasets" / "latency_questions.jsonl"
MAX_BYTES = 15 * 1024 * 1024

LABEL_COLS = (
    "setid, spl_version, title, labeler, marketing_category, effective_date, "
    "ingredient_key, ingredient_rxcuis, product_rxcuis"
)
CHUNK_COLS = (
    "id, setid, spl_version, ingredient_key, section_loinc, section_title, subsection_title, "
    "ordinal, text, spans, embedding"
)


async def select_keys(source: str, latency_questions: int) -> tuple[set[str], dict[str, list[str]]]:
    """The ingredient keys whose labels go into the fixture, and why (for the log)."""
    thresholds = load_thresholds()
    pool = await create_pool(source, min_size=1, max_size=2)
    why: dict[str, list[str]] = {}

    def add(key: str, reason: str) -> None:
        why.setdefault(key, []).append(reason)

    try:
        store = PgLocalStore(pool)
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT rxcui, name FROM ingredient")
            by_name = {name.lower(): rxcui for rxcui, name in await cur.fetchall()}

        for row in la.load_dataset("test"):
            add(row.gold_ingredient_key, f"lookalike gold: {row.gold_name}")
            partner = by_name.get(row.partner_name.lower())
            if partner is None:
                raise SystemExit(f"partner {row.partner_name!r} is not in the ingredient table")
            add(partner, f"lookalike partner of {row.gold_name}")
            norm = normalize(row.mention)
            near = await store.trigram(
                norm,
                min_similarity=thresholds.resolver.trigram_min,
                k=thresholds.resolver.trigram_k,
            )
            for r in near:
                add(r.ingredient_key, f"trigram neighbour of {row.mention!r}")
            for r in await store.exact({norm, collapse(row.mention)}):
                add(r.ingredient_key, f"exact alias of {row.mention!r}")

        lines = [json.loads(x) for x in LATENCY_QUESTIONS.read_text().splitlines() if x.strip()]
        for q in lines[:latency_questions]:
            rows = await store.exact({normalize(q["drug"]), collapse(q["drug"])})
            if not rows:
                raise SystemExit(f"latency drug {q['drug']!r} is not in the alias table")
            for r in rows:
                add(r.ingredient_key, f"latency {q['id']}: {q['drug']}")
    finally:
        await pool.close()
    return set(why), why


def copy_block(w, conn: psycopg.Connection, table: str, cols: str, select: str, params=()) -> int:
    """Write one `COPY ... FROM stdin` block holding the rows of `select`; return the row count."""
    w.write(f"COPY {table} ({cols}) FROM stdin;\n".encode())
    with conn.cursor() as cur:
        with cur.copy(f"COPY ({select}) TO STDOUT", params) as copy:
            for data in copy:
                w.write(bytes(data))
        n = cur.rowcount
    w.write(b"\\.\n\n")
    return n


def export(source: str, keys: set[str], out: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    ordered = sorted(keys)
    out.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and no stored file name: the same corpus gives the same bytes, so a re-export that
    # changes nothing shows no diff in git.
    with (
        psycopg.connect(source) as conn,
        out.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as w,
    ):
        conn.read_only = True
        version = conn.execute("SELECT corpus_version FROM corpus_meta").fetchone()[0]
        header = (
            "-- PharmaRAG CI fixture: a subset of the dev corpus, data only (schema comes from\n"
            "-- `alembic upgrade head`). Written by scripts/export_ci_fixture.py.\n"
            f"-- source corpus_version: {version}; ingredient keys with labels: {len(keys)}\n"
            "BEGIN;\n\n"
        )
        w.write(header.encode())
        counts["ingredient"] = copy_block(
            w,
            conn,
            "ingredient",
            "rxcui, name",
            "SELECT rxcui, name FROM ingredient ORDER BY rxcui",
        )
        counts["drug_alias"] = copy_block(
            w,
            conn,
            "drug_alias",
            "alias, alias_tty, source_rxcui, ingredient_key",
            "SELECT alias, alias_tty, source_rxcui, ingredient_key FROM drug_alias "
            "ORDER BY alias, source_rxcui",
        )
        counts["label"] = copy_block(
            w,
            conn,
            "label",
            LABEL_COLS,
            f"SELECT {LABEL_COLS} FROM label WHERE ingredient_key = ANY(%s) "
            "ORDER BY ingredient_key, setid, spl_version",
            (ordered,),
        )
        counts["chunk"] = copy_block(
            w,
            conn,
            "chunk",
            CHUNK_COLS,
            f"SELECT {CHUNK_COLS} FROM chunk WHERE ingredient_key = ANY(%s) ORDER BY id",
            (ordered,),
        )
        tail = (
            f"INSERT INTO corpus_meta (id, corpus_version) VALUES (1, '{version}+ci');\n"
            "SELECT setval(pg_get_serial_sequence('chunk', 'id'), "
            "coalesce((SELECT max(id) FROM chunk), 1));\n"
            "COMMIT;\n"
        )
        w.write(tail.encode())
        counts["corpus_meta"] = 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=DEFAULT_SOURCE, help="database to read (read-only)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--latency-questions",
        type=int,
        default=30,
        help="labels for the first N questions of latency_questions.jsonl",
    )
    args = p.parse_args(argv)

    keys, why = asyncio.run(select_keys(args.source, args.latency_questions))
    for key in sorted(why):
        print(f"{key:>10}  {'; '.join(why[key][:3])}")
    counts = export(args.source, keys, args.out)
    size = args.out.stat().st_size
    print(f"rows: {counts}")
    print(f"wrote {args.out}: {size / 1024 / 1024:.2f} MB (limit {MAX_BYTES / 1024 / 1024:.0f} MB)")
    if size > MAX_BYTES:
        print("TOO LARGE: lower --latency-questions or drop labels (see the module docstring)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

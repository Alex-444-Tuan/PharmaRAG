"""Ingestion CLI.

    uv run python -m pharmarag.ingest.run --seed data/seed_ingredients.csv [--limit N]

Reads the seed list, picks one reference label per ingredient, and loads labels, chunks,
ingredients and aliases into the database named by DATABASE_URL. Safe to re-run: labels already
loaded at the same version are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import time
from collections import Counter
from pathlib import Path

from pharmarag.app.settings import REPO_ROOT, get_settings
from pharmarag.db import connect
from pharmarag.ingest.dailymed import DailyMedClient, load_rxnorm_mappings
from pharmarag.ingest.load import refresh_corpus_version
from pharmarag.ingest.pins import Pins
from pharmarag.ingest.pipeline import Ingestor
from pharmarag.ingest.rxnav import RxNavClient
from pharmarag.models.embedder import SentenceTransformerEmbedder

log = logging.getLogger("pharmarag.ingest")

DEFAULT_MAPPINGS = REPO_ROOT / "data" / "cache" / "rxnorm_mappings.zip"
DEFAULT_PINS = REPO_ROOT / "data" / "label_pins.csv"
MAPPINGS_MAX_AGE_S = 7 * 24 * 3600
INGEST_RXNAV_TIMEOUT_S = 10.0


def read_seed(path: Path) -> list[tuple[str, str]]:
    """(name, lookalike_group) rows of the seed CSV. Blank names are ignored."""
    with path.open(newline="") as f:
        return [
            (row["name"].strip(), (row.get("lookalike_group") or "").strip())
            for row in csv.DictReader(f)
            if row.get("name", "").strip()
        ]


def _mappings_stale(path: Path) -> bool:
    return not path.exists() or time.time() - path.stat().st_mtime > MAPPINGS_MAX_AGE_S


async def ensure_mappings(dailymed: DailyMedClient, path: Path, refresh: bool) -> Path:
    if refresh or await asyncio.to_thread(_mappings_stale, path):
        log.info("downloading rxnorm_mappings.zip to %s", path)
        await dailymed.download_rxnorm_mappings(path)
    return path


def select_names(names: list[str], limit: int | None, only: list[str] | None) -> list[str]:
    """The seed names to ingest. `only` keeps just those names (case-insensitive) in seed order
    and fails on a name that is not in the seed, so a typo cannot silently skip an ingredient.
    `limit` then keeps the first N of what is left."""
    if only:
        wanted = {n.strip().lower() for n in only if n.strip()}
        unknown = wanted - {n.lower() for n in names}
        if unknown:
            raise ValueError(f"not in the seed list: {sorted(unknown)}")
        names = [n for n in names if n.lower() in wanted]
    return names[:limit] if limit is not None else names


async def run(
    seed: Path,
    limit: int | None,
    mappings_path: Path,
    refresh: bool,
    pins_path: Path = DEFAULT_PINS,
    repin: bool = False,
    only: list[str] | None = None,
) -> Counter[str]:
    settings = get_settings()
    started = time.monotonic()
    names = select_names([n for n, _ in read_seed(seed)], limit, only)

    dailymed = DailyMedClient(settings.dailymed_base_url)
    # SPEC: settings.rxnav_timeout_s (2 s) is the request-time budget. RxNav answers some bulk
    # lookups in 2 to 3 s, so ingestion allows at least INGEST_RXNAV_TIMEOUT_S.
    rxnav = RxNavClient(
        settings.rxnav_base_url, timeout_s=max(settings.rxnav_timeout_s, INGEST_RXNAV_TIMEOUT_S)
    )
    conn = await connect(settings.database_url)
    pins = Pins.load(pins_path)
    try:
        path = await ensure_mappings(dailymed, mappings_path, refresh)
        log.info("loading mapping index")
        mappings = await asyncio.to_thread(load_rxnorm_mappings, path)
        embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
        ingestor = Ingestor(
            rxnav=rxnav,
            dailymed=dailymed,
            mappings=mappings,
            embedder=embedder,
            conn=conn,
            labels_per_ingredient=settings.labels_per_ingredient,
            pins=pins,
            repin=repin,
        )
        tally: Counter[str] = Counter()
        for i, name in enumerate(names, 1):
            try:
                result = await ingestor.ingest_seed(name)
            except Exception:  # one bad ingredient must not stop the run
                log.exception("%s: failed", name)
                tally["failed"] += 1
                await conn.rollback()
                continue
            tally[result.status] += 1
            pins.save()  # after every seed, so an interrupted run keeps its picks
            log.info("[%d/%d] %s: %s", i, len(names), name, result.status)
        version = await refresh_corpus_version(conn, settings.embedding_model)
        counts = {}
        for table in ("ingredient", "drug_alias", "label", "chunk"):
            cur = await conn.execute(f"SELECT count(*) FROM {table}")  # noqa: S608 - fixed names
            counts[table] = (await cur.fetchone())[0]
        log.info("seed results: %s", dict(tally))
        log.info("table counts: %s", counts)
        log.info("corpus_version=%s wall=%.1fs", version, time.monotonic() - started)
        return tally
    finally:
        await conn.close()
        await rxnav.aclose()
        await dailymed.aclose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pharmarag.ingest.run", description=__doc__)
    parser.add_argument("--seed", type=Path, required=True, help="CSV with name,lookalike_group")
    parser.add_argument("--limit", type=int, default=None, help="only the first N seed rows")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated seed names to ingest (for re-pinning a few ingredients)",
    )
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--refresh-mappings", action="store_true")
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS, help="pinned labels CSV")
    parser.add_argument(
        "--repin", action="store_true", help="ignore the pins and select labels afresh"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(
        run(
            args.seed,
            args.limit,
            args.mappings,
            args.refresh_mappings,
            args.pins,
            args.repin,
            args.only.split(",") if args.only else None,
        )
    )


if __name__ == "__main__":
    main()

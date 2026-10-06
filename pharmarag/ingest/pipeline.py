"""Ingest one seed ingredient end to end: resolve, select a label, parse, chunk, embed, load."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field, replace

import httpx
import psycopg

from pharmarag.ingest.chunker import ChunkRecord, chunk_label, embed_text
from pharmarag.ingest.dailymed import (
    MARKETING_CATEGORY_BLA,
    MARKETING_CATEGORY_NDA,
    DailyMedClient,
)
from pharmarag.ingest.load import (
    LabelRow,
    load_label,
    loaded_version,
    prune_other_labels,
    upsert_aliases,
    upsert_ingredients,
)
from pharmarag.ingest.pins import Pins
from pharmarag.ingest.rxnav import Ingredient, RxNavClient
from pharmarag.ingest.select import Candidate, ingredient_key, rank
from pharmarag.ingest.spl_parser import ParsedLabel, SplParseError, parse_spl
from pharmarag.models.embedder import Embedder
from pharmarag.sections import PRESCRIPTION_LABEL_LOINC

log = logging.getLogger(__name__)

# A label must have content (a non-empty block) in these sections and in at least MIN_SECTIONS
# kept sections overall. Some repackager labels are stubs, and some put dosing inside
# OVERDOSAGE with no Dosage and administration section; both are skipped for the next label.
CORE_SECTIONS = {"34067-9": "Indications and usage", "34068-7": "Dosage and administration"}
MIN_SECTIONS = 3
# At most this many label XMLs are downloaded per seed before giving up.
MAX_FETCHES = 12
# At most this many candidates get their ingredient set resolved (the pool holds up to 300).
MAX_CHECKS = 200


def usability_problem(parsed: ParsedLabel) -> str | None:
    """Why a parsed label cannot be the reference label, or None if it can."""
    with_content = {s.loinc for s in parsed.sections if any(sub.blocks for sub in s.subsections)}
    missing = [name for code, name in CORE_SECTIONS.items() if code not in with_content]
    if missing:
        return "no text in " + " and ".join(missing)
    if len(with_content) < MIN_SECTIONS:
        return f"only {len(with_content)} sections with text"
    return None


@dataclass
class SeedResult:
    name: str
    status: str  # loaded | unchanged | stale | unresolved | no_label
    ingredient_rxcui: str | None = None
    setids: list[str] = field(default_factory=list)
    chunks: int = 0  # chunks written this run (inserted or replaced labels only)
    stale: int = 0  # labels whose version in the database is newer than DailyMed's
    aliases: int = 0
    pruned: int = 0


@dataclass
class _Evaluated:
    candidate: Candidate
    product_rxcuis: list[str]
    ingredients: list[Ingredient]


class Ingestor:
    def __init__(
        self,
        *,
        rxnav: RxNavClient,
        dailymed: DailyMedClient,
        mappings: dict[str, set[str]],
        embedder: Embedder,
        conn: psycopg.AsyncConnection,
        labels_per_ingredient: int = 1,
        max_checks: int = MAX_CHECKS,
        max_fetches: int = MAX_FETCHES,
        pins: Pins | None = None,
        repin: bool = False,
    ) -> None:
        self.rxnav = rxnav
        self.dailymed = dailymed
        self.mappings = mappings
        self.embedder = embedder
        self.conn = conn
        self.labels_per_ingredient = labels_per_ingredient
        self.max_checks = max_checks
        self.max_fetches = max_fetches
        self.pins = pins or Pins()
        self.repin = repin

    async def _candidates(self, ing: Ingredient) -> AsyncIterator[_Evaluated]:
        """Every prescription label for the ingredient, best first, resolved on demand.

        The NDA list, the BLA list and the full list are merged (a setid keeps its NDA or BLA
        category) and ranked once with `select.rank`, because the base-form rule outranks the
        tier. Resolving the ingredient set lazily matters: a candidate costs RxNav calls, and
        the first usable single-ingredient label usually ends the search.

        SPEC: the list API gives DailyMed's publication date, not the SPL effective date, so
        it stands in for "latest effective date" when ranking. The label's own effective
        date is stored once it is parsed.
        """
        pool: dict[str, Candidate] = {}
        for code, category in ((MARKETING_CATEGORY_NDA, "NDA"), (MARKETING_CATEGORY_BLA, "BLA")):
            refs = await self.dailymed.list_labels(ing.rxcui, marketing_category_code=code)
            for r in refs:
                pool[r.setid] = Candidate(
                    r.setid, r.spl_version, category, r.published_date, "", r.title
                )
        for r in await self.dailymed.list_labels(ing.rxcui):
            pool.setdefault(
                r.setid,
                Candidate(r.setid, r.spl_version, "OTHER", r.published_date, "", r.title),
            )
        for cand in rank(pool.values())[: self.max_checks]:
            product_rxcuis = self.mappings.get(cand.setid)
            if not product_rxcuis:
                log.debug("setid %s is not in the RxNorm mapping file; skipped", cand.setid)
                continue
            key, ingredients = await self.rxnav.product_ingredient_key(product_rxcuis)
            if key:
                yield _Evaluated(
                    replace(cand, ingredient_key=key), sorted(product_rxcuis), ingredients
                )

    async def _fetch_usable(self, setid: str) -> ParsedLabel | None:
        """Download, parse and check one label. None means "not usable" (reason is logged)."""
        try:
            xml = await self.dailymed.fetch_spl_xml(setid)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                log.warning("setid %s: no longer on DailyMed", setid)
                return None
            raise
        try:
            parsed = await asyncio.to_thread(parse_spl, xml)
        except SplParseError as e:
            log.warning("setid %s: unparseable (%s)", setid, e)
            return None
        if parsed.doc_type_loinc != PRESCRIPTION_LABEL_LOINC:
            log.info(
                "setid %s: document type %s is not a prescription label",
                setid,
                parsed.doc_type_loinc,
            )
            return None
        problem = usability_problem(parsed)
        if problem:
            log.info("setid %s (%s): %s; trying the next label", setid, parsed.title[:50], problem)
            return None
        return parsed

    async def _pinned(
        self, ing: Ingredient, target_key: str
    ) -> list[tuple[_Evaluated, ParsedLabel]]:
        """Pinned labels that still exist, still pass the checks, and still match the key."""
        out: list[tuple[_Evaluated, ParsedLabel]] = []
        for setid in self.pins.get(ing.rxcui)[: self.labels_per_ingredient]:
            product_rxcuis = self.mappings.get(setid)
            if not product_rxcuis:
                log.warning(
                    "%s: pinned setid %s is not in the mapping file; reselecting", ing.name, setid
                )
                continue
            key, ingredients = await self.rxnav.product_ingredient_key(product_rxcuis)
            parsed = await self._fetch_usable(setid) if key == target_key else None
            if parsed is None:
                log.warning("%s: pinned setid %s is no longer usable; reselecting", ing.name, setid)
                continue
            cand = Candidate(setid, parsed.version, "PIN", None, key, parsed.title)
            out.append((_Evaluated(cand, sorted(product_rxcuis), ingredients), parsed))
        return out

    async def ingest_seed(self, name: str) -> SeedResult:
        ing = await self.rxnav.find_ingredient(name)
        if ing is None:
            log.warning("%s: no RxNorm ingredient found", name)
            return SeedResult(name, "unresolved")
        aliases = await self.rxnav.aliases(ing)
        await upsert_ingredients(self.conn, [ing])
        await upsert_aliases(self.conn, aliases)

        target_key = ingredient_key([ing.rxcui])
        chosen: list[tuple[_Evaluated, ParsedLabel]] = []
        if not self.repin:
            chosen = await self._pinned(ing, target_key)
        have = {parsed.setid for _, parsed in chosen}
        fetches = 0
        if len(chosen) < self.labels_per_ingredient:
            async with aclosing(self._candidates(ing)) as candidates:
                async for evaluated in candidates:
                    if evaluated.candidate.ingredient_key != target_key:
                        continue  # a combination product, or one that mixes ingredient sets
                    if evaluated.candidate.setid in have:
                        continue
                    if fetches >= self.max_fetches:
                        log.warning("%s: gave up after %d label downloads", name, fetches)
                        break
                    fetches += 1
                    parsed = await self._fetch_usable(evaluated.candidate.setid)
                    if parsed is not None:
                        chosen.append((evaluated, parsed))
                        if len(chosen) >= self.labels_per_ingredient:
                            break
        if not chosen:
            log.warning("%s (%s): no usable single-ingredient prescription label", name, ing.rxcui)
            return SeedResult(name, "no_label", ing.rxcui, aliases=len(aliases))

        result = SeedResult(name, "unchanged", ing.rxcui, aliases=len(aliases))
        for evaluated, parsed in chosen:
            status, written = await self._load(evaluated, parsed)
            result.setids.append(parsed.setid)
            result.chunks += written
            if status in ("inserted", "replaced"):
                result.status = "loaded"
            elif status == "stale":
                result.stale += 1
        if result.status == "unchanged" and result.stale == len(chosen):
            result.status = "stale"
        self.pins.set(ing.rxcui, ing.name, result.setids)
        result.pruned = await prune_other_labels(self.conn, target_key, result.setids)
        return result

    async def _load(self, evaluated: _Evaluated, parsed: ParsedLabel) -> tuple[str, int]:
        """Returns (status, chunks written). Only inserted and replaced write chunks."""
        have = await loaded_version(self.conn, parsed.setid)
        if have == parsed.version:
            return "unchanged", 0
        if have is not None and have > parsed.version:
            log.warning(
                "%s: database has version %d, DailyMed served %d; keeping the newer one",
                parsed.title,
                have,
                parsed.version,
            )
            return "stale", 0
        chunks: list[ChunkRecord] = await asyncio.to_thread(
            chunk_label, parsed, self.embedder.count_tokens
        )
        drug_name = " / ".join(i.name for i in evaluated.ingredients)
        texts = [embed_text(drug_name, c) for c in chunks]
        embeddings = await asyncio.to_thread(self.embedder.embed_documents, texts)
        await upsert_ingredients(self.conn, evaluated.ingredients)
        row = LabelRow(
            setid=parsed.setid,
            spl_version=parsed.version,
            title=parsed.title,
            labeler=parsed.labeler,
            marketing_category=parsed.marketing_category,
            effective_date=parsed.effective_date,
            ingredient_key=evaluated.candidate.ingredient_key,
            ingredient_rxcuis=[i.rxcui for i in evaluated.ingredients],
            product_rxcuis=evaluated.product_rxcuis,
        )
        status = await load_label(self.conn, row, chunks, embeddings)
        log.info("%s: %s, %d chunks", parsed.title, status, len(chunks))
        return status, len(chunks) if status in ("inserted", "replaced") else 0

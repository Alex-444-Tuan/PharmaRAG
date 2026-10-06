"""Ingest one seed ingredient end to end: resolve, select a label, parse, chunk, embed, load."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field, replace

import psycopg

from pharmarag.ingest.chunker import ChunkRecord, chunk_label, embed_text
from pharmarag.ingest.dailymed import (
    MARKETING_CATEGORY_BLA,
    MARKETING_CATEGORY_NDA,
    DailyMedClient,
    LabelRef,
)
from pharmarag.ingest.load import (
    LabelRow,
    label_loaded,
    load_label,
    prune_other_labels,
    upsert_aliases,
    upsert_ingredients,
)
from pharmarag.ingest.rxnav import Ingredient, RxNavClient
from pharmarag.ingest.select import Candidate, ingredient_key, rank
from pharmarag.ingest.spl_parser import ParsedLabel, SplParseError, parse_spl
from pharmarag.models.embedder import Embedder
from pharmarag.sections import PRESCRIPTION_LABEL_LOINC

log = logging.getLogger(__name__)

# A label that keeps fewer distinct kept sections than this is treated as truncated
# (some repackager labels carry only a few sections) and the next candidate is tried.
MIN_SECTIONS = 3
# At most this many candidates per tier get their ingredient set resolved.
MAX_CHECKS = 60


@dataclass
class SeedResult:
    name: str
    status: str  # loaded | unchanged | unresolved | no_label
    ingredient_rxcui: str | None = None
    setids: list[str] = field(default_factory=list)
    chunks: int = 0
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
    ) -> None:
        self.rxnav = rxnav
        self.dailymed = dailymed
        self.mappings = mappings
        self.embedder = embedder
        self.conn = conn
        self.labels_per_ingredient = labels_per_ingredient
        self.max_checks = max_checks

    async def _evaluate(self, refs: list[LabelRef], category: str) -> AsyncIterator[_Evaluated]:
        """Yield one tier's candidates best first, resolving each ingredient set on demand.

        Ranking is `select.rank` (latest date, then lowest SETID; the tier is constant here).
        Resolving lazily matters: a candidate costs RxNav calls, and the first usable
        single-ingredient label usually ends the search.

        SPEC: the list API gives DailyMed's publication date, not the SPL effective date, so
        it stands in for "latest effective date" when ranking. The label's own effective
        date is stored once it is parsed.
        """
        unresolved = [
            Candidate(r.setid, r.spl_version, category, r.published_date, "") for r in refs
        ]
        for cand in rank(unresolved)[: self.max_checks]:
            product_rxcuis = self.mappings.get(cand.setid)
            if not product_rxcuis:
                log.debug("setid %s is not in the RxNorm mapping file; skipped", cand.setid)
                continue
            key, ingredients = await self.rxnav.product_ingredient_key(product_rxcuis)
            if key:
                yield _Evaluated(
                    replace(cand, ingredient_key=key), sorted(product_rxcuis), ingredients
                )

    async def _candidates(self, ing: Ingredient) -> AsyncIterator[_Evaluated]:
        """NDA labels, then BLA, then every other prescription label, each best first."""
        seen: set[str] = set()
        for code, category in ((MARKETING_CATEGORY_NDA, "NDA"), (MARKETING_CATEGORY_BLA, "BLA")):
            refs = await self.dailymed.list_labels(ing.rxcui, marketing_category_code=code)
            seen.update(r.setid for r in refs)
            async for ev in self._evaluate(refs, category):
                yield ev
        everything = await self.dailymed.list_labels(ing.rxcui)
        async for ev in self._evaluate([r for r in everything if r.setid not in seen], "OTHER"):
            yield ev

    async def _fetch_usable(self, setid: str) -> ParsedLabel | None:
        try:
            parsed = parse_spl(await self.dailymed.fetch_spl_xml(setid))
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
        if len({s.loinc for s in parsed.sections}) < MIN_SECTIONS:
            log.info(
                "setid %s: only %d kept sections; trying the next label",
                setid,
                len(parsed.sections),
            )
            return None
        return parsed

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
        async with aclosing(self._candidates(ing)) as candidates:
            async for evaluated in candidates:
                if evaluated.candidate.ingredient_key != target_key:
                    continue  # a combination product, or one that mixes ingredient sets
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
            status = await self._load(evaluated, parsed)
            result.setids.append(parsed.setid)
            result.chunks += status[1]
            if status[0] != "unchanged":
                result.status = "loaded"
        result.pruned = await prune_other_labels(self.conn, target_key, result.setids)
        return result

    async def _load(self, evaluated: _Evaluated, parsed: ParsedLabel) -> tuple[str, int]:
        key = evaluated.candidate.ingredient_key
        if await label_loaded(self.conn, parsed.setid, parsed.version):
            return "unchanged", 0
        chunks: list[ChunkRecord] = chunk_label(parsed, self.embedder.count_tokens)
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
            ingredient_key=key,
            ingredient_rxcuis=[i.rxcui for i in evaluated.ingredients],
            product_rxcuis=evaluated.product_rxcuis,
        )
        status = await load_label(self.conn, row, chunks, embeddings)
        log.info("%s: %s, %d chunks", parsed.title, status, len(chunks))
        return status, len(chunks)

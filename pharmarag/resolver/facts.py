"""RxNorm facts branch (spec section 8): short fact lines about an ingredient.

For each ingredient RxCUI: its brand names, dose forms and strengths from RxNav, one line
each. The writer may cite a line (r1, r2, ...) and the verifier treats it as a span, so every
line is a plain statement of what RxNorm lists, nothing more.

Lines are deterministic: names sorted, strengths in numeric order, ingredients interleaved
in the order given. At most `thresholds.retrieval.max_facts` lines in total.

Facts are supplementary. If RxNav is down the source returns no lines instead of raising,
and does not cache the gap: the answer then rests on label text alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from itertools import zip_longest

from pharmarag.cache import Cache
from pharmarag.resolver.base import RxFact
from pharmarag.resolver.rxnav_client import Concept, RxNavSource, RxNavUnavailable

log = logging.getLogger(__name__)

CACHE_TTL_S = 24 * 3600
MAX_BRANDS = 8
MAX_DOSE_FORMS = 8
MAX_STRENGTHS = 10
# RxNorm lists a combination product's brand under each of its ingredients (Tribenzor is a
# "brand of amlodipine"). The same `related.json` call also returns the branded drugs (SBD,
# "amlodipine 5 MG Oral Tablet [Norvasc]"), and a brand is kept only if one of its branded
# drugs has a single ingredient. That is one RxNav call per ingredient, not one per brand.

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_BRAND_TAG = re.compile(r"^(?P<product>.*?)\s*\[(?P<brand>[^\[\]]+)\]\s*$")


def single_ingredient_brands(branded_drugs: list[Concept]) -> set[str]:
    """Lowercase brand names that have at least one single-ingredient branded drug."""
    out: set[str] = set()
    for c in branded_drugs:
        m = _BRAND_TAG.match(c.name)
        # " / " separates ingredients in a product name; "1 MG/ML" is a strength.
        if m and " / " not in m.group("product"):
            out.add(m.group("brand").lower())
    return out


def cache_key(rxcui: str) -> str:
    return f"rx:facts:{rxcui}"


def _strength(name: str, ingredient: str) -> str:
    """ "amlodipine 2.5 MG" -> "2.5 MG". Strips the leading ingredient name when present."""
    prefix = ingredient.lower() + " "
    return name[len(prefix) :] if name.lower().startswith(prefix) else name


def _numeric_order(text: str) -> tuple[float, str]:
    m = _NUMBER.search(text)
    return (float(m.group()) if m else float("inf"), text)


class RxNavFactsSource:
    def __init__(self, rxnav: RxNavSource, cache: Cache, max_facts: int) -> None:
        self._rx = rxnav
        self._cache = cache
        self._max = max_facts

    async def facts(self, ingredient_rxcuis: list[str]) -> list[RxFact]:
        rxcuis = list(dict.fromkeys(r for r in ingredient_rxcuis if r))
        if not rxcuis or self._max <= 0:
            return []
        per_ingredient = await asyncio.gather(*(self._lines(r) for r in rxcuis))
        # Interleave (first line of each ingredient, then the second, ...) so that when
        # max_facts cuts the list, every ingredient still has its brand line.
        merged = [
            RxFact(rxcui=rxcuis[i], text=line)
            for group in zip_longest(*per_ingredient)
            for i, line in enumerate(group)
            if line is not None
        ]
        return merged[: self._max]

    async def _lines(self, rxcui: str) -> list[str]:
        raw = await self._cache.get(cache_key(rxcui))
        if raw is not None:
            try:
                lines = json.loads(raw)
                if isinstance(lines, list) and all(isinstance(x, str) for x in lines):
                    return lines
            except ValueError:
                pass  # a corrupt entry is a miss
        try:
            lines = await self._fetch(rxcui)
        except RxNavUnavailable:
            log.warning("rx facts unavailable: RxNav did not answer")
            return []
        if lines:
            await self._cache.set(cache_key(rxcui), json.dumps(lines).encode(), CACHE_TTL_S)
        return lines

    async def _fetch(self, rxcui: str) -> list[str]:
        related = await self._rx.related(rxcui)
        if related.ingredient is None:
            return []
        name = related.ingredient.name

        single = single_ingredient_brands(related.branded_drugs)
        brands = sorted(
            {c.name for c in related.brands if c.name.lower() in single}, key=str.lower
        )[:MAX_BRANDS]

        forms = sorted({c.name for c in related.dose_forms}, key=str.lower)[:MAX_DOSE_FORMS]
        strengths = sorted(
            {_strength(c.name, name) for c in related.strengths}, key=_numeric_order
        )[:MAX_STRENGTHS]

        lines: list[str] = []
        if brands:
            lines.append(f"RxNorm brand names for {name}: {', '.join(brands)}.")
        if forms:
            lines.append(f"RxNorm dose forms for {name}: {', '.join(forms)}.")
        if strengths:
            lines.append(f"RxNorm strengths for {name}: {', '.join(strengths)}.")
        return lines

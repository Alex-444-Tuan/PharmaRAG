"""The resolution cascade (spec section 7, steps 1 to 10).

    1. normalise              6. map candidates to ingredient_key, de-duplicate
    2. local alias exact      7. score each key (Jaro-Winkler against all of its names)
    3. RxNav exact            8. decision gate  (gate.py)
    4. RxNav approximate      9. coverage       (a label must exist)
    5. local trigram         10. cache

Steps 2 and 3 stop on a hit: a correctly spelled name is never "corrected". Steps 4 and 5 both
run when 2 and 3 miss, and their candidates are scored together.

When RxNav is unavailable after the client's retries, steps 3 and 4 are skipped and the cascade
runs on the local alias table alone (NFR-4). The result is marked degraded in `Outcome` and in
the logs. `Resolution` itself has no field for it (see the report: contract change request).

SPEC: step 10 says to cache the result. What is cached here is the result of steps 1 to 7 (the
scored candidates), not of the gate and coverage. Coverage then follows the corpus, not the
24 h cache (a label ingested an hour ago is found), and a threshold change takes effect without
a flush. Results from the local alias table alone are not cached either: a Postgres primary-key
lookup costs what a Redis round trip costs, and the cache entry could only go stale.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from rapidfuzz.distance import JaroWinkler

from pharmarag.app.settings import ResolverThresholds
from pharmarag.cache import Cache
from pharmarag.graph.state import Candidate, MatchKind, Resolution
from pharmarag.ingest.select import ingredient_key
from pharmarag.resolver.gate import KeyScore, decide
from pharmarag.resolver.local import AliasRow, LocalStore, strongest
from pharmarag.resolver.normalize import collapse, normalize
from pharmarag.resolver.rxnav_client import RxNavSource, RxNavUnavailable

log = logging.getLogger(__name__)

CACHE_TTL_S = 24 * 3600
CACHE_VERSION = 1
# RxNav can return many rows for one term. Mapping each to its ingredient costs a request, so
# only the best few distinct concepts are mapped (rank order).
MAX_APPROX_CONCEPTS = 10

Step = Literal["empty", "local_exact", "rxnav_exact", "approximate"]

_BRAND_TTYS = frozenset({"BN", "SBD", "SBDC", "SBDF", "SBDG", "BPCK"})


@dataclass
class Scored:
    """One candidate ingredient key with its local score (steps 6 and 7)."""

    key: str
    name: str
    score: float
    match: MatchKind | None = None  # set by exact hits; None means "approximate"
    product_hint: str | None = None


@dataclass
class Cascaded:
    """What steps 1 to 7 produced for one mention. This is what the cache stores."""

    norm: str
    exact: bool
    scored: list[Scored] = field(default_factory=list)
    step: Step = "approximate"
    degraded: bool = False

    def to_json(self) -> bytes:
        return json.dumps({"v": CACHE_VERSION, **asdict(self)}, sort_keys=True).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> Cascaded | None:
        try:
            d = json.loads(raw)
            if d.get("v") != CACHE_VERSION:
                return None
            return cls(
                norm=d["norm"],
                exact=bool(d["exact"]),
                scored=[Scored(**s) for s in d["scored"]],
                step=d["step"],
                degraded=bool(d["degraded"]),
            )
        except (ValueError, KeyError, TypeError):
            return None  # a corrupt or old entry is a miss


@dataclass(frozen=True)
class Outcome:
    """A `Resolution` plus how it was reached. `Resolver.resolve` returns only the resolution."""

    resolution: Resolution
    step: Step
    degraded: bool
    cached: bool


def cache_key(norm: str) -> str:
    return f"rx:resolve:{norm}"


def _split_key(key: str) -> list[str]:
    return key.split("+")


def _pin_matches(key: str, pins: Collection[str]) -> bool:
    """A pin names one ingredient RxCUI. A combination key matches only when the pin spells
    out the same ingredient set ("6809+593411"), never when it merely shares one ingredient."""
    return any(
        ingredient_key(_split_key(p)) == key
        for p in pins
        if all(part.isdigit() for part in _split_key(p))
    )


def _jw(a: str, b: str) -> float:
    return float(JaroWinkler.similarity(a, b))


class CascadeResolver:
    def __init__(
        self,
        store: LocalStore,
        rxnav: RxNavSource | None,
        thresholds: ResolverThresholds,
        cache: Cache,
    ) -> None:
        """`rxnav=None` is local-only mode (no RxNav calls, not degraded)."""
        self._store = store
        self._rx = rxnav
        self._th = thresholds
        self._cache = cache

    # --- public ---------------------------------------------------------------------------

    async def resolve(
        self, mentions: list[str], *, pinned_rxcuis: list[str] | None = None
    ) -> list[Resolution]:
        return [o.resolution for o in await self.resolve_detailed(mentions, pinned_rxcuis)]

    async def resolve_detailed(
        self, mentions: Sequence[str], pinned_rxcuis: Sequence[str] | None = None
    ) -> list[Outcome]:
        pins = [p.strip() for p in pinned_rxcuis or [] if p and p.strip()]
        return list(await asyncio.gather(*(self._resolve_one(m, pins) for m in mentions)))

    # --- one mention ----------------------------------------------------------------------

    async def _resolve_one(self, mention: str, pins: list[str]) -> Outcome:
        norm = normalize(mention)
        if not norm:
            return Outcome(Resolution(mention=mention, status="unknown"), "empty", False, False)

        cached = await self._cache_get(norm)
        cascaded = cached or await self._cascade(mention, norm)
        if self._cacheable(cached, cascaded):
            await self._cache.set(cache_key(norm), cascaded.to_json(), CACHE_TTL_S)
        if cascaded.degraded:
            log.warning("resolver degraded: RxNav unavailable, local alias table only")
        resolution = await self._finish(mention, cascaded, pins)
        return Outcome(resolution, cascaded.step, cascaded.degraded, cached is not None)

    def _cacheable(self, cached: Cascaded | None, c: Cascaded) -> bool:
        """Only results that needed RxNav are worth caching, and only if RxNav answered. A
        local-only resolver (no RxNav) must not seed the cache that an RxNav-backed one reads."""
        return (
            cached is None
            and self._rx is not None
            and not c.degraded
            and c.step in ("rxnav_exact", "approximate")
        )

    async def _cache_get(self, norm: str) -> Cascaded | None:
        raw = await self._cache.get(cache_key(norm))
        return Cascaded.from_json(raw) if raw is not None else None

    async def _cascade(self, mention: str, norm: str) -> Cascaded:
        # Step 2. The collapsed form is also tried: some brands carry a form word ("toprol-xl",
        # "glucotrol xl") that normalisation strips.
        rows = await self._store.exact({norm, collapse(mention)})
        if rows:
            return Cascaded(norm, True, await self._exact_from_rows(rows), "local_exact")

        names: dict[str, set[str]] = {}  # key -> extra names learned from RxNav
        candidates: set[str] = set()
        degraded = False
        if self._rx is not None:
            try:
                # Step 3.
                hits = await self._rxnav_exact(norm, names)
                if hits:
                    return Cascaded(norm, True, hits, "rxnav_exact")
                # Step 4.
                candidates |= await self._rxnav_approximate(norm, names)
            except RxNavUnavailable:
                degraded = True
                names.clear()
                candidates.clear()

        # Step 5.
        near = await self._store.trigram(
            norm, min_similarity=self._th.trigram_min, k=self._th.trigram_k
        )
        candidates |= {r.ingredient_key for r in near}

        # Steps 6 and 7.
        scored = await self._score(norm, candidates, names)
        return Cascaded(norm, False, scored, "approximate", degraded)

    # --- steps 2 and 3: exact hits ---------------------------------------------------------

    @staticmethod
    def _match_for(tty: str) -> MatchKind:
        if tty == "PIN":
            return "salt"
        if tty in _BRAND_TTYS:
            return "brand"
        return "exact"  # IN, SY, MIN and generic clinical drugs

    async def _exact_from_rows(self, rows: list[AliasRow]) -> list[Scored]:
        by_key: dict[str, list[AliasRow]] = {}
        for r in rows:
            by_key.setdefault(r.ingredient_key, []).append(r)
        display = await self._display_names(
            list(by_key), {}, {k: strongest(v).alias for k, v in by_key.items()}
        )
        out: list[Scored] = []
        for key in sorted(by_key, key=_key_order):
            best = strongest(by_key[key])
            match = self._match_for(best.tty)
            out.append(
                Scored(
                    key,
                    display[key],
                    1.0,
                    match,
                    best.source_rxcui if match in ("brand", "salt") else None,
                )
            )
        return out

    async def _rxnav_exact(self, norm: str, names: dict[str, set[str]]) -> list[Scored]:
        assert self._rx is not None
        rxcuis = await self._rx.find_rxcuis(norm)
        if not rxcuis:
            return []
        local = await self._store.by_source(rxcuis)
        mapped = await asyncio.gather(*(self._classify(r, local.get(r), names) for r in rxcuis))
        best: dict[str, Scored] = {}
        for item in mapped:
            if item is None:
                continue
            key, match, hint = item
            old = best.get(key)
            if old is None or _MATCH_ORDER[match] < _MATCH_ORDER[old.match or "approximate"]:
                best[key] = Scored(
                    key, "", 1.0, match, hint if match in ("brand", "salt") else None
                )
        display = await self._display_names(list(best), names)
        for key, s in best.items():
            s.name = display[key]
        return [best[k] for k in sorted(best, key=_key_order)]

    async def _classify(
        self, rxcui: str, local: AliasRow | None, names: dict[str, set[str]]
    ) -> tuple[str, MatchKind, str | None] | None:
        """(ingredient_key, match kind, product rxcui) of one RxNav hit, or None if the concept
        has no ingredient. The alias table answers when it knows the concept."""
        assert self._rx is not None
        if local is not None:
            return local.ingredient_key, self._match_for(local.tty), rxcui
        concept = await self._rx.concept(rxcui)
        ingredients = await self._rx.ingredients(rxcui)
        if not ingredients:
            return None
        key = ingredient_key([i.rxcui for i in ingredients])
        for ing in ingredients:
            names.setdefault(ing.rxcui, set()).add(ing.name.lower())
        return key, self._match_for(concept.tty if concept else ""), rxcui

    # --- step 4: approximate candidates ------------------------------------------------------

    async def _rxnav_approximate(self, norm: str, names: dict[str, set[str]]) -> set[str]:
        assert self._rx is not None
        approx = await self._rx.approximate(norm, max_entries=10)
        rxcuis = [a.rxcui for a in approx[:MAX_APPROX_CONCEPTS]]
        local = await self._store.by_source(rxcuis)
        keys: set[str] = {row.ingredient_key for row in local.values()}
        unknown = [r for r in rxcuis if r not in local]
        for ingredients in await asyncio.gather(*(self._rx.ingredients(r) for r in unknown)):
            if ingredients:
                keys.add(ingredient_key([i.rxcui for i in ingredients]))
                for ing in ingredients:
                    names.setdefault(ing.rxcui, set()).add(ing.name.lower())
        return keys

    # --- steps 6 and 7 -----------------------------------------------------------------------

    async def _score(self, norm: str, keys: set[str], names: dict[str, set[str]]) -> list[Scored]:
        if not keys:
            return []
        aliases = await self._store.aliases_of(keys)
        scores: dict[str, tuple[float, str]] = {}  # key -> (best score, best-matching name)
        for key in keys:
            known = list(aliases.get(key, []))
            if "+" not in key:
                known += sorted(names.get(key, ()))
            if known:  # a key with no name at all cannot be compared with the mention
                scores[key] = max((_jw(norm, a), a) for a in known)
        display = await self._display_names(
            sorted(scores), names, {k: v[1] for k, v in scores.items()}
        )
        out = [
            Scored(k, display[k], round(scores[k][0], 4)) for k in sorted(scores, key=_key_order)
        ]
        out.sort(key=lambda s: (-s.score, _key_order(s.key)))
        return out

    async def _display_names(
        self,
        keys: Sequence[str],
        rx_names: dict[str, set[str]],
        fallback: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Display name of each key: its ingredient names joined with " / ". A combination
        with an ingredient outside the corpus and unknown to RxNav shows `fallback[key]` (the
        matched brand name) instead of a bare RxCUI."""
        rxcuis = {r for k in keys for r in _split_key(k)}
        local = await self._store.ingredient_names(rxcuis)
        out: dict[str, str] = {}
        for key in keys:
            parts: list[str] = []
            for rxcui in _split_key(key):
                if rxcui in local:
                    parts.append(local[rxcui])
                elif rx_names.get(rxcui):
                    parts.append(sorted(rx_names[rxcui])[0])
                else:
                    parts = []
                    break
            out[key] = " / ".join(parts) if parts else (fallback or {}).get(key, key)
        return out

    # --- steps 8 and 9, and pins ---------------------------------------------------------------

    async def _finish(self, mention: str, c: Cascaded, pins: list[str]) -> Resolution:
        pinned = [s for s in c.scored if _pin_matches(s.key, pins)] if pins else []
        if pinned:
            winner = min(pinned, key=lambda s: (-s.score, _key_order(s.key)))
            return await self._resolved(mention, winner, "pinned")

        verdict = decide([KeyScore(s.key, s.score) for s in c.scored], self._th, exact=c.exact)
        by_key = {s.key: s for s in c.scored}
        if verdict.outcome == "unknown":
            top = verdict.top
            return Resolution(mention=mention, status="unknown", score=top.score if top else None)
        if verdict.outcome == "ambiguous":
            assert verdict.top is not None
            return Resolution(
                mention=mention,
                status="ambiguous",
                score=verdict.top.score,
                candidates=[
                    Candidate(
                        ingredient_key=o.key, name=by_key[o.key].name, score=round(o.score, 4)
                    )
                    for o in verdict.candidates
                ],
            )
        assert verdict.top is not None
        winner = by_key[verdict.top.key]
        return await self._resolved(mention, winner, winner.match or "approximate")

    async def _resolved(self, mention: str, s: Scored, match: MatchKind) -> Resolution:
        """Step 9: a resolved drug with no label in the corpus is `not_covered`."""
        has_label = s.key in await self._store.covered([s.key])
        return Resolution(
            mention=mention,
            status="resolved" if has_label else "not_covered",
            match=match,
            ingredient_key=s.key,
            name=s.name,
            ingredient_rxcuis=_split_key(s.key),
            product_hint=s.product_hint,
            score=s.score,
        )


_MATCH_ORDER: dict[str, int] = {"exact": 0, "pinned": 0, "salt": 1, "brand": 2, "approximate": 3}


def _key_order(key: str) -> list[int]:
    return [int(p) for p in _split_key(key)]

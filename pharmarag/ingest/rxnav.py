"""Minimal RxNav client for ingestion (spec section 4).

Resolves a seed name to an ingredient RxCUI, maps any RxCUI to its ingredients, and fetches the
names a user might type for an ingredient. This is separate from the request-time resolver
client that Cycle 2 adds under `pharmarag/resolver/`.

RxNav allows 20 requests per second per IP. The limiter here spaces calls by 1 / max_rps and
the default is 15, which leaves headroom for other processes on the same IP.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from pharmarag.ingest.http import RateLimiter, get_with_retry
from pharmarag.ingest.select import ingredient_key

log = logging.getLogger(__name__)

DEFAULT_MAX_RPS = 15.0


@dataclass(frozen=True)
class Ingredient:
    rxcui: str
    name: str


@dataclass(frozen=True)
class Alias:
    """A name a user might type, with the ingredient set it denotes."""

    alias: str  # lowercased
    tty: str  # IN, PIN, BN or SY
    source_rxcui: str
    ingredient_key: str


class RxNavClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
        max_rps: float = DEFAULT_MAX_RPS,
        retries: int = 2,
        backoff_s: float = 0.5,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._limiter = RateLimiter(max_rps)
        self._retries = retries
        self._backoff_s = backoff_s
        self._ingredients_cache: dict[str, list[Ingredient]] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _json(self, path: str, **params: str) -> dict[str, Any]:
        resp = await get_with_retry(
            self._client,
            f"{self._base}{path}",
            retries=self._retries,
            backoff_s=self._backoff_s,
            before_request=self._limiter.wait,
            params=params or None,
        )
        return resp.json()

    async def _related(self, rxcui: str, **params: str) -> list[dict[str, Any]]:
        body = await self._json(f"/rxcui/{rxcui}/related.json", **params)
        out: list[dict[str, Any]] = []
        for group in (body.get("relatedGroup") or {}).get("conceptGroup") or []:
            out.extend(group.get("conceptProperties") or [])
        return out

    async def product_ingredients(self, rxcui: str) -> list[Ingredient]:
        """Ingredients (TTY IN) of a product-level or brand concept. Cached."""
        cached = self._ingredients_cache.get(rxcui)
        if cached is None:
            cached = [
                Ingredient(c["rxcui"], c["name"]) for c in await self._related(rxcui, tty="IN")
            ]
            self._ingredients_cache[rxcui] = cached
        return cached

    async def ingredients_of(self, rxcui: str) -> list[Ingredient]:
        """Ingredient concepts behind any RxCUI. An IN maps to itself."""
        props = (await self._json(f"/rxcui/{rxcui}/properties.json")).get("properties") or {}
        if props.get("tty") == "IN":
            return [Ingredient(rxcui, props["name"])]
        return await self.product_ingredients(rxcui)

    async def find_ingredient(self, name: str) -> Ingredient | None:
        """Seed name to its single ingredient concept, or None if there is no such concept.

        Tries an exact match, then RxNav's normalised match. A match that is a precise
        ingredient (for example "amlodipine besylate") resolves to its base ingredient.
        """
        for search in ("0", "2"):
            body = await self._json("/rxcui.json", name=name, search=search)
            for rxcui in (body.get("idGroup") or {}).get("rxnormId") or []:
                found = await self.ingredients_of(rxcui)
                if len(found) == 1:
                    return found[0]
        return None

    async def product_ingredient_key(
        self, product_rxcuis: set[str]
    ) -> tuple[str, list[Ingredient]]:
        """Ingredient key and ingredient list for the union of a label's product RxCUIs."""
        found: dict[str, Ingredient] = {}
        # Concurrent: RxNav latency (about 1 s) dominates, and the limiter still caps the rate.
        lists = await asyncio.gather(*(self.product_ingredients(r) for r in sorted(product_rxcuis)))
        for ingredients in lists:
            for ing in ingredients:
                found[ing.rxcui] = ing
        return ingredient_key(list(found)), sorted(found.values(), key=lambda i: int(i.rxcui))

    async def aliases(self, ing: Ingredient) -> list[Alias]:
        """Names for an ingredient: IN itself, its precise ingredients, brands, synonyms.

        Brands are looked up one by one because RxNav lists combination-product brands under
        each of their ingredients (Tribenzor is a brand of amlodipine). Each brand gets the
        key of its own ingredient set.
        """
        own_key = ingredient_key([ing.rxcui])
        out: dict[tuple[str, str], Alias] = {}

        def add(name: str, tty: str, source: str, key: str) -> None:
            alias = " ".join(name.lower().split())
            if alias:
                out.setdefault((alias, source), Alias(alias, tty, source, key))

        add(ing.name, "IN", ing.rxcui, own_key)
        for c in await self._related(ing.rxcui, tty="PIN"):
            add(c["name"], "PIN", c["rxcui"], own_key)
        # SPEC: related.json does not accept tty=SY. Synonyms come from the concept's
        # name properties (Tallman and RxNav display names) and the `synonym` field.
        names = await self._json(f"/rxcui/{ing.rxcui}/allProperties.json", prop="names")
        for prop in (names.get("propConceptGroup") or {}).get("propConcept") or []:
            add(prop.get("propValue", ""), "SY", ing.rxcui, own_key)
        brands = await self._related(ing.rxcui, tty="BN")
        brand_ingredients = await asyncio.gather(
            *(self.product_ingredients(c["rxcui"]) for c in brands)
        )
        for c, ingredients in zip(brands, brand_ingredients, strict=True):
            brand_key = ingredient_key([i.rxcui for i in ingredients])
            if brand_key:
                add(c["name"], "BN", c["rxcui"], brand_key)
        return list(out.values())

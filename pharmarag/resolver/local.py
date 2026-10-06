"""Local lookups the cascade needs: the alias table, ingredient names, label coverage (spec
section 5). Read-only SQL. The cascade depends on the `LocalStore` protocol, so it can be unit
tested without PostgreSQL where the SQL itself is not the point.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

# Lower is stronger. An alias string can appear under several term types; the strongest wins.
TTY_ORDER = {"IN": 0, "SY": 1, "PIN": 2, "BN": 3}


@dataclass(frozen=True)
class AliasRow:
    alias: str
    tty: str
    source_rxcui: str
    ingredient_key: str


class LocalStore(Protocol):
    async def exact(self, forms: Collection[str]) -> list[AliasRow]:
        """Alias rows whose alias equals any of `forms` (already lowercase). Step 2."""
        ...

    async def trigram(self, term: str, *, min_similarity: float, k: int) -> list[AliasRow]:
        """The `k` aliases nearest to `term` with `similarity` of at least `min_similarity`,
        nearest first. Step 5."""
        ...

    async def by_source(self, rxcuis: Collection[str]) -> dict[str, AliasRow]:
        """The alias row (strongest term type) of each RxCUI the alias table knows."""
        ...

    async def aliases_of(self, keys: Collection[str]) -> dict[str, list[str]]:
        """Every alias of each ingredient key. Step 7 scores the mention against all of them."""
        ...

    async def ingredient_names(self, rxcuis: Collection[str]) -> dict[str, str]: ...

    async def covered(self, keys: Collection[str]) -> set[str]:
        """The keys that have at least one label. Step 9."""
        ...


class PgLocalStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def exact(self, forms: Collection[str]) -> list[AliasRow]:
        forms = sorted({f for f in forms if f})
        if not forms:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT alias, alias_tty, source_rxcui, ingredient_key FROM drug_alias "
                "WHERE alias = ANY(%s::text[]) ORDER BY alias, source_rxcui",
                (forms,),
            )
            return [AliasRow(*r) for r in await cur.fetchall()]

    async def trigram(self, term: str, *, min_similarity: float, k: int) -> list[AliasRow]:
        if not term:
            return []
        async with self._pool.connection() as conn:
            # The `%` operator is what the GIN trigram index serves, and it compares against
            # pg_trgm.similarity_threshold. set_config(..., true) is SET LOCAL: it reverts when
            # the transaction ends, so a pooled connection is never left with our threshold.
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('pg_trgm.similarity_threshold', %s, true)",
                    (str(min_similarity),),
                )
                cur = await conn.execute(
                    "SELECT alias, alias_tty, source_rxcui, ingredient_key FROM drug_alias "
                    "WHERE alias %% %s ORDER BY similarity(alias, %s) DESC, alias, source_rxcui "
                    "LIMIT %s",
                    (term, term, k),
                )
                return [AliasRow(*r) for r in await cur.fetchall()]

    async def by_source(self, rxcuis: Collection[str]) -> dict[str, AliasRow]:
        rxcuis = sorted(set(rxcuis))
        if not rxcuis:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT alias, alias_tty, source_rxcui, ingredient_key FROM drug_alias "
                "WHERE source_rxcui = ANY(%s::text[])",
                (rxcuis,),
            )
            rows = [AliasRow(*r) for r in await cur.fetchall()]
        best: dict[str, AliasRow] = {}
        for row in rows:
            current = best.get(row.source_rxcui)
            if current is None or _strength(row) < _strength(current):
                best[row.source_rxcui] = row
        return best

    async def aliases_of(self, keys: Collection[str]) -> dict[str, list[str]]:
        keys = sorted(set(keys))
        if not keys:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT ingredient_key, alias FROM drug_alias "
                "WHERE ingredient_key = ANY(%s::text[]) ORDER BY ingredient_key, alias",
                (keys,),
            )
            out: dict[str, list[str]] = {}
            for key, alias in await cur.fetchall():
                out.setdefault(key, []).append(alias)
            return out

    async def ingredient_names(self, rxcuis: Collection[str]) -> dict[str, str]:
        rxcuis = sorted(set(rxcuis))
        if not rxcuis:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT rxcui, name FROM ingredient WHERE rxcui = ANY(%s::text[])", (rxcuis,)
            )
            return {r[0]: r[1] for r in await cur.fetchall()}

    async def covered(self, keys: Collection[str]) -> set[str]:
        keys = sorted(set(keys))
        if not keys:
            return set()
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT DISTINCT ingredient_key FROM label WHERE ingredient_key = ANY(%s::text[])",
                (keys,),
            )
            return {r[0] for r in await cur.fetchall()}


def _strength(row: AliasRow) -> tuple[int, str]:
    return (TTY_ORDER.get(row.tty, 9), row.alias)


def strongest(rows: Iterable[AliasRow]) -> AliasRow:
    return min(rows, key=_strength)

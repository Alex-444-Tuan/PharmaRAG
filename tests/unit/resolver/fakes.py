"""Test doubles for the resolver: a scripted RxNav and a small seeded alias table."""

from __future__ import annotations

import uuid

from psycopg import AsyncConnection

from pharmarag.resolver.rxnav_client import (
    ApproxCandidate,
    Concept,
    Ingredient,
    RelatedConcepts,
    RxNavUnavailable,
)


class FakeRx:
    """RxNavSource with canned answers. Records every call, raises when `down`."""

    def __init__(self) -> None:
        self.exact: dict[str, list[str]] = {}
        self.approx: dict[str, list[ApproxCandidate]] = {}
        self.concepts: dict[str, Concept] = {}
        self.ingredient_map: dict[str, list[Ingredient]] = {}
        self.related_map: dict[str, RelatedConcepts] = {}
        self.down = False
        self.calls: list[tuple[str, str]] = []

    def _call(self, name: str, arg: str) -> None:
        self.calls.append((name, arg))
        if self.down:
            raise RxNavUnavailable("down")

    @property
    def names_called(self) -> list[str]:
        return [c[0] for c in self.calls]

    async def find_rxcuis(self, name: str) -> list[str]:
        self._call("find_rxcuis", name)
        return list(self.exact.get(name, []))

    async def approximate(self, term: str, max_entries: int = 10) -> list[ApproxCandidate]:
        self._call("approximate", term)
        return list(self.approx.get(term, []))

    async def concept(self, rxcui: str) -> Concept | None:
        self._call("concept", rxcui)
        return self.concepts.get(rxcui)

    async def ingredients(self, rxcui: str) -> list[Ingredient]:
        self._call("ingredients", rxcui)
        return list(self.ingredient_map.get(rxcui, []))

    async def related(self, rxcui: str) -> RelatedConcepts:
        self._call("related", rxcui)
        return self.related_map.get(rxcui, RelatedConcepts())


# (rxcui, name). 39993 has no label: resolving it is "not covered".
INGREDIENTS = [
    ("5553", "hydroxyzine"),
    ("5470", "hydralazine"),
    ("2599", "clonidine"),
    ("2598", "clonazepam"),
    ("42347", "bupropion"),
    ("1827", "buspirone"),
    ("10689", "tramadol"),
    ("10737", "trazodone"),
    ("6809", "metformin"),
    ("6922", "metronidazole"),
    ("8640", "prednisone"),
    ("8638", "prednisolone"),
    ("5454", "glipizide"),
    ("4815", "glyburide"),
    ("4603", "furosemide"),
    ("39993", "zolpidem"),
]
NO_LABEL = {"39993"}

# (alias, tty, source_rxcui, ingredient_key)
ALIASES = [
    *((name, "IN", rxcui, rxcui) for rxcui, name in INGREDIENTS),
    ("hydroxyzine hydrochloride", "PIN", "154987", "5553"),
    ("hydroxyzine pamoate", "PIN", "203182", "5553"),
    ("vistaril", "BN", "282519", "5553"),
    ("hydralazine hydrochloride", "PIN", "82027", "5470"),
    ("catapres", "BN", "202941", "2599"),
    ("klonopin", "BN", "202585", "2598"),
    ("glucotrol xl", "BN", "205821", "5454"),
    ("ambien", "BN", "211801", "39993"),
    ("kombiglyze", "BN", "1372730", "6809+857974"),  # combination brand: no label
    ("glibenclamide", "SY", "4815", "4815"),
    ("lasix", "BN", "202991", "4603"),
]


async def seed(conn: AsyncConnection) -> None:
    """Insert the tables above into an empty migrated database and commit."""
    for rxcui, name in INGREDIENTS:
        await conn.execute("INSERT INTO ingredient (rxcui, name) VALUES (%s, %s)", (rxcui, name))
    for alias, tty, source, key in ALIASES:
        await conn.execute(
            "INSERT INTO drug_alias (alias, alias_tty, source_rxcui, ingredient_key) "
            "VALUES (%s, %s, %s, %s)",
            (alias, tty, source, key),
        )
    for rxcui, name in INGREDIENTS:
        if rxcui in NO_LABEL:
            continue
        await conn.execute(
            "INSERT INTO label (setid, spl_version, title, ingredient_key, ingredient_rxcuis, "
            "product_rxcuis) VALUES (%s, 1, %s, %s, %s, %s)",
            (uuid.uuid4(), f"{name} label", rxcui, [rxcui], []),
        )
    await conn.commit()

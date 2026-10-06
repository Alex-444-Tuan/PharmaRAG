import json

import pytest

from pharmarag.cache import InMemoryCache
from pharmarag.resolver.base import RxFact
from pharmarag.resolver.facts import RxNavFactsSource, single_ingredient_brands
from pharmarag.resolver.rxnav_client import Concept, Ingredient, RelatedConcepts
from tests.unit.resolver.fakes import FakeRx

AML = Ingredient("17767", "amlodipine")
LIS = Ingredient("29046", "lisinopril")


def concepts(tty, *pairs):
    return [Concept(r, n, tty) for r, n in pairs]


@pytest.fixture
def rx() -> FakeRx:
    fake = FakeRx()
    fake.related_map["17767"] = RelatedConcepts(
        ingredient=AML,
        brands=concepts(
            "BN",
            ("1372732", "Tribenzor"),  # combination products: not brands of amlodipine alone
            ("722127", "Exforge"),
            ("58927", "Norvasc"),
            ("2748470", "Amodip"),
            ("999", "Orphan"),  # a brand with no branded drug listed: cannot be verified
        ),
        branded_drugs=concepts(
            "SBD",
            (
                "1",
                "amlodipine 5 MG / hydrochlorothiazide 12.5 MG / olmesartan medoxomil 40 MG "
                "Oral Tablet [Tribenzor]",
            ),
            ("2", "amlodipine 5 MG / valsartan 160 MG Oral Tablet [Exforge]"),
            ("3", "amlodipine 10 MG / valsartan 320 MG Oral Tablet [Exforge]"),
            ("4", "amlodipine 5 MG Oral Tablet [Norvasc]"),
            ("5", "amlodipine 1 MG/ML Oral Suspension [Amodip]"),  # "1 MG/ML" is one strength
        ),
        dose_forms=concepts("DF", ("317541", "Oral Tablet"), ("316965", "Oral Capsule")),
        strengths=concepts(
            "SCDC",
            ("329526", "amlodipine 10 MG"),
            ("329527", "amlodipine 2.5 MG"),
            ("2184117", "amlodipine 1 MG/ML"),
            ("329528", "amlodipine 5 MG"),
        ),
    )
    fake.related_map["29046"] = RelatedConcepts(
        ingredient=LIS,
        brands=concepts("BN", ("104377", "Zestril")),
        branded_drugs=concepts("SBD", ("6", "lisinopril 10 MG Oral Tablet [Zestril]")),
        dose_forms=concepts("DF", ("317541", "Oral Tablet")),
        strengths=concepts("SCDC", ("197884", "lisinopril 10 MG")),
    )
    return fake


def source(rx, max_facts=10, cache=None) -> RxNavFactsSource:
    return RxNavFactsSource(rx, cache or InMemoryCache(), max_facts)


def texts(facts: list[RxFact]) -> list[str]:
    return [f.text for f in facts]


async def test_lines_for_one_ingredient_are_short_and_deterministic(rx):
    facts = await source(rx).facts(["17767"])
    assert texts(facts) == [
        "RxNorm brand names for amlodipine: Amodip, Norvasc.",
        "RxNorm dose forms for amlodipine: Oral Capsule, Oral Tablet.",
        "RxNorm strengths for amlodipine: 1 MG/ML, 2.5 MG, 5 MG, 10 MG.",
    ]
    assert await source(rx).facts(["17767"]) == facts  # same input, same output


async def test_every_fact_names_the_ingredient_it_belongs_to(rx):
    facts = await source(rx).facts(["17767", "29046"])
    assert {f.rxcui for f in facts if "amlodipine" in f.text} == {"17767"}
    assert {f.rxcui for f in facts if "lisinopril" in f.text} == {"29046"}
    assert all(isinstance(f, RxFact) for f in facts)


async def test_combination_brands_are_not_listed_for_one_ingredient(rx):
    brand_line = texts(await source(rx).facts(["17767"]))[0]
    assert "Tribenzor" not in brand_line
    assert "Exforge" not in brand_line
    assert "Orphan" not in brand_line


async def test_a_cold_ingredient_costs_one_rxnav_call(rx):
    # The old design checked each brand with its own call (up to 20 per ingredient, NFR-1).
    await source(rx).facts(["17767"])
    assert rx.calls == [("related", "17767")]
    rx.calls.clear()
    await source(rx).facts(["17767", "29046"])
    assert rx.names_called == ["related", "related"]


def test_single_ingredient_brands_reads_the_brand_tag_and_the_ingredient_separator():
    drugs = concepts(
        "SBD",
        ("1", "a 5 MG / b 10 MG Oral Tablet [Combo]"),
        ("2", "a 5 MG Oral Tablet [Solo]"),
        ("3", "a 5 MG / b 10 MG Oral Tablet [Mixed]"),
        ("4", "a 1 MG/ML Oral Solution [Mixed]"),  # Mixed also has a single-ingredient product
        ("5", "no brand tag here"),
    )
    assert single_ingredient_brands(drugs) == {"solo", "mixed"}


async def test_strengths_sort_by_number_not_by_text(rx):
    # Text order would put "10 MG" before "2.5 MG".
    assert "2.5 MG, 5 MG, 10 MG" in texts(await source(rx).facts(["17767"]))[2]


async def test_at_most_max_facts_lines_in_total(rx):
    assert len(await source(rx, max_facts=2).facts(["17767"])) == 2
    assert await source(rx, max_facts=0).facts(["17767"]) == []


async def test_a_cut_list_keeps_one_line_per_ingredient(rx):
    # Interleaved, so a small max_facts does not drop the second drug (multi-drug questions).
    facts = await source(rx, max_facts=2).facts(["17767", "29046"])
    assert [(f.rxcui, f.text) for f in facts] == [
        ("17767", "RxNorm brand names for amlodipine: Amodip, Norvasc."),
        ("29046", "RxNorm brand names for lisinopril: Zestril."),
    ]


async def test_two_ingredients_fit_under_the_default_cap(rx):
    facts = await source(rx).facts(["17767", "29046"])
    assert len(facts) == 6
    assert facts[0].text.endswith("Amodip, Norvasc.")
    assert facts[1].text.endswith("Zestril.")


async def test_duplicate_and_empty_rxcuis_are_ignored(rx):
    assert await source(rx).facts([]) == []
    assert len(await source(rx).facts(["17767", "17767", ""])) == 3


async def test_brand_list_is_capped(rx):
    names = [(str(1000 + i), f"Brand{i:02d}") for i in range(15)]
    rx.related_map["17767"].brands = concepts("BN", *names)
    rx.related_map["17767"].branded_drugs = concepts(
        "SBD",
        *((str(2000 + i), f"amlodipine 5 MG Oral Tablet [{n}]") for i, (_, n) in enumerate(names)),
    )
    brand_line = texts(await source(rx).facts(["17767"]))[0]
    assert brand_line.count("Brand") == 8


async def test_result_is_cached_for_24h_under_the_spec_key(rx):
    cache = InMemoryCache()
    first = await source(rx, cache=cache).facts(["17767"])
    raw = await cache.get("rx:facts:17767")
    assert raw is not None
    assert json.loads(raw) == texts(first)  # JSON bytes, one list of lines per ingredient

    rx.calls.clear()
    assert await source(rx, cache=cache).facts(["17767"]) == first
    assert rx.calls == []  # served from the cache, RxNav untouched


async def test_cache_ttl_is_24_hours(rx):
    now = [0.0]
    cache = InMemoryCache(clock=lambda: now[0])
    await source(rx, cache=cache).facts(["17767"])
    now[0] = 24 * 3600 - 1
    assert await cache.get("rx:facts:17767") is not None
    now[0] = 24 * 3600
    assert await cache.get("rx:facts:17767") is None


async def test_a_corrupt_cache_entry_is_a_miss(rx):
    cache = InMemoryCache()
    await cache.set("rx:facts:17767", b"not json", 60)
    assert len(await source(rx, cache=cache).facts(["17767"])) == 3


async def test_rxnav_down_gives_no_facts_and_caches_nothing(rx):
    cache = InMemoryCache()
    rx.down = True
    assert await source(rx, cache=cache).facts(["17767"]) == []
    assert await cache.get("rx:facts:17767") is None
    rx.down = False  # the gap is not remembered: the next request retries
    assert len(await source(rx, cache=cache).facts(["17767"])) == 3


async def test_one_ingredient_down_does_not_hide_the_cached_other(rx):
    cache = InMemoryCache()
    await source(rx, cache=cache).facts(["29046"])
    rx.down = True
    facts = await source(rx, cache=cache).facts(["17767", "29046"])
    assert facts and all("lisinopril" in f.text for f in facts)


async def test_an_unknown_ingredient_has_no_facts(rx):
    assert await source(rx).facts(["999999"]) == []

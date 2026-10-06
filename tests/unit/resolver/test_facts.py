import json

import pytest

from pharmarag.cache import InMemoryCache
from pharmarag.resolver.facts import RxNavFactsSource
from pharmarag.resolver.rxnav_client import Concept, Ingredient, RelatedConcepts
from tests.unit.resolver.fakes import FakeRx

AML = Ingredient("17767", "amlodipine")
OLM = Ingredient("118463", "olmesartan")
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
            ("1372732", "Tribenzor"),  # a combination product: not a brand of amlodipine alone
            ("58927", "Norvasc"),
            ("2748470", "Amodip"),
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
    fake.ingredient_map["1372732"] = [AML, OLM, Ingredient("5487", "hydrochlorothiazide")]
    fake.ingredient_map["58927"] = [AML]
    fake.ingredient_map["2748470"] = [AML]
    fake.related_map["29046"] = RelatedConcepts(
        ingredient=LIS,
        brands=concepts("BN", ("104377", "Zestril")),
        dose_forms=concepts("DF", ("317541", "Oral Tablet")),
        strengths=concepts("SCDC", ("197884", "lisinopril 10 MG")),
    )
    fake.ingredient_map["104377"] = [LIS]
    return fake


def source(rx, max_facts=10, cache=None) -> RxNavFactsSource:
    return RxNavFactsSource(rx, cache or InMemoryCache(), max_facts)


async def test_lines_for_one_ingredient_are_short_and_deterministic(rx):
    lines = await source(rx).facts(["17767"])
    assert lines == [
        "RxNorm brand names for amlodipine: Amodip, Norvasc.",
        "RxNorm dose forms for amlodipine: Oral Capsule, Oral Tablet.",
        "RxNorm strengths for amlodipine: 1 MG/ML, 2.5 MG, 5 MG, 10 MG.",
    ]
    assert await source(rx).facts(["17767"]) == lines  # same input, same output


async def test_a_combination_products_brand_is_not_listed_for_one_ingredient(rx):
    lines = await source(rx).facts(["17767"])
    assert "Tribenzor" not in lines[0]


async def test_strengths_sort_by_number_not_by_text(rx):
    # Text order would put "10 MG" before "2.5 MG".
    assert "2.5 MG, 5 MG, 10 MG" in (await source(rx).facts(["17767"]))[2]


async def test_at_most_max_facts_lines_in_total(rx):
    assert len(await source(rx, max_facts=2).facts(["17767"])) == 2
    assert await source(rx, max_facts=0).facts(["17767"]) == []


async def test_a_cut_list_keeps_one_line_per_ingredient(rx):
    # Interleaved, so a small max_facts does not drop the second drug (multi-drug questions).
    lines = await source(rx, max_facts=2).facts(["17767", "29046"])
    assert lines == [
        "RxNorm brand names for amlodipine: Amodip, Norvasc.",
        "RxNorm brand names for lisinopril: Zestril.",
    ]


async def test_two_ingredients_fit_under_the_default_cap(rx):
    lines = await source(rx).facts(["17767", "29046"])
    assert len(lines) == 6
    assert lines[0].endswith("Amodip, Norvasc.")
    assert lines[1].endswith("Zestril.")


async def test_duplicate_and_empty_rxcuis_are_ignored(rx):
    assert await source(rx).facts([]) == []
    assert len(await source(rx).facts(["17767", "17767", ""])) == 3


async def test_brand_list_is_capped(rx):
    rx.related_map["17767"].brands = concepts(
        "BN", *((str(1000 + i), f"Brand{i:02d}") for i in range(15))
    )
    for i in range(15):
        rx.ingredient_map[str(1000 + i)] = [AML]
    brand_line = (await source(rx).facts(["17767"]))[0]
    assert brand_line.count("Brand") == 8


async def test_result_is_cached_for_24h_under_the_spec_key(rx):
    cache = InMemoryCache()
    first = await source(rx, cache=cache).facts(["17767"])
    raw = await cache.get("rx:facts:17767")
    assert raw is not None
    assert json.loads(raw) == first  # JSON bytes, one list of lines per ingredient

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
    lines = await source(rx, cache=cache).facts(["17767", "29046"])
    assert lines and all("lisinopril" in x for x in lines)


async def test_an_unknown_ingredient_has_no_facts(rx):
    assert await source(rx).facts(["999999"]) == []

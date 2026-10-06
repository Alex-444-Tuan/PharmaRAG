"""Filter-leak test (spec section 13, FR-3) against the real ingested corpus.

For every (drug, question) pair that resolves, filtered dense and lexical retrieval must return
chunks of that ingredient key and no other. The drugs are every seed name, plus misspellings and
brands of every look-alike pair member. Resolution runs in local-only mode: RxNav is stubbed as
unavailable, so the test is deterministic and offline. FakeEmbedder vectors are enough because
the property is about keys, not about ranking quality.

A second property: resolving a misspelling never acts on the look-alike partner (or any other
drug). Asking for clarification or not recognising the name is fine; acting on the wrong drug
is not.

The corpus is read-only here: the pool opens every connection in read-only mode.
"""

import csv

import pytest

from pharmarag.app.settings import REPO_ROOT
from pharmarag.cache import InMemoryCache
from pharmarag.db import create_pool
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.local import PgLocalStore
from pharmarag.retrieval.dense import dense_search
from pharmarag.retrieval.lexical import lexical_search
from tests.unit.resolver.fakes import FakeRx

pytestmark = pytest.mark.corpus

# One question per template. Some carry a section filter (LOINC), as analyze would set.
QUESTIONS = [
    ("What is the usual adult dose of {m}?", ["34068-7"]),
    ("What are the contraindications of {m}?", ["34070-3"]),
    ("Which drugs interact with {m}?", ["34073-7"]),
    ("What adverse reactions are reported with {m}?", None),
    ("Is {m} safe in pregnancy or renal impairment?", None),
]

# Two realistic misspellings (dropped letter, vowel swap, phonetic) of each of the 40 members
# of the 20 look-alike pairs in data/seed_ingredients.csv.
MISSPELLINGS = {
    "hydralazine": ["hydralazin", "hydralizine"],
    "hydroxyzine": ["hydroxyzin", "hydroxizine"],
    "clonidine": ["clonidin", "klonidine"],
    "clonazepam": ["clonazepan", "clonazipam"],
    "bupropion": ["bupropian", "buproprion"],
    "buspirone": ["buspiron", "bispirone"],
    "carbamazepine": ["carbamazepin", "carbemazepine"],
    "oxcarbazepine": ["oxcarbazepin", "oxcarbamazepine"],
    "glipizide": ["glipizid", "glypizide"],
    "glyburide": ["glyburid", "glyberide"],
    "prednisone": ["prednizone", "predisone"],
    "prednisolone": ["prednisolon", "prednisalone"],
    "metformin": ["metformn", "metforman"],
    "metronidazole": ["metronidazol", "metrodinazole"],
    "tramadol": ["tramadal", "tramadoll"],
    "trazodone": ["trazadone", "trazodon"],
    "nicardipine": ["nicardipin", "nicardapine"],
    "nifedipine": ["nifedipin", "nifedepine"],
    "risperidone": ["risperidon", "risperadone"],
    "ropinirole": ["ropinerole", "ropinirol"],
    "lamotrigine": ["lamotrigin", "lamotragine"],
    "lamivudine": ["lamivudin", "lamivuddine"],
    "fluoxetine": ["fluoxetin", "flouxetine"],
    "paroxetine": ["paroxetin", "paroxitine"],
    "alprazolam": ["alprazalam", "alprazolm"],
    "lorazepam": ["lorazepan", "lorazapam"],
    "amiloride": ["amilorid", "amiloreed"],
    "amlodipine": ["amlodipin", "amlodapine"],
    "azathioprine": ["azathioprin", "azathiaprine"],
    "azacitidine": ["azacitadine", "azacytidine"],
    "tizanidine": ["tizanidin", "tizanadine"],
    "tiagabine": ["tiagabin", "tiagabene"],
    "sitagliptin": ["sitagliptn", "sitagliptine"],
    "sumatriptan": ["sumatriptin", "sumatripan"],
    "medroxyprogesterone": ["medroxyprogestrone", "medroxyprogesteron"],
    "methylprednisolone": ["methylprednisolon", "methylprednisalone"],
    "clomiphene": ["clomiphine", "clomephene"],
    "clomipramine": ["clomipramin", "clomipraminne"],
    "quinidine": ["quinidin", "quinadine"],
    "quinine": ["quinin", "quinnine"],
}


def seed_pairs() -> dict[str, str]:
    """name -> look-alike partner, from the seed CSV's lookalike_group column."""
    groups: dict[str, list[str]] = {}
    with (REPO_ROOT / "data" / "seed_ingredients.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lookalike_group"].strip():
                groups.setdefault(row["lookalike_group"].strip(), []).append(row["name"].strip())
    pairs: dict[str, str] = {}
    for members in groups.values():
        assert len(members) == 2
        pairs[members[0]], pairs[members[1]] = members[1], members[0]
    return pairs


# Seed names that RxNorm spells differently, so they are not an `ingredient.name`. They still
# resolve (by alias or approximately) to the corpus ingredient below.
RENAMED_SEEDS = {
    "insulin aspart": "51428",  # RxNorm: "insulin aspart, human"
    "divalproex": "40254",  # RxNorm ingredient: valproate
    "isosorbide mononitrate": "6057",  # RxNorm ingredient: isosorbide
}


def seed_names() -> list[str]:
    with (REPO_ROOT / "data" / "seed_ingredients.csv").open(newline="") as f:
        return [row["name"].strip() for row in csv.DictReader(f) if row["name"].strip()]


@pytest.fixture
async def pool(corpus_url):
    # Read-only at the session level: this test must not be able to change the corpus.
    p = await create_pool(corpus_url + "?options=-c%20default_transaction_read_only%3Don")
    yield p
    await p.close()


@pytest.fixture
def resolver(pool, thresholds):
    down = FakeRx()
    down.down = True  # RxNav unavailable: alias table and trigram only (NFR-4)
    return CascadeResolver(PgLocalStore(pool), down, thresholds.resolver, InMemoryCache())


@pytest.fixture
async def key_of_name(pool) -> dict[str, str]:
    """Lowercase name -> ingredient key, for every seed name and every look-alike member."""
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT lower(name), rxcui FROM ingredient")
        known = {n: k for n, k in await cur.fetchall()}
    return {n.lower(): known.get(n.lower()) or RENAMED_SEEDS[n] for n in seed_names()}


async def brands_of(pool, key: str) -> list[str]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT alias FROM drug_alias WHERE ingredient_key = %s AND alias_tty = 'BN' "
            "ORDER BY alias LIMIT 1",
            (key,),
        )
        return [r[0] for r in await cur.fetchall()]


async def search_filtered(pool, embedder, question, sections, key):
    async with pool.connection() as conn:
        dense = await dense_search(
            conn,
            embedder.embed_query(question),
            ingredient_key=key,
            sections=sections,
            top_k=20,
            attempt=1,
        )
        lexical = await lexical_search(
            conn, question, ingredient_key=key, sections=sections, top_k=20, attempt=1
        )
    return dense, lexical


async def test_resolved_questions_never_return_another_drugs_chunks(
    pool, resolver, fake_embedder, key_of_name
):
    mentions: list[tuple[str, str | None]] = [(n, key_of_name[n.lower()]) for n in seed_names()]
    pairs = seed_pairs()
    for name in pairs:
        for variant in MISSPELLINGS[name]:
            mentions.append((variant, key_of_name[name]))
        for brand in await brands_of(pool, key_of_name[name]):
            mentions.append((brand, key_of_name[name]))

    checked = leaked = dense_hits = lexical_hits = 0
    for i, (mention, gold_key) in enumerate(mentions):
        (res,) = await resolver.resolve([mention])
        if res.status != "resolved":
            continue
        assert res.ingredient_key == gold_key, f"{mention!r} resolved to the wrong drug"
        template, sections = QUESTIONS[i % len(QUESTIONS)]
        question = template.format(m=mention)
        dense, lexical = await search_filtered(
            pool, fake_embedder, question, sections, res.ingredient_key
        )
        checked += 1
        dense_hits += len(dense)
        lexical_hits += len(lexical)
        stray = [h for h in dense + lexical if h.ingredient_key != res.ingredient_key]
        leaked += len(stray)
        assert not stray, f"{mention!r} resolved to {res.ingredient_key}, got {stray[0]}"
        if not sections:  # an unfiltered section can only return rows if the drug has chunks
            assert dense, f"{mention!r}: filtered dense search returned nothing"

    assert checked >= 200
    assert leaked == 0
    # The property is not vacuous: the searches did return rows.
    assert dense_hits >= 10 * checked * 0.5
    assert lexical_hits > 0


async def test_the_test_can_fail_unfiltered_search_does_return_other_drugs(pool, fake_embedder):
    """Negative control. If this ever returned only one key, the test above would prove nothing."""
    foreign = 0
    for template, sections in QUESTIONS:
        question = template.format(m="hydroxyzine")
        async with pool.connection() as conn:
            dense = await dense_search(
                conn,
                fake_embedder.embed_query(question),
                ingredient_key=None,
                sections=sections,
                top_k=20,
                attempt=1,
            )
            lexical = await lexical_search(
                conn, question, ingredient_key=None, sections=sections, top_k=20, attempt=1
            )
        foreign += len({h.ingredient_key for h in dense + lexical})
    assert foreign > len(QUESTIONS)  # several distinct drugs per search


async def test_every_seed_name_resolves_to_itself_in_local_only_mode(resolver, key_of_name):
    names = seed_names()
    assert len(names) == 200
    out = await resolver.resolve_detailed(names)
    wrong = [
        (n, o.resolution.status, o.resolution.ingredient_key)
        for n, o in zip(names, out, strict=True)
        if (o.resolution.status, o.resolution.ingredient_key)
        != ("resolved", key_of_name[n.lower()])
    ]
    assert not wrong
    # An exact name is an exact match, apart from the three that RxNorm spells differently.
    inexact = {n for n, o in zip(names, out, strict=True) if o.resolution.match != "exact"}
    assert inexact <= set(RENAMED_SEEDS)
    # Exact alias hits never need RxNav: only the renamed seeds fall through to the fallback.
    assert {n for n, o in zip(names, out, strict=True) if o.degraded} <= set(RENAMED_SEEDS)


async def test_a_misspelling_never_resolves_to_the_look_alike_partner(resolver, key_of_name):
    """Clarifying or not recognising is fine. Acting on another drug is not."""
    pairs = seed_pairs()
    assert len(pairs) == 40  # 20 pairs, both members
    acted_on_other: list[tuple[str, str, str]] = []
    outcomes = {"resolved": 0, "ambiguous": 0, "unknown": 0}
    for name, partner in pairs.items():
        gold, partner_key = key_of_name[name], key_of_name[partner]
        for variant in MISSPELLINGS[name]:
            (res,) = await resolver.resolve([variant])
            outcomes[res.status] += 1
            assert res.ingredient_key != partner_key, f"{variant!r} resolved to partner {partner}"
            if res.status in ("resolved", "not_covered") and res.ingredient_key != gold:
                acted_on_other.append((variant, name, str(res.ingredient_key)))
    assert not acted_on_other
    # Most misspellings are still served; the gate is not simply refusing everything.
    assert outcomes["resolved"] >= 60, outcomes


async def test_brand_names_of_look_alike_members_resolve_to_their_own_drug(
    pool, resolver, key_of_name
):
    seen = 0
    for name in seed_pairs():
        for brand in await brands_of(pool, key_of_name[name]):
            (res,) = await resolver.resolve([brand])
            assert (res.status, res.match, res.ingredient_key) == (
                "resolved",
                "brand",
                key_of_name[name],
            ), brand
            seen += 1
    assert seen >= 25

"""The committed seed list: size, uniqueness, and look-alike pairs."""

import csv
from collections import defaultdict

from pharmarag.app.settings import REPO_ROOT

SEED = REPO_ROOT / "data" / "seed_ingredients.csv"


def rows() -> list[dict[str, str]]:
    with SEED.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["name", "lookalike_group"]
        return list(reader)


def groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for r in rows():
        if r["lookalike_group"]:
            out[r["lookalike_group"]].append(r["name"])
    return dict(out)


def test_seed_has_200_unique_single_ingredient_names():
    names = [r["name"] for r in rows()]
    assert len(names) == 200
    assert len({n.lower() for n in names}) == 200
    assert all(n == n.strip() and "/" not in n and "," not in n for n in names)


def test_lookalike_groups_are_complete_pairs_and_there_are_at_least_15():
    found = groups()
    assert len(found) >= 15
    for group, members in found.items():
        assert len(members) == 2, group  # both members of every pair are present
        assert group == "-".join(sorted(members))


def test_the_pairs_named_in_the_spec_are_present():
    pairs = {frozenset(v) for v in groups().values()}
    assert frozenset({"hydroxyzine", "hydralazine"}) in pairs
    assert frozenset({"clonidine", "clonazepam"}) in pairs

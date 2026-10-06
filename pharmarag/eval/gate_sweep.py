"""Dev-set evidence for the decision gate thresholds (spec section 7, "tuned on dev only").

    DATABASE_URL=.../pharmarag_dev uv run python -m pharmarag.eval.gate_sweep

The 20 dev queries are too few to tune three thresholds on. This script makes many more dev
queries: every single-edit variant (a deleted letter, two swapped letters, a changed vowel, a
doubled letter) of the 40 members of the 20 look-alike pairs, and applies a grid of gate settings
to each one. For each setting it counts how many variants end as correct, clarified (gold among
the candidates), not recognised, or WRONG (resolved to another drug).

The variants come from the seed names, never from the frozen test file. Variants that are
themselves a known alias are skipped: they are valid names of something, not misspellings.
The cascade runs local-only (alias table and trigram), so the result is deterministic and offline.
RxNav would add candidates, never remove them.
"""

from __future__ import annotations

import asyncio
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

from pharmarag.app.settings import REPO_ROOT, ResolverThresholds, get_settings, load_thresholds
from pharmarag.cache import InMemoryCache
from pharmarag.db import create_pool
from pharmarag.eval.report import base_meta, corpus_version, write_report
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.gate import KeyScore, decide
from pharmarag.resolver.local import PgLocalStore

VOWELS = "aeiouy"
MIN_LENGTH = 5

RESOLVED_MIN = (0.85, 0.88, 0.90, 0.92, 0.95)
MARGIN = (0.02, 0.03, 0.05, 0.07, 0.10)
AMBIGUOUS_MIN = (0.75, 0.80, 0.85)


def edit_variants(word: str) -> list[str]:
    """Single-edit misspellings of `word`, sorted, at least MIN_LENGTH letters long."""
    out: set[str] = set()
    for i, ch in enumerate(word):
        out.add(word[:i] + word[i + 1 :])  # deleted letter
        if i + 1 < len(word) and ch != word[i + 1]:
            out.add(word[:i] + word[i + 1] + ch + word[i + 2 :])  # swapped neighbours
        if ch in VOWELS:
            out.update(word[:i] + v + word[i + 1 :] for v in VOWELS if v != ch)  # other vowel
        out.add(word[:i] + ch + word[i:])  # doubled letter
    out.discard(word)
    return sorted(v for v in out if len(v) >= MIN_LENGTH)


def lookalike_members(seed: Path) -> list[str]:
    with seed.open(newline="") as f:
        return [r["name"].strip() for r in csv.DictReader(f) if r["lookalike_group"].strip()]


def grid() -> list[tuple[float, float, float]]:
    return list(itertools.product(RESOLVED_MIN, MARGIN, AMBIGUOUS_MIN))


def tally(cases, base: ResolverThresholds, setting: tuple[float, float, float]) -> dict[str, int]:
    rm, mg, am = setting
    th = base.model_copy(update={"resolved_min": rm, "margin": mg, "ambiguous_min": am})
    counts: Counter[str] = Counter()
    for gold, cascaded in cases:
        v = decide([KeyScore(s.key, s.score) for s in cascaded.scored], th, exact=False)
        if v.outcome == "resolved":
            counts["correct" if v.top.key == gold else "wrong"] += 1
        elif v.outcome == "ambiguous":
            counts[
                "clarified" if any(o.key == gold for o in v.candidates) else "clarify_no_gold"
            ] += 1
        else:
            counts["not_recognised"] += 1
    return {
        k: counts.get(k, 0)
        for k in ("correct", "clarified", "clarify_no_gold", "not_recognised", "wrong")
    }


async def run(database_url: str | None = None) -> dict:
    settings = get_settings()
    thresholds = load_thresholds(settings.thresholds_path)
    pool = await create_pool(database_url or settings.database_url)
    try:
        resolver = CascadeResolver(PgLocalStore(pool), None, thresholds.resolver, InMemoryCache())
        async with pool.connection() as conn:
            known = {
                n: k
                for n, k in await (
                    await conn.execute("SELECT lower(name), rxcui FROM ingredient")
                ).fetchall()
            }
            aliases = {
                a for (a,) in await (await conn.execute("SELECT alias FROM drug_alias")).fetchall()
            }
            report = base_meta(
                eval_name="gate_sweep",
                split="dev",
                mode="local-only",
                runs=1,
                model_id=None,
                corpus=await corpus_version(conn),
                thresholds=thresholds,
                dataset=REPO_ROOT / "data" / "seed_ingredients.csv",
            )
        cases = []
        for member in lookalike_members(REPO_ROOT / "data" / "seed_ingredients.csv"):
            for variant in edit_variants(member.lower()):
                if variant not in aliases:
                    cases.append((known[member.lower()], await resolver.cascade(variant)))
    finally:
        await pool.close()
    current = (
        thresholds.resolver.resolved_min,
        thresholds.resolver.margin,
        thresholds.resolver.ambiguous_min,
    )
    report |= {
        "resolver_only": True,
        "n_queries": len(cases),
        "current": list(current),
        "grid": [
            {
                "resolved_min": s[0],
                "margin": s[1],
                "ambiguous_min": s[2],
                **tally(cases, thresholds.resolver, s),
            }
            for s in grid()
        ],
    }
    return report


def render_markdown(report: dict) -> str:
    cur = tuple(report["current"])
    lines = [
        "# Decision gate sweep, dev evidence",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- corpus_version: {report['corpus_version']}",
        f"- thresholds commit: {report['thresholds_commit']} (dirty: {report['thresholds_dirty']})",
        f"- Variants: {report['n_queries']} single-edit misspellings of the 40 look-alike members, "
        "local-only cascade",
        f"- Current setting (resolved_min, margin, ambiguous_min): {cur}",
        "",
        "| resolved_min | margin | ambiguous_min | correct | clarified | clarify w/o gold | "
        "not recognised | WRONG |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in report["grid"]:
        key = (g["resolved_min"], g["margin"], g["ambiguous_min"])
        mark = " (current)" if key == cur else ""
        lines.append(
            f"| {key[0]} | {key[1]}{mark} | {key[2]} | {g['correct']} | {g['clarified']} | "
            f"{g['clarify_no_gold']} | {g['not_recognised']} | {g['wrong']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = asyncio.run(run())
    json_path, md_path = write_report(report, REPO_ROOT / "reports", render_markdown(report))
    print(f"wrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

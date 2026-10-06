"""Run the look-alike eval and build its report dict and Markdown summary (spec section 11.1).

Both entry points take their resources as arguments (resolver, embedder, `ask` function, database
connection), so tests can run them against FakeLLM and a stub RxNav. `pharmarag.eval.run` builds
the real ones.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import mean

from psycopg import AsyncConnection

from pharmarag.app.settings import Thresholds
from pharmarag.eval import lookalike as la
from pharmarag.eval.report import base_meta, corpus_version
from pharmarag.models.embedder import Embedder
from pharmarag.resolver.base import Resolver

PROXY_NOTE = (
    "PROXY, not the AT-1 number. This baseline runs unfiltered dense and lexical retrieval on the "
    "question, fuses the lists, and checks which drug the top evidence belongs to. No answer is "
    "written, so it does not say whether a writer would have cited that chunk."
)


def _spread(values: Sequence[int]) -> dict:
    return {"values": list(values), "min": min(values), "max": max(values), "mean": mean(values)}


def _arm_summary(per_run_rows: Sequence[Sequence], outcome_attr: str = "outcome") -> dict:
    runs = []
    for rows in per_run_rows:
        runs.append(
            {
                "counts": la.counts(getattr(r, outcome_attr) for r in rows),
                "rows": la.rows_as_dicts(rows),
            }
        )
    mis = [r["counts"]["misinterpreted"] for r in runs]
    clarified = [r["counts"][la.CLARIFIED] for r in runs]
    return {
        "runs": runs,
        "misinterpreted": _spread(mis),
        "clarified": _spread(clarified),
    }


def _by_kind(rows: Sequence[la.Row]) -> dict[str, int]:
    c = Counter(r.kind for r in rows)
    return {k: c.get(k, 0) for k in la.KINDS}


async def run_resolver_only(
    *,
    split: str,
    mode: str,
    runs: int,
    resolver_factory,
    conn: AsyncConnection,
    embedder: Embedder | None,
    thresholds: Thresholds,
    rxnav_mode: str,
) -> dict:
    """No LLM. System arm: the resolver alone on each row's `mention`. Baseline arm: the proxy.

    `resolver_factory()` returns a fresh resolver (and so a fresh cache) for every run.
    """
    rows = la.load_dataset(split)
    report = base_meta(
        eval_name="lookalike",
        split=split,
        mode=mode,
        runs=runs,
        model_id=None,
        corpus=await corpus_version(conn),
        thresholds=thresholds,
        dataset=la.dataset_path(split),
    )
    report |= {"resolver_only": True, "rxnav": rxnav_mode, "n_queries": len(rows)}
    report["kinds"] = _by_kind(rows)

    if mode in ("system", "both"):
        per_run = []
        for _ in range(runs):
            resolver: Resolver = resolver_factory()
            per_run.append(await la.run_resolver_system(rows, resolver))
        summary = _arm_summary(per_run)
        summary["degraded_rows"] = _spread([sum(r.degraded for r in rs) for rs in per_run])
        report["system"] = summary

    if mode in ("baseline", "both"):
        if embedder is None:
            raise ValueError("the baseline proxy needs an embedder")
        proxy = await la.run_baseline_proxy(rows, conn, embedder, thresholds.retrieval)
        # Deterministic given the corpus and the model, so it runs once whatever --runs says.
        report["baseline_proxy"] = {
            "note": PROXY_NOTE,
            "embedding_model": embedder.model_name,
            "counts": la.proxy_counts(proxy),
            "rows": la.rows_as_dicts(proxy),
        }
    return report


async def run_full(
    *,
    split: str,
    mode: str,
    runs: int,
    ask: la.AskFn,
    conn: AsyncConnection,
    model_id: str,
    thresholds: Thresholds,
    resolver_available: bool = True,
) -> dict:
    """Whole graph through `run_ask`, N runs per arm. Needs an LLM behind `ask`.

    `resolver_available` is `deps.resolver_available`. When it is False `run_ask` quietly turns
    the resolver flag off, so a "system" arm would measure the baseline twice. That is an error.
    """
    if not resolver_available and mode in ("system", "both"):
        raise ValueError(
            "the system arm needs a resolver: deps.resolver_available is False, so run_ask would "
            "run it without one"
        )
    rows = la.load_dataset(split)
    report = base_meta(
        eval_name="lookalike",
        split=split,
        mode=mode,
        runs=runs,
        model_id=model_id,
        corpus=await corpus_version(conn),
        thresholds=thresholds,
        dataset=la.dataset_path(split),
    )
    report |= {"resolver_only": False, "n_queries": len(rows), "kinds": _by_kind(rows)}
    arms = [a for a in ("baseline", "system") if mode in (a, "both")]
    results: dict[str, list] = {a: [] for a in arms}
    for _ in range(runs):
        for arm in arms:  # interleaved, so drift in the LLM affects both arms alike
            results[arm].append(await la.run_full_arm(rows, arm, ask, conn))
    for arm in arms:
        report[arm] = _arm_summary(results[arm])
    if "baseline" in report and "system" in report:
        b, s = report["baseline"]["misinterpreted"], report["system"]["misinterpreted"]
        report["metric"] = {
            "definition": "(B - S) / B with B, S the misinterpreted counts of baseline, system",
            "per_run": [
                la.reduction(bv, sv) for bv, sv in zip(b["values"], s["values"], strict=True)
            ],
            "of_means": la.reduction(b["mean"], s["mean"]),
            "baseline_misinterpreted": b["values"],
            "system_misinterpreted": s["values"],
            "system_clarified": report["system"]["clarified"]["values"],
        }
    return report


# --- Markdown ---


def _row_table(rows: list[dict], cols: Sequence[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return out


def render_markdown(report: dict) -> str:
    m = report
    lines = [
        f"# Look-alike eval, {m['split']} split, "
        f"{'resolver-only' if m['resolver_only'] else 'full graph'}",
        "",
        f"- Timestamp: {m['timestamp']}",
        f"- Model ID: {m['model_id'] or 'none (resolver-only, no LLM)'}",
        f"- corpus_version: {m['corpus_version']}",
        f"- config/thresholds.yaml commit: {m['thresholds_commit']} "
        f"(uncommitted changes: {m['thresholds_dirty']})",
        f"- Runs: {m['runs']}   Queries: {m['n_queries']}   Kinds: {m['kinds']}",
        f"- Dataset: {m['dataset']['file']} sha256 {m['dataset']['sha256'][:16]}",
        f"- Resolver thresholds: {m['thresholds']['resolver']}",
    ]
    if m["resolver_only"]:
        lines.append(f"- RxNav: {m['rxnav']}")
    lines.append("")

    if "system" in m:
        sysd = m["system"]
        last = sysd["runs"][-1]
        lines += [
            "## System" + (" (resolver only)" if m["resolver_only"] else ""),
            "",
            "| outcome | count (last run) |",
            "|---|---|",
            *(f"| {o} | {last['counts'][o]} |" for o in la.OUTCOMES),
            f"| **misinterpreted** | **{last['counts']['misinterpreted']}** of "
            f"{last['counts']['total']} |",
            "",
            f"Across {m['runs']} run(s): misinterpreted {sysd['misinterpreted']['values']}, "
            f"clarified {sysd['clarified']['values']}.",
        ]
        if "degraded_rows" in sysd:
            lines.append(
                f"Rows resolved in degraded mode (RxNav down): {sysd['degraded_rows']['values']}."
            )
        lines.append("")
        if m["resolver_only"]:
            cols = ["mention", "kind", "outcome", "status", "match", "resolved_name", "score"]
            lines += ["### Rows (last run)", "", *_row_table(last["rows"], cols), ""]

    if "baseline" in m:
        base = m["baseline"]
        last = base["runs"][-1]
        lines += [
            "## Baseline (resolver off)",
            "",
            f"Misinterpreted per run: {base['misinterpreted']['values']} "
            f"of {last['counts']['total']}.",
            "",
        ]

    if "baseline_proxy" in m:
        p = m["baseline_proxy"]
        c = p["counts"]
        lines += [
            "## Baseline proxy",
            "",
            f"**{p['note']}**",
            "",
            f"- Embedding model: {p['embedding_model']}",
            f"- Top evidence chunk belongs to another drug: **{c['top1_wrong']}** of {c['total']}",
            f"- Any of the top 8 evidence chunks belongs to another drug: "
            f"**{c['any_top8_wrong']}** of {c['total']}",
            f"- No evidence returned: {c['no_evidence']}",
            "",
        ]

    if "metric" in m:
        x = m["metric"]
        lines += [
            "## AT-1 metric",
            "",
            f"{x['definition']}",
            "",
            f"- Baseline misinterpreted per run: {x['baseline_misinterpreted']}",
            f"- System misinterpreted per run: {x['system_misinterpreted']}",
            f"- System clarified per run (counted separately): {x['system_clarified']}",
            f"- Reduction per run: {x['per_run']}",
            f"- Reduction of the mean counts: {x['of_means']}",
            "",
        ]
    return "\n".join(lines)

"""Evaluation CLI (spec section 11).

    uv run python -m pharmarag.eval.run lookalike --split dev --mode both --runs 3
    uv run python -m pharmarag.eval.run lookalike --split dev --mode both --resolver-only

`--resolver-only` needs no LLM and no API key (RxNav is called unless `--offline`). Without it the
whole graph runs through `run_ask` with EVAL_MODE on and needs the LLM_* settings.

The `grounding` eval (hallucinated claims, spec 11.2) has its own actions; see grounding_cli.py.

CI gates (spec section 13, "Eval in CI"; rules in ci_gates.py). Each exits non-zero below target:

    ... lookalike --split test --mode system --resolver-only --offline --ci
    ... grounding --score --ci
    ... latency --ci                    # replay, real verifier, 10 questions x 3 reps

Reports go to reports/lookalike_<split>_<timestamp>.json with a .md summary beside it.
The database is DATABASE_URL (use .../pharmarag_dev for the ingested corpus).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pharmarag.app.settings import REPO_ROOT, Settings, get_settings, load_thresholds
from pharmarag.cache import InMemoryCache
from pharmarag.db import connect, create_pool
from pharmarag.eval import ci_gates, grounding_cli, lookalike, lookalike_run
from pharmarag.eval.report import write_report
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.local import PgLocalStore
from pharmarag.resolver.rxnav_client import RxNavClient

log = logging.getLogger("pharmarag.eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m pharmarag.eval.run")
    sub = p.add_subparsers(dest="eval", required=True)
    la = sub.add_parser("lookalike", help="look-alike misinterpretation (AT-1, spec 11.1)")
    la.add_argument("--split", choices=["dev", "test"], required=True)
    la.add_argument("--mode", choices=["baseline", "system", "both"], default="both")
    la.add_argument("--runs", type=int, default=1, help="repeats of the LLM-dependent arms")
    la.add_argument(
        "--resolver-only",
        action="store_true",
        help="no LLM: resolver on the dataset's mention, baseline as a retrieval proxy",
    )
    la.add_argument(
        "--offline",
        action="store_true",
        help="with --resolver-only: no RxNav calls (local alias table and trigram only)",
    )
    la.add_argument(
        "--ci",
        action="store_true",
        help="CI gate: exit 1 on any wrong-drug resolution (needs --split test --resolver-only "
        "--offline)",
    )
    la.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    grounding_cli.add_parser(sub)
    lat = sub.add_parser(
        "latency",
        help="CI latency gate: replay the recordings in the 3 verify modes, exit 1 if AT-3 fails",
    )
    lat.add_argument("--ci", action="store_true", required=True)
    lat.add_argument("--questions", type=int, default=10)
    lat.add_argument("--reps", type=int, default=3)
    lat.add_argument("--verifier", choices=["fake", "real"], default="real")
    lat.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    args = p.parse_args(argv)
    if args.eval in ("grounding", "latency"):
        return args
    if args.ci and not (args.resolver_only and args.offline and args.split == "test"):
        p.error("--ci needs --split test --resolver-only --offline")
    if args.ci and args.mode == "baseline":
        p.error("--ci gates the system arm: use --mode system or both")
    if args.runs < 1:
        p.error("--runs must be at least 1")
    if args.offline and not args.resolver_only:
        p.error("--offline only applies to --resolver-only")
    return args


async def _resolver_only(args: argparse.Namespace, settings: Settings) -> dict:
    from pharmarag.models.embedder import SentenceTransformerEmbedder

    thresholds = load_thresholds(settings.thresholds_path)
    pool = await create_pool(settings.database_url)
    conn = await connect(settings.database_url)
    rxnav = (
        None
        if args.offline
        else RxNavClient(settings.rxnav_base_url, timeout_s=settings.rxnav_timeout_s)
    )
    embedder = None
    if args.mode in ("baseline", "both"):
        embedder = await asyncio.to_thread(
            SentenceTransformerEmbedder, settings.embedding_model, settings.embedding_dim
        )
    try:
        return await lookalike_run.run_resolver_only(
            split=args.split,
            mode=args.mode,
            runs=args.runs,
            resolver_factory=lambda: CascadeResolver(
                PgLocalStore(pool), rxnav, thresholds.resolver, InMemoryCache()
            ),
            conn=conn,
            embedder=embedder,
            thresholds=thresholds,
            rxnav_mode=(
                "n/a" if args.mode == "baseline" else "offline" if args.offline else "live"
            ),
        )
    finally:
        if rxnav is not None:
            await rxnav.aclose()
        await conn.close()
        await pool.close()


async def _full(args: argparse.Namespace, settings: Settings) -> dict:
    from pharmarag.app.runtime import build_runtime, close_runtime
    from pharmarag.graph.run import run_ask

    settings = settings.model_copy(update={"eval_mode": True})  # honour the flags (FR-9)
    runtime = await build_runtime(settings)
    conn = await connect(settings.database_url)
    try:

        async def ask(question: str, flags) -> dict:
            return await run_ask(runtime.graph, runtime.deps, question=question, flags=flags)

        return await lookalike_run.run_full(
            split=args.split,
            mode=args.mode,
            runs=args.runs,
            ask=ask,
            conn=conn,
            model_id=runtime.deps.llm.model_id,
            thresholds=runtime.deps.thresholds,
            resolver_available=runtime.deps.resolver_available,
        )
    finally:
        await conn.close()
        await close_runtime(runtime)


def _gate_result(name: str, failures: list[str]) -> int:
    if failures:
        print(f"CI GATE FAILED ({name}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"CI gate passed ({name})")
    return 0


async def _latency_gate(args: argparse.Namespace) -> int:
    code, report = await asyncio.to_thread(
        ci_gates.run_latency_replay,
        args.out_dir,
        questions=args.questions,
        reps=args.reps,
        verifier=args.verifier,
    )
    if report is None:
        return _gate_result("latency", [f"the replay did not produce a report (exit {code})"])
    return _gate_result("latency", ci_gates.latency_failures(report))


async def main_async(args: argparse.Namespace) -> int:
    if args.eval == "grounding":
        return await grounding_cli.main_async(args)
    if args.eval == "latency":
        return await _latency_gate(args)
    settings = get_settings()
    report = await (_resolver_only if args.resolver_only else _full)(args, settings)
    json_path, md_path = write_report(report, args.out_dir, lookalike_run.render_markdown(report))
    print(f"wrote {json_path}\nwrote {md_path}")
    if args.ci:
        rows = len(lookalike.load_dataset(args.split))
        return _gate_result("lookalike", ci_gates.lookalike_failures(report, dataset_rows=rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())

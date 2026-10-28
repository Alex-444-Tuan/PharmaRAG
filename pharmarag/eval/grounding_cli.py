"""Command line for the grounding eval (spec section 11.2), wired into `pharmarag.eval.run`.

    python -m pharmarag.eval.run grounding --build-dev     synthetic dev set from the corpus
    python -m pharmarag.eval.run grounding --tune          tune thresholds on dev (no LLM)
    python -m pharmarag.eval.run grounding --latency       per-sentence verifier latency
    python -m pharmarag.eval.run grounding --generate      40 claims with the REAL LLM
    python -m pharmarag.eval.run grounding --worksheet     labelling worksheet for the human
    python -m pharmarag.eval.run grounding --import        worksheet -> grounding_labels_test.jsonl
    python -m pharmarag.eval.run grounding --topup         perturbed claims if fewer than 10 are H
    python -m pharmarag.eval.run grounding --score         AT-2 and AT-2b on the labelled claims
    python -m pharmarag.eval.run grounding --score --ci    the same, exit 1 unless AT-2 is valid

Only --generate calls the LLM. Everything else is local models and files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from pharmarag.app.settings import REPO_ROOT, Settings, Thresholds, get_settings, load_thresholds
from pharmarag.db import connect
from pharmarag.eval import (
    ci_gates,
    grounding_bench,
    grounding_dev,
    grounding_generate,
    grounding_labels,
    grounding_score,
    grounding_tune,
)
from pharmarag.eval import (
    grounding_data as gd,
)
from pharmarag.eval import report as report_mod
from pharmarag.eval.report import corpus_version, sha256_of, write_report
from pharmarag.grounding.verifier import GroundingVerifier

ACTIONS = ("build-dev", "tune", "latency", "generate", "worksheet", "import", "topup", "score")


def add_parser(sub) -> argparse.ArgumentParser:
    g = sub.add_parser("grounding", help="hallucinated claims (AT-2, AT-2b, spec 11.2)")
    group = g.add_mutually_exclusive_group(required=True)
    for a in ACTIONS:
        group.add_argument(f"--{a}", dest="action", action="store_const", const=a)
    g.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    g.add_argument("--force", action="store_true", help="overwrite a frozen file or a worksheet")
    g.add_argument(
        "--ci", action="store_true", help="with --score: exit 1 unless summary.at2_valid is true"
    )
    g.add_argument("--nli-model", default=None, help="override NLI_MODEL (e.g. the xsmall model)")
    g.add_argument("--context", choices=["neighbour", "span", "both", "cascade"], default=None)
    g.add_argument("--torch-threads", type=int, default=None, help="torch.set_num_threads(N)")
    g.add_argument("--heading", choices=["none", "sub", "full"], default=None)
    g.add_argument("--candidates", type=int, default=None, help="override candidate_spans")
    g.add_argument("--repeats", type=int, default=3, help="--latency: timed passes")
    g.add_argument("--target-claims", type=int, default=grounding_generate.TARGET_CLAIMS)
    return g


def _load_verifier(args: argparse.Namespace, settings: Settings, thresholds: Thresholds):
    from pharmarag.grounding.factory import build_verifier
    from pharmarag.models.embedder import SentenceTransformerEmbedder

    if args.torch_threads:
        import torch

        torch.set_num_threads(args.torch_threads)

    if args.nli_model:
        # A pinned commit belongs to the pinned model: another model loads at its default branch.
        settings = settings.model_copy(
            update={"nli_model": args.nli_model, "nli_model_revision": ""}
        )
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_dim,
        revision=settings.embedding_model_revision or None,
    )
    verifier = build_verifier(settings, thresholds, embedder)
    assert isinstance(verifier, GroundingVerifier)
    if args.context:
        verifier.context = args.context
    if args.heading:
        verifier.heading = args.heading
    if args.candidates:
        verifier.thresholds = verifier.thresholds.model_copy(
            update={"candidate_spans": args.candidates}
        )
    return verifier


def _hardware() -> str:
    return f"{platform.platform()} {platform.processor() or platform.machine()}, CPU only"


async def _build_dev(args, settings) -> int:
    conn = await connect(settings.database_url)
    try:
        claims = await grounding_dev.build_dev(conn)
    finally:
        await conn.close()
    gd.write_jsonl(gd.DEV_CLAIMS, claims)
    n_s = sum(c.label == "S" for c in claims)
    print(f"wrote {gd.DEV_CLAIMS} ({len(claims)} claims: {n_s} S, {len(claims) - n_s} H)")
    print(f"sha256 {sha256_of(gd.DEV_CLAIMS)}")
    return 0


async def _tune(args, settings) -> int:
    thresholds = load_thresholds(settings.thresholds_path)
    verifier = await asyncio.to_thread(_load_verifier, args, settings, thresholds)
    report = await grounding_tune.run_tune(verifier, gd.load_claims(gd.DEV_CLAIMS), thresholds)
    j, m = write_report(report, args.out_dir, grounding_tune.render_markdown(report))
    print(f"wrote {j}\nwrote {m}")
    for mode, r in report["modes"].items():
        c, p = r["current_summary"], r["proposed_summary"]
        print(
            f"[{mode}] current: precision {c['precision_text']}, recall {c['recall_text']}, "
            f"false-reject {c['at2b_text']}, reduction {c['reduction_text']}"
        )
        print(
            f"[{mode}] proposed {r['proposed']}: precision {p['precision_text']}, "
            f"recall {p['recall_text']}, false-reject {p['at2b_text']}, "
            f"reduction {p['reduction_text']}"
        )
    return 0


async def _latency(args, settings) -> int:
    thresholds = load_thresholds(settings.thresholds_path)
    verifier = await asyncio.to_thread(_load_verifier, args, settings, thresholds)
    claims = gd.load_claims(gd.DEV_CLAIMS)
    if gd.TEST_CLAIMS.exists():
        claims += gd.load_claims(gd.TEST_CLAIMS)
    variants = {
        "spec: neighbour, no heading, 5 candidates": {
            "context": "neighbour",
            "heading": "none",
            "candidate_spans": 5,
        },
        "neighbour + heading, 3 candidates": {
            "context": "neighbour",
            "heading": "sub",
            "candidate_spans": 3,
        },
        "cascade + heading, 5 candidates": {
            "context": "cascade",
            "heading": "sub",
            "candidate_spans": 5,
        },
        "shipped: cascade + heading, 3 candidates": {
            "context": "cascade",
            "heading": "sub",
            "candidate_spans": 3,
        },
    }
    report = await grounding_bench.run_bench(
        verifier,
        claims,
        thresholds,
        variants=variants,
        repeats=args.repeats,
        hardware=_hardware(),
    )
    j, m = write_report(report, args.out_dir, grounding_bench.render_markdown(report))
    print(f"wrote {j}\nwrote {m}")
    print(grounding_bench.render_markdown(report))
    return 0


async def _generate(args, settings) -> int:
    from pharmarag.app.runtime import build_runtime, close_runtime
    from pharmarag.llm.factory import get_llm

    if gd.TEST_CLAIMS.exists() and not args.force:
        print(f"{gd.TEST_CLAIMS} is frozen; refusing to regenerate (pass --force)")
        return 2
    settings = settings.model_copy(update={"eval_mode": True})  # honour flags.verifier=off
    llm = grounding_generate.CountingLLM(get_llm(settings))
    runtime = await build_runtime(settings, llm=llm)  # type: ignore[arg-type]
    conn = await connect(settings.database_url)
    try:
        claims, log = await grounding_generate.generate(
            runtime.graph,
            runtime.deps,
            gd.load_questions(),
            llm=llm,
            target=args.target_claims,
        )
        corpus = await corpus_version(conn)
    finally:
        await conn.close()
        await close_runtime(runtime)
    gd.write_jsonl(gd.TEST_CLAIMS, claims)
    summary = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": llm.model_id,
        "corpus_version": corpus,
        "claims": len(claims),
        "claims_sha256": sha256_of(gd.TEST_CLAIMS),
        "usage": llm.usage(),
        "per_question": log,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "grounding_generation_log.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {gd.TEST_CLAIMS} ({len(claims)} claims)\nwrote {out}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=2))
    return 0 if len(claims) >= args.target_claims else 1


def _worksheet_path(args: argparse.Namespace) -> Path:
    return args.out_dir / grounding_labels.WORKSHEET.name


async def _worksheet(args, settings) -> int:
    path = grounding_labels.write_worksheet(
        gd.load_claims(gd.TEST_CLAIMS), _worksheet_path(args), force=args.force
    )
    print(f"wrote {path}\nType S or H after each LABEL: line, then run --import.")
    return 0


async def _import(args, settings) -> int:
    claims = gd.load_claims(gd.TEST_CLAIMS)
    try:
        rows = grounding_labels.import_worksheet(claims, _worksheet_path(args), gd.TEST_LABELS)
    except grounding_labels.WorksheetError as exc:
        print(exc)
        return 2
    n_h = sum(r["label"] == "H" for r in rows)
    print(f"wrote {gd.TEST_LABELS}: {len(rows)} labels, {n_h} H, {len(rows) - n_h} S")
    if gd.TEST_TOPUP.exists():
        gd.TEST_TOPUP.unlink()  # it was planned against the old labels
        print(f"removed the old {gd.TEST_TOPUP.name}; run --topup again if it is needed")
    if n_h < grounding_labels.MIN_HALLUCINATED:
        print(f"Fewer than {grounding_labels.MIN_HALLUCINATED} are H: run --topup (spec 11.2).")
    return 0


async def _topup(args, settings) -> int:
    claims = gd.load_claims(gd.TEST_CLAIMS)
    labels = grounding_labels.read_labels(gd.TEST_LABELS)
    topup = grounding_labels.write_topup(claims, labels, gd.TEST_TOPUP)
    n_h = sum(v == "H" for v in labels.values())
    if not topup:
        print(
            f"{n_h} natural claims are H (at least {grounding_labels.MIN_HALLUCINATED}): "
            "no top-up needed."
        )
        return 0
    print(f"{n_h} natural H; replacing {len(topup)} supported claims with perturbed ones:")
    for c in topup:
        print(f"  {c.meta['replaces']} -> {c.id} [{c.kind}] {c.sentence}")
    return 0


SCORE_FORBIDDEN = ("context", "heading", "candidates", "nli_model", "torch_threads")


def score_overrides(args: argparse.Namespace) -> list[str]:
    """The tuning-only options that were given. Test-set scoring accepts none of them: its
    settings come from the frozen config/thresholds.yaml and the environment, never the run."""
    return [f"--{n.replace('_', '-')}" for n in SCORE_FORBIDDEN if getattr(args, n, None)]


async def _score(args, settings) -> int:
    given = score_overrides(args)
    if given:
        print(
            f"--score refuses {', '.join(given)}: test-set scoring must use the frozen settings "
            "(config/thresholds.yaml). Change the file, tune on dev, commit, then score."
        )
        return 2
    thresholds = load_thresholds(settings.thresholds_path)
    # Before any model loads or any claim is read.
    if report_mod.thresholds_provenance().get("thresholds_dirty") is True:
        print(
            "--score refuses a dirty config/thresholds.yaml: commit it first "
            "(thresholds must predate the test run, spec section 11)."
        )
        return 2
    verifier = await asyncio.to_thread(_load_verifier, args, settings, thresholds)
    conn = await connect(settings.database_url)
    try:
        report, md = await grounding_score.run_score(
            verifier=verifier, thresholds=thresholds, conn=conn, llm_model_id=None
        )
    except grounding_score.ScoreRefused as exc:
        print(f"--score refused: {exc}")
        return 2
    finally:
        await conn.close()
    j, m = write_report(report, args.out_dir, md)
    print(f"wrote {j}\nwrote {m}")
    print(md)
    if args.ci:
        failures = ci_gates.grounding_failures(
            report, reference=ci_gates.load_grounding_reference()
        )
        for f in failures:
            print(f"CI GATE FAILED (grounding): {f}")
        return 1 if failures else 0
    return 0


HANDLERS = {
    "build-dev": _build_dev,
    "tune": _tune,
    "latency": _latency,
    "generate": _generate,
    "worksheet": _worksheet,
    "import": _import,
    "topup": _topup,
    "score": _score,
}


async def main_async(args: argparse.Namespace) -> int:
    if args.ci and args.action != "score":
        print("--ci applies to --score only")
        return 2
    return await HANDLERS[args.action](args, get_settings())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pharmarag.eval.run")
    add_parser(p.add_subparsers(dest="eval", required=True))
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())

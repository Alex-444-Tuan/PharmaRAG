"""Per-sentence verifier latency on CPU, real corpus evidence (spec section 9, "Latency design").

The budget is 150 ms p95 per sentence. This measures `verify_sentence` the way stream mode calls
it (thread hop, sentence embedding, candidate selection, number check, one NLI batch), after
`prepare` has run. `prepare` is timed separately: in stream mode it overlaps the LLM's time to
first token.

This is not the AT-3 end-to-end latency eval (that is Lane B's `eval/latency.py`). It isolates the
verifier so the 150 ms budget can be checked on its own and the fixes named in the spec compared.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from pharmarag.app.settings import GroundingThresholds, Thresholds
from pharmarag.eval import grounding_data as gd
from pharmarag.eval.grounding_data import Claim
from pharmarag.eval.report import base_meta
from pharmarag.graph.state import Sentence
from pharmarag.grounding.verifier import GroundingVerifier

BUDGET_P95_MS = 150.0


def percentiles(values_ms: Sequence[float]) -> dict:
    a = np.asarray(values_ms, dtype=float)
    return {
        "n": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "max_ms": round(float(a.max()), 1),
        "mean_ms": round(float(a.mean()), 1),
    }


async def time_verifier(
    verifier: GroundingVerifier, claims: Sequence[Claim], *, repeats: int = 3
) -> dict:
    """Latency of `verify_sentence` over `claims`, `repeats` passes, after one untimed pass."""
    prepared = []
    prep_ms = []
    for c in claims:
        t0 = time.perf_counter()
        prepared.append(await verifier.prepare(c.evidence, c.facts))
        prep_ms.append((time.perf_counter() - t0) * 1000)
    sentences = [Sentence(text=c.sentence, citations=c.citations) for c in claims]

    for s, p in zip(sentences, prepared, strict=True):  # warm-up pass, not timed
        await verifier.verify_sentence(s, p)

    all_ms: list[float] = []
    nli_ms: list[float] = []  # sentences whose number check passed, so the NLI model ran
    supported_ms: list[float] = []
    for _ in range(repeats):
        for s, p in zip(sentences, prepared, strict=True):
            t0 = time.perf_counter()
            out = await verifier.verify_sentence(s, p)
            ms = (time.perf_counter() - t0) * 1000
            all_ms.append(ms)
            if out.reason != "numeric_mismatch":
                nli_ms.append(ms)
            if out.verdict == "supported":
                supported_ms.append(ms)
    return {
        "verify_sentence": percentiles(all_ms),
        "verify_sentence_nli_path": percentiles(nli_ms),
        "verify_sentence_supported": percentiles(supported_ms) if supported_ms else None,
        "prepare": percentiles(prep_ms),
        "spans_per_evidence_set_p50": float(statistics.median(len(p.records) for p in prepared)),
    }


def variant(verifier: GroundingVerifier, **changes) -> GroundingVerifier:
    """A verifier sharing the loaded models but with other thresholds or premise mode."""
    th: GroundingThresholds = verifier.thresholds.model_copy(
        update={k: v for k, v in changes.items() if k in GroundingThresholds.model_fields}
    )
    return GroundingVerifier(
        verifier.embedder,
        verifier.nli,
        th,
        context=changes.get("context", verifier.context),
        heading=changes.get("heading", verifier.heading),
    )


async def run_bench(
    verifier: GroundingVerifier,
    claims: Sequence[Claim],
    thresholds: Thresholds,
    *,
    variants: dict[str, dict],
    repeats: int,
    hardware: str,
    when: datetime | None = None,
) -> dict:
    report = base_meta(
        eval_name="grounding_latency",
        split="dev",
        mode="verifier_only_cpu",
        runs=repeats,
        model_id=None,
        corpus="n/a (evidence snapshots are stored in the claim files)",
        thresholds=thresholds,
        dataset=gd.DEV_CLAIMS,
        now=when,
    )
    report |= {
        "nli_model": getattr(verifier.nli, "model_name", None),
        "embedding_model": getattr(verifier.embedder, "model_name", None),
        "hardware": hardware,
        "claims": len(claims),
        "budget_p95_ms": BUDGET_P95_MS,
        "variants": {},
    }
    for name, changes in variants.items():
        v = variant(verifier, **changes)
        stats = await time_verifier(v, claims, repeats=repeats)
        stats["config"] = changes
        stats["meets_budget"] = stats["verify_sentence"]["p95_ms"] <= BUDGET_P95_MS
        report["variants"][name] = stats
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Verifier per-sentence latency (CPU, real corpus evidence)",
        "",
        f"- Timestamp: {report['timestamp']}   Hardware: {report['hardware']}",
        f"- NLI model: {report['nli_model']}   Embedding model: {report['embedding_model']}",
        f"- Claims: {report['claims']} (dev set evidence), repeats: {report['runs']}, "
        f"one untimed warm-up pass",
        f"- Budget: at most {report['budget_p95_ms']:.0f} ms p95 per sentence (spec section 9)",
        "- `verify_sentence` = thread hop + sentence embedding + candidate selection + number "
        "check + one NLI batch. `prepare` (span embeddings) is separate: in stream mode it "
        "overlaps the LLM's time to first token.",
        "",
        "| variant | p50 ms | p95 ms | NLI-path p50 | NLI-path p95 | verdict supported p50 "
        "| verdict supported p95 | prepare p50 | budget (all sentences) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, v in report["variants"].items():
        a, n, p = v["verify_sentence"], v["verify_sentence_nli_path"], v["prepare"]
        s = v.get("verify_sentence_supported") or {"p50_ms": "-", "p95_ms": "-"}
        lines.append(
            f"| {name} | {a['p50_ms']} | {a['p95_ms']} | {n['p50_ms']} | {n['p95_ms']} "
            f"| {s['p50_ms']} | {s['p95_ms']} | {p['p50_ms']} "
            f"| {'met' if v['meets_budget'] else 'EXCEEDED'} |"
        )
    lines.append("")
    return "\n".join(lines)

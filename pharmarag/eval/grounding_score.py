"""Score the verifier on labelled claims (spec section 11.2, AT-2 and AT-2b).

    B = hallucinated claims. With the verifier off, all B reach the user.
    S = hallucinated claims the verifier passes (verdict "supported").
    reduction = (B - S) / B                                     target at least 73% (AT-2)
    false-reject rate = supported claims not passed / supported claims     at most 10% (AT-2b)

The guard comes first: a verifier that rejects everything has reduction 100%, so the reduction
only counts as a pass when AT-2b also passes. Every percentage is reported beside its counts.

No LLM runs here. The claims carry their evidence, so this is a pure function of the frozen
files, the verifier models and the thresholds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from psycopg import AsyncConnection

from pharmarag.app.settings import GroundingThresholds, Thresholds
from pharmarag.eval import grounding_data as gd
from pharmarag.eval import report as report_mod
from pharmarag.eval.grounding_data import Claim, Label
from pharmarag.eval.report import base_meta, corpus_version
from pharmarag.graph.state import Sentence
from pharmarag.grounding.verifier import Analysis, GroundingVerifier, verdict_fields

AT2_TARGET = 0.73
AT2B_MAX = 0.10


@dataclass
class LabelledClaim:
    claim: Claim
    label: Label
    origin: str  # "natural" or "perturbed"


@dataclass
class Scored:
    claim_id: str
    label: Label
    origin: str
    kind: str
    verdict: str
    reason: str | None
    score: float | None
    span: str | None
    best_contra: float | None
    sentence: str


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100 * n / d:.1f}%" if d else f"{n}/{d} = n/a"


def ratio(n: int, d: int) -> float | None:
    return n / d if d else None


# --- assembling the test set ---


def load_test_set(
    claims: Sequence[Claim],
    labels: dict[str, Label],
    topup: Sequence[Claim] = (),
) -> list[LabelledClaim]:
    """Natural claims with their human labels, minus those the top-up replaced, plus the
    perturbed replacements (labelled H by construction)."""
    replaced = {c.meta["replaces"] for c in topup}
    unknown = replaced - {c.id for c in claims}
    if unknown:
        raise ValueError(f"top-up replaces unknown claims: {sorted(unknown)}")
    stale = sorted(r for r in replaced if labels.get(r) != "S")
    if stale:
        raise ValueError(
            f"top-up replaces claims not labelled S: {stale}; it was planned against older "
            "labels, so re-run --topup"
        )
    out: list[LabelledClaim] = []
    for c in claims:
        if c.id in replaced:
            continue
        if c.id not in labels:
            raise ValueError(f"claim {c.id} has no label")
        out.append(LabelledClaim(c, labels[c.id], "natural"))
    for c in topup:
        if c.label != "H":
            raise ValueError(f"top-up claim {c.id} must be labelled H, not {c.label}")
        out.append(LabelledClaim(c, "H", "perturbed"))
    return out


# --- running the verifier ---


async def analyse_all(verifier: GroundingVerifier, claims: Sequence[Claim]) -> list[Analysis]:
    """Analyse each claim's sentence. Claims with the same evidence share one `prepare`."""
    prepared_cache: dict[tuple, object] = {}
    out: list[Analysis] = []
    for c in claims:
        key = (
            tuple((e.id, e.chunk_id) for e in c.evidence),
            tuple((f.id, f.text) for f in c.facts),
        )
        if key not in prepared_cache:
            prepared_cache[key] = await verifier.prepare(c.evidence, c.facts)
        sentence = Sentence(text=c.sentence, citations=c.citations)
        out.append(await verifier.analyse_sentence(sentence, prepared_cache[key]))  # type: ignore[arg-type]
    return out


def score_analyses(
    labelled: Sequence[LabelledClaim],
    analyses: Sequence[Analysis],
    thresholds: GroundingThresholds,
) -> list[Scored]:
    rows = []
    for lc, a in zip(labelled, analyses, strict=True):
        f = verdict_fields(a, thresholds)
        rows.append(
            Scored(
                claim_id=lc.claim.id,
                label=lc.label,
                origin=lc.origin,
                kind=lc.claim.kind,
                verdict=f["verdict"],
                reason=f["reason"],
                score=f["score"],
                span=f["span"],
                best_contra=a.best_contra,
                sentence=lc.claim.sentence,
            )
        )
    return rows


# --- the metrics ---


def summarise(rows: Sequence[Scored]) -> dict:
    """AT-2 and AT-2b from scored rows, with raw counts and the guard applied."""
    hallucinated = [r for r in rows if r.label == "H"]
    supported = [r for r in rows if r.label == "S"]
    b = len(hallucinated)
    s_pass = sum(r.verdict == "supported" for r in hallucinated)
    n_sup = len(supported)
    false_rejects = sum(r.verdict != "supported" for r in supported)
    reduction = ratio(b - s_pass, b)
    fr_rate = ratio(false_rejects, n_sup)
    at2b_pass = fr_rate is not None and fr_rate <= AT2B_MAX
    at2_met = reduction is not None and reduction >= AT2_TARGET
    tp = n_sup - false_rejects
    by_kind: dict[str, dict] = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in rows:
        k = by_kind[f"{r.label}:{r.kind}"]
        k["n"] += 1
        k["passed"] += r.verdict == "supported"
    reasons = Counter(r.reason or "-" for r in rows if r.verdict != "supported")
    return {
        "claims": len(rows),
        "origin": dict(Counter(f"{r.label}:{r.origin}" for r in rows)),
        "B_hallucinated": b,
        "S_hallucinated_passed": s_pass,
        "reduction": reduction,
        "reduction_text": pct(b - s_pass, b),
        "supported_total": n_sup,
        "supported_false_rejected": false_rejects,
        "at2b_false_reject_rate": fr_rate,
        "at2b_text": pct(false_rejects, n_sup),
        "at2b_pass": at2b_pass,
        "at2_target": AT2_TARGET,
        "at2_met": at2_met,
        # The reduction is only a result when the guard holds.
        "at2_valid": at2_met and at2b_pass,
        "precision_supported": ratio(tp, tp + s_pass),
        "precision_text": pct(tp, tp + s_pass),
        "recall_supported": ratio(tp, n_sup),
        "recall_text": pct(tp, n_sup),
        "by_kind": dict(by_kind),
        "rejection_reasons": dict(reasons),
    }


# --- the report ---


def build_report(
    *,
    split: str,
    rows: Sequence[Scored],
    thresholds: Thresholds,
    verifier: GroundingVerifier,
    llm_model_id: str | None,
    corpus: str,
    claims_path,
    extra_datasets: dict[str, str],
    note: str,
    when: datetime | None = None,
) -> dict:
    report = base_meta(
        eval_name="grounding",
        split=split,
        mode="verifier_vs_off",
        runs=1,
        model_id=llm_model_id,
        corpus=corpus,
        thresholds=thresholds,
        dataset=claims_path,
        now=when,
    )
    report["thresholds"]["grounding"] = thresholds.grounding.model_dump()
    report["nli_model"] = getattr(verifier.nli, "model_name", None)
    report["embedding_model"] = getattr(verifier.embedder, "model_name", None)
    report["verifier_config"] = {
        "context": verifier.context,
        "heading": verifier.heading,
        "candidate_spans": verifier.thresholds.candidate_spans,
        "nearest_spans": verifier.thresholds.nearest_spans,
        "entail_min": verifier.thresholds.entail_min,
        "contradict_min": verifier.thresholds.contradict_min,
    }
    report["other_datasets_sha256"] = extra_datasets
    report["note"] = note
    report["runs_note"] = "The verifier is deterministic, so one run; there is no LLM in scoring."
    report["summary"] = summarise(rows)
    if report["thresholds_dirty"] is True:
        # The numbers were produced with thresholds that are not committed, so nobody can
        # reproduce them from a commit: the result does not count (spec section 11).
        report["summary"]["at2_valid"] = False
        report["summary"]["invalid_reason"] = "config/thresholds.yaml has uncommitted changes"
    report["rows"] = [r.__dict__ for r in rows]
    return report


def render_markdown(report: dict) -> str:
    s = report["summary"]
    th = report["thresholds"]["grounding"]
    lines = [
        f"# Grounding eval (AT-2, AT-2b), {report['split']} split",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Writer LLM (claim generation): {report['model_id']}",
        f"- NLI model: {report['nli_model']}   Embedding model: {report['embedding_model']}",
        f"- corpus_version: {report['corpus_version']}",
        f"- config/thresholds.yaml commit: {report['thresholds_commit']} "
        f"(uncommitted changes: {report['thresholds_dirty']})",
        f"- Grounding thresholds (config/thresholds.yaml): {th}",
        f"- Verifier as run: {report['verifier_config']}",
        f"- Claims: {report['dataset']['file']} sha256 {report['dataset']['sha256'][:16]}",
        f"- Other datasets: {report['other_datasets_sha256']}",
        f"- Runs: {report['runs']} ({report['runs_note']})",
        f"- Natural vs perturbed claims: {s['origin']}",
        "",
        "## Result",
        "",
        "| metric | value | counts | target |",
        "|---|---|---|---|",
        f"| AT-2 reduction (B - S) / B | "
        f"{_p(s['reduction'])} | B={s['B_hallucinated']}, S={s['S_hallucinated_passed']} "
        f"| at least {100 * s['at2_target']:.0f}% |",
        f"| AT-2b false-reject rate | {_p(s['at2b_false_reject_rate'])} | "
        f"{s['supported_false_rejected']} of {s['supported_total']} supported claims rejected "
        f"| at most {100 * AT2B_MAX:.0f}% |",
        f"| precision of `supported` | {_p(s['precision_supported'])} | {s['precision_text']} | |",
        f"| recall of `supported` | {_p(s['recall_supported'])} | {s['recall_text']} | |",
        "",
        f"AT-2b guard: {'PASS' if s['at2b_pass'] else 'FAIL'}. "
        f"AT-2: {'MET' if s['at2_met'] else 'NOT MET'}. "
        f"AT-2 counts as a result only when the guard passes: "
        f"{'valid' if s['at2_valid'] else 'NOT VALID'}"
        + (f" ({s['invalid_reason']})" if s.get("invalid_reason") else "")
        + ".",
        "",
        "## By claim kind",
        "",
        "| label:kind | claims | passed as supported |",
        "|---|---|---|",
        *(f"| {k} | {v['n']} | {v['passed']} |" for k, v in sorted(s["by_kind"].items())),
        "",
        f"Rejection reasons: {s['rejection_reasons']}",
        "",
        report["note"],
        "",
    ]
    return "\n".join(lines)


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


class ScoreRefused(RuntimeError):
    """Test-set scoring was asked to run with settings it must not choose."""


def check_scoring_is_frozen(verifier: GroundingVerifier, thresholds: Thresholds) -> None:
    """Refuse to score the test claims unless the verifier is exactly the frozen one.

    Spec section 11: thresholds are tuned on dev only and frozen by commit before a test-set run.
    The premise mode, the heading, the candidate count and the thresholds are all in
    `config/thresholds.yaml`, so a verifier that differs from it, or a thresholds file with
    uncommitted edits, would let the test run pick its own settings.
    """
    g = thresholds.grounding
    if verifier.thresholds != g:
        raise ScoreRefused("the verifier's thresholds differ from config/thresholds.yaml")
    if (verifier.context, verifier.heading) != (g.premise_context, g.premise_heading):
        raise ScoreRefused(
            f"premise mode {verifier.context}/{verifier.heading} differs from the frozen "
            f"{g.premise_context}/{g.premise_heading}"
        )
    if report_mod.thresholds_provenance().get("thresholds_dirty") is True:
        raise ScoreRefused(
            "config/thresholds.yaml has uncommitted changes; commit them before scoring the "
            "test set (spec section 11)"
        )


async def run_score(
    *,
    verifier: GroundingVerifier,
    thresholds: Thresholds,
    conn: AsyncConnection,
    llm_model_id: str | None,
) -> tuple[dict, str]:
    """Score the frozen test claims with the human labels (and top-up, if any)."""
    from pharmarag.eval.grounding_labels import read_labels

    check_scoring_is_frozen(verifier, thresholds)
    claims = gd.load_claims(gd.TEST_CLAIMS)
    labels = read_labels(gd.TEST_LABELS)
    topup = gd.load_claims(gd.TEST_TOPUP) if gd.TEST_TOPUP.exists() else []
    labelled = load_test_set(claims, labels, topup)
    analyses = await analyse_all(verifier, [lc.claim for lc in labelled])
    rows = score_analyses(labelled, analyses, thresholds.grounding)
    extra = {gd.TEST_LABELS.name: gd.sha256_file(gd.TEST_LABELS)}
    if topup:
        extra[gd.TEST_TOPUP.name] = gd.sha256_file(gd.TEST_TOPUP)
    note = (
        "Labels: a human marked each natural claim S or H against the cited evidence shown in "
        "the worksheet (rubric at the top of reports/grounding_label_worksheet.md). "
        + (
            f"Top-up fired: {len(topup)} supported claims were replaced by perturbed claims "
            "labelled H by construction. "
            if topup
            else "Top-up did not fire. "
        )
        + "Baseline (verifier off) returns every sentence, so all B reach the user."
    )
    report = build_report(
        split="test",
        rows=rows,
        thresholds=thresholds,
        verifier=verifier,
        llm_model_id=llm_model_id or _claims_model(claims),
        corpus=await corpus_version(conn),
        claims_path=gd.TEST_CLAIMS,
        extra_datasets=extra,
        note=note,
    )
    return report, render_markdown(report)


def _claims_model(claims: Sequence[Claim]) -> str | None:
    for c in claims:
        if c.meta.get("llm_model_id"):
            return c.meta["llm_model_id"]
    return None

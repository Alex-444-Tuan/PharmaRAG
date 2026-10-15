"""Tune the verifier thresholds on the synthetic dev set (spec section 11.2, "Rules for all three").

Dev claims only. Thresholds in config/thresholds.yaml are a contract file and are not edited
here: this proposes values, the orchestrator freezes them by commit before any test-set run.

For every context mode (what the NLI model reads as the premise) the verifier runs once per
claim. The thresholds are then swept over the saved analyses, which costs no model time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pharmarag.app.settings import GroundingThresholds, Thresholds
from pharmarag.eval import grounding_data as gd
from pharmarag.eval.grounding_data import Claim
from pharmarag.eval.grounding_score import (
    LabelledClaim,
    analyse_all,
    pct,
    score_analyses,
    summarise,
)
from pharmarag.eval.report import base_meta
from pharmarag.grounding.verifier import Analysis, GroundingVerifier, decide

ENTAIL_GRID = [round(0.05 * i, 2) for i in range(1, 20)] + [0.97, 0.99]
CONTRADICT_GRID = [round(0.05 * i, 2) for i in range(10, 20)] + [0.97, 0.99]
# On dev, the proposed entail_min must keep the false-reject rate at or below this (if any value
# can). The AT-2b limit is 10%; a 5% dev budget leaves headroom for a different test set.
DEV_FALSE_REJECT_BUDGET = 0.05
# And contradict_min is the value nearest the current one at which `contradicted` verdicts are at
# least this precise.
CONTRADICT_PRECISION = 0.95
PLATEAU_TOLERANCE = 0.015  # balanced accuracy: one claim out of the 34 supported ones


def _with(th: GroundingThresholds, **kw) -> GroundingThresholds:
    return th.model_copy(update=kw)


def sweep_entail(
    labelled: Sequence[LabelledClaim], analyses: Sequence[Analysis], base: GroundingThresholds
) -> list[dict]:
    rows = []
    for e in ENTAIL_GRID:
        s = summarise(score_analyses(labelled, analyses, _with(base, entail_min=e)))
        rows.append(
            {
                "entail_min": e,
                "false_reject_rate": s["at2b_false_reject_rate"],
                "false_reject": s["at2b_text"],
                "reduction": s["reduction"],
                "reduction_text": s["reduction_text"],
                "precision": s["precision_supported"],
                "precision_text": s["precision_text"],
                "recall": s["recall_supported"],
                "recall_text": s["recall_text"],
            }
        )
    return rows


def propose_entail(sweep: Sequence[dict], current: float) -> float:
    """The threshold within one claim of the best balanced accuracy that is nearest `current`.

    Balanced accuracy is the mean of the supported-recall and the hallucinated-rejection rate.
    The dev set has 76 claims, so a difference of one claim is noise: the spec's value is kept
    unless another value is better by more than PLATEAU_TOLERANCE. Values over the false-reject
    budget are excluded when any value is within it. Ties go to the higher value.
    """
    pool = [r for r in sweep if (r["false_reject_rate"] or 0.0) <= DEV_FALSE_REJECT_BUDGET]
    pool = pool or list(sweep)

    def bal(r: dict) -> float:
        return ((r["recall"] or 0.0) + (r["reduction"] or 0.0)) / 2

    best = max(bal(r) for r in pool)
    near = [r["entail_min"] for r in pool if bal(r) >= best - PLATEAU_TOLERANCE]
    return min(near, key=lambda v: (round(abs(v - current), 6), -v))


def sweep_contradict(
    labelled: Sequence[LabelledClaim],
    analyses: Sequence[Analysis],
    base: GroundingThresholds,
) -> list[dict]:
    """`contradicted` and `unsupported` are both removed, so contradict_min changes the reason
    in the debug payload and nothing the user sees. It is still tuned: a `contradicted` verdict
    should be one a reader can trust."""
    rows = []
    for c in CONTRADICT_GRID:
        th = _with(base, contradict_min=c)
        scored = score_analyses(labelled, analyses, th)
        contradicted = [r for r in scored if r.verdict == "contradicted"]
        wrong = sum(r.label == "S" for r in contradicted)
        hall = [r for r in scored if r.label == "H"]
        rows.append(
            {
                "contradict_min": c,
                "contradicted": len(contradicted),
                "contradicted_on_supported": wrong,
                "precision": (1 - wrong / len(contradicted)) if contradicted else None,
                "precision_text": pct(len(contradicted) - wrong, len(contradicted)),
                "hallucinated_flagged_contradicted": pct(
                    sum(r.verdict == "contradicted" for r in hall), len(hall)
                ),
            }
        )
    return rows


def propose_contradict(sweep: Sequence[dict], current: float) -> float:
    """The value nearest `current` at which `contradicted` verdicts are at least 95% precise."""
    ok = [
        r["contradict_min"]
        for r in sweep
        if r["contradicted"] == 0 or (r["precision"] or 0.0) >= CONTRADICT_PRECISION
    ]
    ok = ok or [max(r["contradict_min"] for r in sweep)]
    return min(ok, key=lambda v: (round(abs(v - current), 6), -v))


async def run_tune(
    verifier: GroundingVerifier,
    claims: Sequence[Claim],
    thresholds: Thresholds,
    *,
    modes: Sequence[str] = ("neighbour", "span", "both"),
    when: datetime | None = None,
) -> dict:
    labelled = [LabelledClaim(c, c.label, "synthetic") for c in claims if c.label]
    report = base_meta(
        eval_name="grounding_tune",
        split="dev",
        mode="threshold_sweep",
        runs=1,
        model_id=None,
        corpus="n/a (evidence snapshots are stored in the dev file)",
        thresholds=thresholds,
        dataset=gd.DEV_CLAIMS,
        now=when,
    )
    report["nli_model"] = getattr(verifier.nli, "model_name", None)
    report["embedding_model"] = getattr(verifier.embedder, "model_name", None)
    report["current"] = thresholds.grounding.model_dump()
    report["candidate_spans_used"] = verifier.thresholds.candidate_spans
    report["heading_mode"] = verifier.heading
    report["claims"] = {
        "S": sum(lc.label == "S" for lc in labelled),
        "H": sum(lc.label == "H" for lc in labelled),
    }
    report["modes"] = {}
    original = verifier.context
    try:
        for mode in modes:
            verifier.context = mode  # type: ignore[assignment]
            analyses = await analyse_all(verifier, [lc.claim for lc in labelled])
            cur = summarise(score_analyses(labelled, analyses, thresholds.grounding))
            entail_sweep = sweep_entail(labelled, analyses, thresholds.grounding)
            p_entail = propose_entail(entail_sweep, thresholds.grounding.entail_min)
            proposed = _with(thresholds.grounding, entail_min=p_entail)
            contra_sweep = sweep_contradict(labelled, analyses, proposed)
            p_contra = propose_contradict(contra_sweep, thresholds.grounding.contradict_min)
            proposed = _with(proposed, contradict_min=p_contra)
            prop = summarise(score_analyses(labelled, analyses, proposed))
            # The same claims without the "heading" kind. That kind was built to need the heading
            # in the premise, so results with it flatter a heading-prefixed verifier.
            keep = [i for i, lc in enumerate(labelled) if lc.claim.kind != "heading"]
            sub_l = [labelled[i] for i in keep]
            sub_a = [analyses[i] for i in keep]
            sub_sweep = sweep_entail(sub_l, sub_a, thresholds.grounding)
            sub_entail = propose_entail(sub_sweep, thresholds.grounding.entail_min)
            sub_prop = _with(thresholds.grounding, entail_min=sub_entail)
            report["modes"][mode] = {
                "current_summary": cur,
                "proposed": {"entail_min": p_entail, "contradict_min": p_contra},
                "proposed_summary": prop,
                "without_heading_kind": {
                    "claims": len(sub_l),
                    "current_summary": summarise(
                        score_analyses(sub_l, sub_a, thresholds.grounding)
                    ),
                    "at_proposed_summary": summarise(score_analyses(sub_l, sub_a, proposed)),
                    "own_proposed_entail_min": sub_entail,
                    "own_proposed_summary": summarise(score_analyses(sub_l, sub_a, sub_prop)),
                },
                "entail_sweep": entail_sweep,
                "contradict_sweep": contra_sweep,
            }
        if "both" in report["modes"]:
            # "cascade" must give the same verdicts as "both" at the same thresholds. Checked
            # here on the dev claims with the proposed values, with the real models.
            p = report["modes"]["both"]["proposed"]
            at = _with(thresholds.grounding, **p)
            saved = verifier.thresholds
            verifier.thresholds = saved.model_copy(update=p)
            verifier.context = "cascade"
            try:
                cas = await analyse_all(verifier, [lc.claim for lc in labelled])
                both = await _both(verifier, labelled)
            finally:
                verifier.thresholds = saved
            same = sum(decide(a, at)[0] == decide(b, at)[0] for a, b in zip(cas, both, strict=True))
            report["cascade_check"] = {
                "thresholds": p,
                "same_verdict": same,
                "claims": len(labelled),
                "summary": summarise(score_analyses(labelled, cas, at)),
            }
    finally:
        verifier.context = original
    return report


async def _both(verifier: GroundingVerifier, labelled: Sequence[LabelledClaim]) -> list[Analysis]:
    verifier.context = "both"
    return await analyse_all(verifier, [lc.claim for lc in labelled])


def _slice_lines(m: dict, report: dict) -> list[str]:
    """Per-kind results at the current and proposed thresholds, and the run without the
    heading-dependent kind."""
    cur, prop = m["current_summary"], m["proposed_summary"]
    w = m["without_heading_kind"]
    kinds = sorted(cur["by_kind"])
    lines = [
        "Per kind, claims passed as `supported` (S kinds should pass, H kinds should not):",
        "",
        "| kind | claims | passed at current | passed at proposed |",
        "|---|---|---|---|",
        *(
            f"| {k} | {cur['by_kind'][k]['n']} | {cur['by_kind'][k]['passed']} "
            f"| {prop['by_kind'][k]['passed']} |"
            for k in kinds
        ),
        "",
        f"Without the `heading` kind ({w['claims']} claims; that kind was built to need the "
        "heading):",
        "",
        "| thresholds | precision | recall | false-reject | reduction |",
        "|---|---|---|---|---|",
    ]
    for label, s in (
        (f"current ({report['current']['entail_min']})", w["current_summary"]),
        (f"proposed above ({m['proposed']['entail_min']})", w["at_proposed_summary"]),
        (f"own proposal ({w['own_proposed_entail_min']})", w["own_proposed_summary"]),
    ):
        lines.append(
            f"| {label} | {s['precision_text']} | {s['recall_text']} | {s['at2b_text']} "
            f"| {s['reduction_text']} |"
        )
    return [*lines, ""]


def render_markdown(report: dict) -> str:
    lines = [
        "# Grounding threshold tuning, dev split (synthetic claims)",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- NLI model: {report['nli_model']}   Embedding model: {report['embedding_model']}",
        f"- Dev claims: {report['dataset']['file']} sha256 {report['dataset']['sha256'][:16]}; "
        f"S={report['claims']['S']}, H={report['claims']['H']}",
        f"- config/thresholds.yaml commit: {report['thresholds_commit']} "
        f"(uncommitted changes: {report['thresholds_dirty']})",
        f"- Current grounding thresholds: {report['current']}",
        f"- Candidate spans used in this run: {report['candidate_spans_used']}   "
        f"Heading in premise: {report['heading_mode']}",
        "- Dev claims are synthetic (rule-based rewrites and perturbations of real label "
        "sentences). The test claims are natural writer sentences labelled by a human.",
        "- Precision and recall are for the verdict `supported` against label S: "
        "precision = supported verdicts that are S; recall = S claims given `supported`.",
        "- `contradict_min` changes only the reason in the debug payload: `unsupported` and "
        "`contradicted` are both removed.",
        "",
    ]
    if "cascade_check" in report:
        cc = report["cascade_check"]
        s = cc["summary"]
        lines += [
            "## Cascade check",
            "",
            f"`cascade` (span plus neighbour first, the span alone only on a miss) at the proposed "
            f"`both` thresholds {cc['thresholds']}: same verdict as `both` on "
            f"{cc['same_verdict']}/{cc['claims']} claims; precision {s['precision_text']}, "
            f"recall {s['recall_text']}, false-reject {s['at2b_text']}, "
            f"reduction {s['reduction_text']}.",
            "",
        ]
    for mode, m in report["modes"].items():
        cur, prop = m["current_summary"], m["proposed_summary"]
        p = m["proposed"]
        lines += [
            f"## Premise: {mode}",
            "",
            "| thresholds | precision | recall | false-reject | reduction |",
            "|---|---|---|---|---|",
            f"| current ({report['current']['entail_min']}/{report['current']['contradict_min']}) "
            f"| {cur['precision_text']} | {cur['recall_text']} | {cur['at2b_text']} "
            f"| {cur['reduction_text']} |",
            f"| proposed ({p['entail_min']}/{p['contradict_min']}) "
            f"| {prop['precision_text']} | {prop['recall_text']} | {prop['at2b_text']} "
            f"| {prop['reduction_text']} |",
            "",
            *_slice_lines(m, report),
            "entail_min sweep:",
            "",
            "| entail_min | precision | recall | false-reject | reduction |",
            "|---|---|---|---|---|",
            *(
                f"| {r['entail_min']} | {r['precision_text']} | {r['recall_text']} "
                f"| {r['false_reject']} | {r['reduction_text']} |"
                for r in m["entail_sweep"]
            ),
            "",
            "contradict_min sweep (at the proposed entail_min):",
            "",
            "| contradict_min | contradicted verdicts | on supported claims | precision |",
            "|---|---|---|---|",
            *(
                f"| {r['contradict_min']} | {r['contradicted']} | {r['contradicted_on_supported']} "
                f"| {r['precision_text']} |"
                for r in m["contradict_sweep"]
            ),
            "",
        ]
    return "\n".join(lines)

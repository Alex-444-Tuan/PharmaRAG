"""Human labelling of the test claims (spec section 11.2, "Labels" and "Top-up rule").

    --worksheet  write reports/grounding_label_worksheet.md. Labels are NEVER pre-filled: the
                 human labels must be independent of the verifier and of the author.
    --import     parse the filled-in worksheet into grounding_labels_test.jsonl and validate it
    --topup      if fewer than 10 claims are hallucinated, replace supported claims by labelled
                 perturbations (H by construction) in grounding_topup_test.jsonl

The frozen claims file is never edited: the top-up lives in its own file, so the claims SHA
stays valid and the natural-versus-perturbed split is always visible.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path

from pharmarag.app.settings import REPO_ROOT
from pharmarag.eval import grounding_data as gd
from pharmarag.eval import perturb
from pharmarag.eval.grounding_data import Claim, Label

WORKSHEET = REPO_ROOT / "reports" / "grounding_label_worksheet.md"
MIN_HALLUCINATED = 10  # spec 11.2 top-up rule: fewer than 10 of 40 triggers it
TOPUP_SEED = 20261009

RUBRIC = """\
# Grounding claim labelling worksheet

## Rubric (read this first)

Each block below is one sentence written by the answer model, the evidence blocks it cites, and
two empty lines for you to fill in.

- Type **S** after `LABEL:` if the sentence is **supported**: every fact in it is stated in, or
  directly follows from, the cited evidence shown in that block.
- Type **H** if it is **hallucinated**: **any** fact in it (a number, a unit, a drug, a
  population, a frequency, a negation, a causal claim) is not stated in or directly entailed by
  the cited evidence. **This holds even if the fact is true in the world.** Medical knowledge
  you hold does not count as evidence.
- Judge against the cited evidence shown, nothing else. A sentence whose support is not in the
  blocks it cites is H, even if some other block of the original answer would have supported it.
- Reasonable paraphrase is fine ("twice daily" for "two times a day"). A changed or extra
  number, a flipped negation, a different population or a stronger claim ("always", "prevents")
  than the evidence makes is not.
- Use the NOTE line for anything you want to remember. It is not read by any script.
- Do not use any other letter. Do not leave a LABEL line empty. Keep the `## Claim` headers and
  the `LABEL:` and `NOTE:` lines as they are: the importer reads them.
- Do not look at the verifier's output before labelling.

---
"""


class WorksheetError(ValueError):
    """The worksheet is not importable. The message lists every problem found."""


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in text.splitlines() or [""])


def render_worksheet(claims: Sequence[Claim]) -> str:
    parts = [RUBRIC]
    for c in claims:
        chunks = {e.id: e for e in c.evidence}
        facts = {f.id: f for f in c.facts}
        block = [f"## Claim {c.id}", ""]
        if c.question:
            block.append(f"QUESTION: {c.question}")
        block += [f"SENTENCE: {c.sentence}", f"CITES: {', '.join(c.citations) or '(none)'}", ""]
        for cid in c.citations:
            if cid in chunks:
                e = chunks[cid]
                title = e.section_title + (f" > {e.subsection_title}" if e.subsection_title else "")
                block += [f"EVIDENCE {cid} ({title}):", _quote(e.text), ""]
            elif cid in facts:
                block += [f"EVIDENCE {cid} (RxNorm fact):", _quote(facts[cid].text), ""]
        block += ["LABEL:", "NOTE:", "", "---", ""]
        parts.append("\n".join(block))
    return "\n".join(parts)


def worksheet_has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    return any(re.match(r"LABEL:\s*\S", line) for line in path.read_text().splitlines())


def write_worksheet(claims: Sequence[Claim], path: Path = WORKSHEET, force: bool = False) -> Path:
    if worksheet_has_labels(path) and not force:
        raise FileExistsError(
            f"{path} already has labels in it; refusing to overwrite a human's work "
            "(pass --force to replace it)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_worksheet(claims))
    return path


_HEADER = re.compile(r"^## Claim (\S+)\s*$", re.MULTILINE)
_LABEL = re.compile(r"^LABEL:(.*)$", re.MULTILINE)
_NOTE = re.compile(r"^NOTE:(.*)$", re.MULTILINE)


def parse_worksheet(text: str, claim_ids: Sequence[str]) -> list[dict]:
    """Rows {claim_id, label, note} in worksheet order. Raises WorksheetError listing every
    problem: an unknown or repeated claim, a missing, empty or invalid LABEL, a missing claim.

    A label is exactly S or H. Lower case is accepted and normalised; nothing else is.
    """
    problems: list[str] = []
    headers = list(_HEADER.finditer(text))
    rows: list[dict] = []
    seen: set[str] = set()
    known = set(claim_ids)
    for i, h in enumerate(headers):
        cid = h.group(1)
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[h.end() : end]
        if cid not in known:
            problems.append(f"{cid}: not a claim in the frozen set")
            continue
        if cid in seen:
            problems.append(f"{cid}: appears more than once")
            continue
        seen.add(cid)
        labels = _LABEL.findall(block)
        if len(labels) != 1:
            problems.append(f"{cid}: expected one LABEL line, found {len(labels)}")
            continue
        value = labels[0].strip()
        if not value:
            problems.append(f"{cid}: LABEL is empty")
            continue
        if value.upper() not in ("S", "H") or len(value) != 1:
            problems.append(f"{cid}: LABEL must be S or H, got {value!r}")
            continue
        note = _NOTE.search(block)
        rows.append(
            {
                "claim_id": cid,
                "label": value.upper(),
                "note": note.group(1).strip() if note else "",
            }
        )
    for cid in claim_ids:
        if cid not in seen:
            problems.append(f"{cid}: missing from the worksheet")
    if problems:
        raise WorksheetError("worksheet is not importable:\n  " + "\n  ".join(problems))
    return rows


def import_worksheet(
    claims: Sequence[Claim], worksheet: Path = WORKSHEET, out: Path = gd.TEST_LABELS
) -> list[dict]:
    rows = parse_worksheet(worksheet.read_text(), [c.id for c in claims])
    gd.write_jsonl(out, rows)
    return rows


def read_labels(path: Path = gd.TEST_LABELS) -> dict[str, Label]:
    out: dict[str, Label] = {}
    for r in gd.load_jsonl(path):
        if r["label"] not in ("S", "H"):
            raise ValueError(f"bad label {r['label']!r} for {r['claim_id']}")
        out[r["claim_id"]] = r["label"]
    return out


# --- top-up ---


def _evidence_texts(c: Claim) -> list[str]:
    return [e.text for e in c.evidence] + [f.text for f in c.facts]


def _ingredient(c: Claim) -> str:
    return c.evidence[0].ingredient_key if c.evidence else ""


def _content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def perturb_claim(
    claim: Claim, others: Sequence[Claim], rng: random.Random, kind: str
) -> str | None:
    texts = _evidence_texts(claim)
    s = claim.sentence
    if kind == "numeric":
        return perturb.change_number(s, texts, rng)
    if kind == "negation":
        return perturb.negate(s, texts)
    if kind == "frequency":
        return perturb.swap_frequency(s, texts)
    if kind == "population":
        return perturb.swap_population(s, texts)
    if kind == "other_drug":
        pool = [o for o in others if _ingredient(o) != _ingredient(claim) and o.id != claim.id]
        rng.shuffle(pool)
        have = _content_words(" ".join(texts))
        for o in pool:
            if len(_content_words(o.sentence) - have) >= 2:
                return o.sentence
        return None
    raise ValueError(kind)


TOPUP_KINDS = ("numeric", "negation", "other_drug", "frequency", "population")


def plan_topup(
    claims: Sequence[Claim], labels: dict[str, Label], seed: int = TOPUP_SEED
) -> list[Claim]:
    """Perturbed replacements for supported claims, so that at least 10 claims are H.

    Empty when 10 or more natural claims are already H. Supported claims are taken in claim
    order; each gets the first perturbation kind (rotating through the kinds) that applies.
    Raises when not enough supported claims can be perturbed.
    """
    n_h = sum(labels[c.id] == "H" for c in claims)
    need = MIN_HALLUCINATED - n_h
    if need <= 0:
        return []
    rng = random.Random(seed)
    out: list[Claim] = []
    rotation = 0
    for c in claims:
        if labels[c.id] != "S" or len(out) == need:
            continue
        for step in range(len(TOPUP_KINDS)):
            kind = TOPUP_KINDS[(rotation + step) % len(TOPUP_KINDS)]
            new = perturb_claim(c, claims, rng, kind)
            if new and new != c.sentence:
                out.append(
                    c.model_copy(
                        update={
                            "id": f"{c.id}p",
                            "sentence": new,
                            "label": "H",
                            "kind": kind,
                            "meta": {**c.meta, "replaces": c.id, "source_sentence": c.sentence},
                        }
                    )
                )
                rotation = (rotation + step + 1) % len(TOPUP_KINDS)
                break
    if len(out) < need:
        raise RuntimeError(f"could only perturb {len(out)} of the {need} claims needed")
    return out


def write_topup(
    claims: Sequence[Claim], labels: dict[str, Label], out: Path = gd.TEST_TOPUP
) -> list[Claim]:
    topup = plan_topup(claims, labels)
    if topup:
        gd.write_jsonl(out, topup)
    elif out.exists():
        out.unlink()  # a stale top-up from an earlier labelling must not leak into the score
    return topup

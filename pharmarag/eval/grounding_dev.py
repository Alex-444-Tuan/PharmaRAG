"""Build the synthetic dev tuning set for the verifier (spec section 11.2, "Rules for all three").

Claims are made from real chunks of the ingested corpus and labelled by construction, so no human
labelling is needed:

- S (supported): a label sentence restated verbatim, lightly rewritten by rules, two adjacent
  label sentences joined, or a sentence led by the population its subsection heading names
  ("In adults, ..." under "2.1 Adult Dosage"), as a writer does.
- H (hallucinated): a perturbation of a real sentence (changed dose, negation, swapped frequency
  or population, a claim stronger than the label makes), or a sentence moved in from another
  drug's label. Each perturbation is checked against the whole evidence set so the changed fact
  is not stated anywhere in it.

# SPEC: these dev claims are synthetic. The test claims are natural writer sentences labelled by
# a human. Thresholds are tuned here and nowhere else.

Run: python -m pharmarag.eval.run grounding --build-dev
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from psycopg import AsyncConnection

from pharmarag.eval import perturb
from pharmarag.eval.grounding_data import Claim
from pharmarag.graph.state import Chunk, Span
from pharmarag.grounding.numeric import header_unit

SEED = 20261008
N_DRUGS = 45
EVIDENCE_CHUNKS = 4  # the cited chunk plus three others from the same label
# How many claims of each kind to build.
PLAN: dict[str, int] = {
    "verbatim": 14,
    "rewrite": 14,
    "join": 6,
    "heading": 10,
    "numeric": 10,
    "negation": 8,
    "other_drug": 8,
    "frequency": 5,
    "population": 5,
    "overclaim": 6,
    # Harder supported kinds that mimic real writer output (cycle 3 review): the sentence is not a
    # copy of one label sentence.
    "tallman": 6,  # drug name in tall-man lettering in the label, plain in the sentence
    "combine": 8,  # two non-adjacent sentences of one chunk in one claim
    "row_restate": 8,  # a table row restated as prose
    # And its hallucinated twin: the same prose with one number changed.
    "row_numeric": 6,
}
LABEL_OF = {
    "verbatim": "S",
    "rewrite": "S",
    "join": "S",
    "heading": "S",
    "numeric": "H",
    "negation": "H",
    "other_drug": "H",
    "frequency": "H",
    "population": "H",
    "overclaim": "H",
    "tallman": "S",
    "combine": "S",
    "row_restate": "S",
    "row_numeric": "H",
}
# Labels whose drug names are in tall-man lettering (hydrALAZINE, PredniSONE). They are always in
# the sample, because a random sample of 45 drugs may hold none.
TALLMAN_KEYS = ("5470", "8640")
TALLMAN_NAMES = ("hydralazine", "prednisone")
# Sections where claims are common in real answers.
SECTIONS = (
    "34067-9",  # indications
    "34068-7",  # dosage and administration
    "34070-3",  # contraindications
    "43685-7",  # warnings and precautions
    "34084-4",  # adverse reactions
    "34073-7",  # drug interactions
    "43684-0",  # use in specific populations
    "34088-5",  # overdosage
    "34090-1",  # clinical pharmacology
)

_BAD = re.compile(r"\b(?:see|table|figure|section|refer)\b|[\[\]|]", re.IGNORECASE)


@dataclass
class DrugChunks:
    ingredient_key: str
    title: str
    chunks: list[Chunk]  # fused_score 0, id assigned later


def usable_span_texts(chunk: Chunk) -> list[tuple[Span, str]]:
    """Spans that read like a standalone claim: one plain sentence, 6 to 35 words."""
    out = []
    for sp in chunk.spans:
        text = perturb.clean_sentence(chunk.text[sp.start : sp.end])
        words = text.split()
        if not (6 <= len(words) <= 35):
            continue
        if not (text[0].isupper() and text.endswith(".")) or _BAD.search(text):
            continue
        if text.count(".") > 1 and not re.search(r"\d\.\d", text):
            continue  # more than one sentence
        out.append((sp, text))
    return out


async def load_drugs(conn: AsyncConnection, rng: random.Random, n: int) -> list[DrugChunks]:
    cur = await conn.execute(
        "SELECT DISTINCT ON (l.ingredient_key) l.ingredient_key, l.title, l.setid, l.spl_version "
        "FROM label l ORDER BY l.ingredient_key"
    )
    labels = await cur.fetchall()
    labels = sorted(labels, key=lambda r: r[0])
    rng.shuffle(labels)
    forced = [r for r in labels if r[0] in TALLMAN_KEYS]
    labels = forced + [r for r in labels if r[0] not in TALLMAN_KEYS]
    drugs: list[DrugChunks] = []
    for key, title, setid, ver in labels[: n * 2]:
        cur = await conn.execute(
            "SELECT id, setid, spl_version, ingredient_key, section_loinc, section_title, "
            "subsection_title, text, spans FROM chunk WHERE setid = %s AND spl_version = %s "
            "ORDER BY ordinal",
            (setid, ver),
        )
        chunks = [
            Chunk(
                id="c0",
                chunk_id=r[0],
                setid=r[1],
                spl_version=r[2],
                ingredient_key=r[3],
                section_loinc=r[4],
                section_title=r[5],
                subsection_title=r[6],
                text=r[7],
                spans=[Span(**s) for s in r[8]],
                fused_score=0.0,
            )
            for r in await cur.fetchall()
            if r[4] in SECTIONS
        ]
        if len(chunks) >= EVIDENCE_CHUNKS + 1:
            drugs.append(DrugChunks(key, title, chunks))
        if len(drugs) == n:
            break
    return drugs


def _evidence(drug: DrugChunks, cited: Chunk, rng: random.Random) -> tuple[list[Chunk], str]:
    """The cited chunk plus three other chunks of the same label, ids c1..c4, shuffled."""
    others = [c for c in drug.chunks if c.chunk_id != cited.chunk_id]
    # prefer other sections, so the nearest-span search has something to be distracted by
    others.sort(key=lambda c: (c.section_loinc == cited.section_loinc, rng.random()))
    picked = [cited, *others[: EVIDENCE_CHUNKS - 1]]
    rng.shuffle(picked)
    numbered = [c.model_copy(update={"id": f"c{i + 1}"}) for i, c in enumerate(picked)]
    cited_id = next(c.id for c in numbered if c.chunk_id == cited.chunk_id)
    return numbered, cited_id


# subsection title pattern -> (phrase a writer would lead with, word that means it is already there)
_HEADINGS: list[tuple[str, str, str]] = [
    (r"adult", "In adults", "adult"),
    (r"pediatric", "In pediatric patients", "pediatric"),
    (r"geriatric", "In geriatric patients", "geriatric"),
    (r"pregnan", "During pregnancy", "pregnan"),
    (r"lactation|nursing", "During lactation", "lactat"),
    (r"renal impairment", "In patients with renal impairment", "renal"),
    (r"hepatic impairment", "In patients with hepatic impairment", "hepatic"),
]


def heading_prefix(subsection_title: str | None) -> tuple[str, str] | None:
    """(lead-in phrase, word that marks it as already stated) for a population subsection."""
    for pat, phrase, key in _HEADINGS:
        if subsection_title and re.search(pat, subsection_title, re.IGNORECASE):
            return phrase, key
    return None


_TALLMAN_TOKEN = re.compile(r"\b[A-Za-z]*[a-z][A-Za-z]*[A-Z]{2,}[A-Za-z]*\b")


def tallman_name(text: str) -> str | None:
    """The tall-man drug name in `text` (hydrALAZINE, PredniSONE), or None."""
    for m in _TALLMAN_TOKEN.finditer(text):
        if m.group(0).lower() in TALLMAN_NAMES:
            return m.group(0)
    return None


def plain_name(text: str, name: str) -> str:
    """`text` with the tall-man `name` in ordinary casing, as a writer types it."""
    plain = name.lower()
    if text.startswith(name):
        plain = plain.capitalize()
    return text.replace(name, plain)


def row_spans(chunk: Chunk) -> list[tuple[Span, str]]:
    """Table-row spans: "Header (unit): value; Header (unit): value", short enough to restate."""
    out = []
    for sp in chunk.spans:
        text = chunk.text[sp.start : sp.end].strip()
        cells = [c.partition(": ") for c in text.split("; ")]
        if len(cells) >= 2 and all(sep for _, sep, _ in cells) and len(text.split()) <= 80:
            out.append((sp, text))
    return out


_VALUE = re.compile(r"^[<>\u2265\u2264=~ ]*\d[\d.,]*(?: (?:to|-|\u2013|or) \d[\d.,]*)?$")


def row_sentence(row: str, rng: random.Random) -> str | None:
    """A table row restated as a sentence, the way a writer would put one cell in prose.

    "Adverse Reaction: Arthralgia; % 10 mg N=3,908: 8.9" -> "Arthralgia occurred in 8.9% of
    patients in the 10 mg group." and "Renal Function Creatinine Clearance (mL/min): >30 to 59;
    Total Daily Dose Range (mg/day): 400 to 1400" -> "For renal function creatinine clearance
    >30 to 59 mL/min, the total daily dose range is 400 to 1400 mg/day."
    """
    cells = [c.partition(": ") for c in row.split("; ")]
    h0, _, v0 = cells[0]
    cols = [(h, v) for h, _, v in cells[1:] if _VALUE.match(v.strip())]
    if not cols or len(v0) > 40:
        return None
    h, v = rng.choice(cols)
    v = v.strip()
    if h.startswith("%"):
        group = re.sub(r"\s*N=.*$", "", h[1:]).strip()
        if not group or len(v0.split()) > 6:
            return None
        return f"{v0} occurred in {v}% of patients in the {group} group."
    unit = header_unit(h)
    base = re.sub(r"\s*\([^()]*\)\s*$", "", h).strip().lower()
    if not unit or not base:
        return None
    unit0 = header_unit(h0)
    base0 = re.sub(r"\s*\([^()]*\)\s*$", "", h0).strip().lower()
    where = f"{v0.strip()} {unit0}" if unit0 and _VALUE.match(v0.strip()) else v0.strip()
    return f"For {base0} {where}, the {base} is {v} {unit}."


def _first_word(title: str) -> str:
    return re.split(r"[\s(]", title.strip().lower(), maxsplit=1)[0]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def make_claim(
    kind: str, drugs: list[DrugChunks], rng: random.Random, used: set[tuple[int, int]]
) -> Claim | None:
    """One attempt at a claim of `kind`. None when the draw does not suit the rule."""
    drug = rng.choice(drugs)
    pool = drug.chunks
    if kind == "heading":
        pool = [c for c in drug.chunks if heading_prefix(c.subsection_title)]
    elif kind in ("row_restate", "row_numeric"):
        pool = [c for c in drug.chunks if row_spans(c)]
    elif kind == "tallman":
        pool = [c for c in drug.chunks if any(tallman_name(t) for _, t in usable_span_texts(c))]
    if not pool:
        return None
    cited = rng.choice(pool)
    spans = row_spans(cited) if kind in ("row_restate", "row_numeric") else usable_span_texts(cited)
    if kind == "tallman":
        spans = [(sp, t) for sp, t in spans if tallman_name(t)]
    if not spans:
        return None
    sp, text = rng.choice(spans)
    if (cited.chunk_id, sp.i) in used:
        return None
    evidence, cited_id = _evidence(drug, cited, rng)
    texts = [c.text for c in evidence]
    meta: dict = {
        "drug": drug.title,
        "ingredient_key": drug.ingredient_key,
        "section_loinc": cited.section_loinc,
        "source_span": f"{cited_id}:{sp.i}",
        "source_sentence": text,
    }
    if kind == "verbatim":
        sentence = perturb.rewrite_verbatim(text)
    elif kind == "rewrite":
        sentence = perturb.rewrite_light(text, rng)
    elif kind == "heading":
        prefix, key = heading_prefix(cited.subsection_title)  # type: ignore[misc]
        if key in text.lower():
            return None  # the sentence already says it; the heading adds nothing
        sentence = f"{prefix}, {perturb.lower_first(text)}"
        meta["heading"] = cited.subsection_title
    elif kind == "tallman":
        sentence = plain_name(text, tallman_name(text))  # type: ignore[arg-type]
        meta["tallman"] = tallman_name(text)
    elif kind == "combine":
        far = [t for s2, t in spans if abs(s2.i - sp.i) >= 2]
        if not far:
            return None
        other_text = rng.choice(far)
        sentence = f"{text.rstrip('.')}, and {perturb.lower_first(other_text)}"
        meta["source_sentence_2"] = other_text
    elif kind in ("row_restate", "row_numeric"):
        sentence = row_sentence(text, rng)
        if sentence and kind == "row_numeric":
            meta["restated"] = sentence
            sentence = perturb.change_number(sentence, texts, rng)
    elif kind == "join":
        nxt = {s.i: t for s, t in spans}.get(sp.i + 1)
        if not nxt:
            return None
        sentence = perturb.join_sentences(text, nxt)
        meta["source_sentence_2"] = nxt
    elif kind == "numeric":
        sentence = perturb.change_number(text, texts, rng)
    elif kind == "negation":
        sentence = perturb.negate(text, texts)
    elif kind == "frequency":
        sentence = perturb.swap_frequency(text, texts)
    elif kind == "overclaim":
        sentence = perturb.overclaim(text, texts)
    elif kind == "population":
        sentence = perturb.swap_population(text, texts)
    elif kind == "other_drug":
        other = rng.choice([d for d in drugs if d.ingredient_key != drug.ingredient_key])
        oc = rng.choice(
            [c for c in other.chunks if c.section_loinc == cited.section_loinc] or [None]
        )
        if oc is None:
            return None
        ospans = usable_span_texts(oc)
        if not ospans:
            return None
        _, sentence = rng.choice(ospans)
        # The moved sentence must say something the evidence does not: at least two content
        # words absent from it, and no mention of the other drug's name (a writer's error would
        # not carry the name along).
        if _first_word(other.title) in sentence.lower():
            return None
        if len(_tokens(sentence) - _tokens(" ".join(texts))) < 2:
            return None
        meta["moved_from"] = other.title
    else:
        raise ValueError(kind)
    if not sentence or (kind not in ("verbatim", "join", "tallman") and sentence == text):
        return None
    used.add((cited.chunk_id, sp.i))
    return Claim(
        id="",
        sentence=sentence,
        citations=[cited_id],
        evidence=evidence,
        label=LABEL_OF[kind],  # type: ignore[arg-type]
        kind=kind,
        meta=meta,
    )


def build_claims(drugs: list[DrugChunks], rng: random.Random) -> list[Claim]:
    used: set[tuple[int, int]] = set()
    by_kind: dict[str, list[Claim]] = defaultdict(list)
    for kind, n in PLAN.items():
        attempts = 0
        while len(by_kind[kind]) < n:
            attempts += 1
            if attempts > 20000:
                raise RuntimeError(f"could not build {n} claims of kind {kind}")
            claim = make_claim(kind, drugs, rng, used)
            if claim is not None:
                by_kind[kind].append(claim)
    claims = [c for kind in PLAN for c in by_kind[kind]]
    rng.shuffle(claims)
    for i, c in enumerate(claims, 1):
        c.id = f"d{i:03d}"
    return claims


async def build_dev(conn: AsyncConnection, seed: int = SEED) -> list[Claim]:
    rng = random.Random(seed)
    drugs = await load_drugs(conn, rng, N_DRUGS)
    return build_claims(drugs, rng)

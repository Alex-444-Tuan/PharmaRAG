"""Grounding verifier (spec section 9, "Verifier, per sentence").

For each cited sentence: pick candidate spans, run the deterministic number check, then score
(premise, sentence) pairs with an NLI cross-encoder in one batch and decide a verdict.

    prepare(evidence, facts)            embed every span once per generation
    verify_sentence(sentence, prep)     one verdict; safe to call concurrently
    verify_many(sentences, prep)        several sentences in ONE NLI batch, same results
    verify(sentences, evidence, facts)  sequential convenience

All model work runs in worker threads (`asyncio.to_thread`), so the event loop is never blocked.
The NLI model's tokenizer and torch module are not safe to enter from several threads at once, so
it sits behind a lock. (The embedder has its own lock, in `SentenceTransformerEmbedder`.)

The premise construction (`premise_context`, `premise_heading`) and the candidate count are read
from `config/thresholds.yaml`, so the threshold-freeze rule covers them. Constructor overrides
exist for the tuning and latency tools only; `--score` refuses them.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from pharmarag.app.settings import GroundingThresholds
from pharmarag.graph.state import Chunk, Fact, Sentence, Verdict
from pharmarag.grounding.numeric import Quantity, missing_quantities
from pharmarag.grounding.spans import SpanRecord, build_spans, select_candidates
from pharmarag.models.embedder import Embedder

REASON_NUMERIC = "numeric_mismatch"
REASON_NO_CITATION = "no_citation"
REASON_NO_EVIDENCE = "no_evidence"
REASON_CONTRADICTED = "nli_contradiction"
REASON_LOW_ENTAILMENT = "low_entailment"


class NLIScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """(premise, hypothesis) pairs -> (p_entailment, p_contradiction), each shape (n,)."""
        ...


def resolve_label_indices(id2label: dict) -> tuple[int, int]:
    """(entailment index, contradiction index) read from the model's own label map.

    The order differs between NLI models (this one is contradiction, entailment, neutral;
    others put entailment first), so it is never assumed. A model whose labels cannot be
    identified is an error, not a guess.
    """
    by_name = {str(v).strip().lower(): int(k) for k, v in id2label.items()}
    try:
        return by_name["entailment"], by_name["contradiction"]
    except KeyError:
        raise ValueError(
            f"NLI model labels {sorted(by_name)} do not include 'entailment' and 'contradiction'"
        ) from None


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class CrossEncoderNLI:
    """`sentence_transformers.CrossEncoder` NLI model, one batch per call, one caller at a time."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        model=None,
        revision: str | None = None,
    ) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            # `revision` pins the Hugging Face commit, so a moved `main` cannot change scores.
            model = CrossEncoder(model_name, device="cpu", max_length=max_length, revision=revision)
        self.model_name = model_name
        self._model = model
        self.entail_idx, self.contra_idx = resolve_label_indices(model.model.config.id2label)
        self._lock = threading.Lock()

    def score(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        if not pairs:
            return np.zeros(0), np.zeros(0)
        with self._lock:
            logits = self._model.predict(
                pairs, batch_size=len(pairs), activation_fn=_identity, show_progress_bar=False
            )
        probs = softmax(np.asarray(logits, dtype=np.float64).reshape(len(pairs), -1))
        return probs[:, self.entail_idx], probs[:, self.contra_idx]


def _identity(x):
    return x


# Tall-man lettering writes look-alike drug names with capitals inside the word ("PredniSONE",
# "PARoxetine", "hydrOXYzine"). The NLI model reads "PredniSONE" and "prednisone" as different
# words and answers "contradiction" (0.97 on a real label). Mixed-case tokens with a run of two
# or more capitals are folded to lower case, in the premise and the hypothesis alike, so they
# can only agree. All-capital acronyms (HIV, CYP2D6) and ordinary capitalised words are left.
_TALLMAN = re.compile(r"\b(?=[A-Za-z]*[a-z])[A-Za-z]*[A-Z]{2,}[A-Za-z]*\b")


def fold_tallman(text: str) -> str:
    return _TALLMAN.sub(lambda m: m.group(0).lower(), text)


@dataclass
class PreparedSpans:
    """What `prepare` builds: span records and their embeddings, immutable afterwards."""

    records: list[SpanRecord]
    matrix: np.ndarray  # (n_spans, dim), L2-normalised
    sources: frozenset[str] = field(default_factory=frozenset)


@dataclass
class PremiseScore:
    """One (premise, sentence) pair after NLI: which span, the two probabilities, and whether the
    sentence's quantities (number with unit) all appear in the premise text itself."""

    span: str
    entail: float
    contra: float
    numbers_ok: bool = True


@dataclass
class Analysis:
    """Everything the verifier learned about one sentence, before thresholds are applied.

    The tuning code sweeps thresholds over saved analyses without re-running the models.
    `missing` is the pool-level number check: quantities found in none of the candidate spans.
    A sentence with `missing` never reaches the NLI model, so `premises` is empty.
    """

    candidates: list[str]  # span ids, best cosine first
    missing: list[Quantity]
    premises: list[PremiseScore] = field(default_factory=list)

    @property
    def best_entail(self) -> float | None:
        return max((p.entail for p in self.premises), default=None)

    @property
    def best_entail_span(self) -> str | None:
        return max(self.premises, key=lambda p: p.entail).span if self.premises else None

    @property
    def best_contra(self) -> float | None:
        return max((p.contra for p in self.premises), default=None)

    @property
    def best_contra_span(self) -> str | None:
        return max(self.premises, key=lambda p: p.contra).span if self.premises else None


def _supporting(a: Analysis, thresholds: GroundingThresholds) -> list[PremiseScore]:
    """Premises that entail the sentence at `entail_min` AND contain its quantities."""
    return [p for p in a.premises if p.entail >= thresholds.entail_min and p.numbers_ok]


def decide(a: Analysis, thresholds: GroundingThresholds) -> tuple[Verdict, str | None]:
    """Verdict and reason from an analysis (spec section 9, step 4).

    Supported needs one premise that entails the sentence at `entail_min` and that itself states
    every number-with-unit in the sentence. Checking the numbers against the premise that
    entails, and not against the whole candidate pool, stops a dose from passing on the strength
    of some other span (in a two-drug answer, the other drug's) while NLI is satisfied by a span
    that does not contain it.
    """
    if not a.candidates:
        return "unsupported", REASON_NO_EVIDENCE
    if a.missing:
        return "unsupported", REASON_NUMERIC
    if _supporting(a, thresholds):
        return "supported", None
    if any(p.entail >= thresholds.entail_min for p in a.premises):
        return "unsupported", REASON_NUMERIC  # entailed, but not by a premise with the numbers
    if (a.best_contra or 0.0) >= thresholds.contradict_min:
        return "contradicted", REASON_CONTRADICTED
    return "unsupported", REASON_LOW_ENTAILMENT


def verdict_fields(a: Analysis, thresholds: GroundingThresholds) -> dict:
    """verdict, reason, score and span for a `Sentence`. Score is the best entailment (of the
    supporting premises when supported); span is the premise that drove the verdict."""
    verdict, reason = decide(a, thresholds)
    if verdict == "supported":
        best = max(_supporting(a, thresholds), key=lambda p: p.entail)
        score, span = best.entail, best.span
    elif verdict == "contradicted":
        score, span = a.best_entail, a.best_contra_span
    else:
        score, span = a.best_entail, a.best_entail_span
    return {"verdict": verdict, "score": score, "span": span, "reason": reason}


def apply_analysis(sentence: Sentence, a: Analysis, thresholds: GroundingThresholds) -> Sentence:
    return sentence.model_copy(update=verdict_fields(a, thresholds))


# What the NLI model reads as the premise for one candidate span:
#   "neighbour"  the span plus one neighbour sentence (the spec's design)
#   "span"       the span alone
#   "both"       every premise in one batch (neighbour, span, union); the best counts
#   "cascade"    span plus neighbour first; then the span alone; then the candidate spans read
#                together ("union"); each stage only for sentences the earlier ones did not support
# "cascade" gives the same verdict as "both" at the same entail_min (supported iff some premise
# supports), and costs a single batch for sentences the first premise already supports. It is
# not a valid mode for threshold tuning, because its later stages depend on entail_min: tune
# with "both".
ContextMode = Literal["neighbour", "span", "both", "cascade"]
# Whether the label heading leads the premise: none, the subsection title (or the section title
# when there is none), or "section > subsection".
HeadingMode = Literal["none", "sub", "full"]

# "union" is the candidate spans read together as one premise. A sentence that merges two facts of
# the label ("X is 10 mg, and Y is contraindicated") is entailed by the two spans jointly, never
# by either alone, so the single-span premises reject it. The union is one extra pair per
# sentence, tried last and only for sentences nothing else supported.
_STAGES: dict[str, list[tuple[str, ...]]] = {
    "neighbour": [("neighbour",)],
    "span": [("span",)],
    "both": [("neighbour", "span", "union")],  # one batch with every premise
    "cascade": [("neighbour",), ("span",), ("union",)],
}


@dataclass
class _Work:
    """One sentence in flight through `_analyse_many_sync`."""

    sentence: Sentence
    cands: list[SpanRecord]
    analysis: Analysis
    settled: bool = False


class GroundingVerifier:
    def __init__(
        self,
        embedder: Embedder,
        nli: NLIScorer,
        thresholds: GroundingThresholds,
        *,
        context: ContextMode | None = None,
        heading: HeadingMode | None = None,
    ) -> None:
        """`context` and `heading` default to the frozen `premise_context` / `premise_heading`.
        The overrides are for the tuning and latency tools; scoring refuses them."""
        self.embedder = embedder
        self.nli = nli
        self.thresholds = thresholds
        self.context: ContextMode = context or thresholds.premise_context
        self.heading: HeadingMode = heading or thresholds.premise_heading

    # --- model work (runs in worker threads) ---

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)

    @staticmethod
    def _body(r: SpanRecord, kind: str) -> str:
        return r.text if kind == "span" else r.context

    def _premise(self, r: SpanRecord, kind: str) -> str:
        """The premise text for one candidate: the span or span-plus-neighbour, optionally led by
        the label heading ("2.1 Adult Dosage: ..."). A writer says "in adults" because the heading
        does; without it the NLI model sees the claim as adding a fact."""
        body = self._body(r, kind)
        if self.heading == "none":
            return body
        head = r.subsection or r.section
        if self.heading == "full" and r.subsection and r.section:
            head = f"{r.section} > {r.subsection}"
        return f"{head}: {body}" if head else body

    def _premises(self, cands: Sequence[SpanRecord], kind: str) -> list[tuple[str, str, str]]:
        """(span id, premise text, premise body) for one sentence and one premise kind.

        `neighbour` and `span` give one premise per candidate; `union` gives one premise for all
        candidates together (when there are at least two). The body is the text without the
        heading, which is what the number check reads.
        """
        if kind == "union":
            if len(cands) < 2:
                return []
            ids = "+".join(r.id for r in cands)
            return [
                (
                    ids,
                    " ".join(self._premise(r, "span") for r in cands),
                    " ".join(r.text for r in cands),
                )
            ]
        return [(r.id, self._premise(r, kind), self._body(r, kind)) for r in cands]

    def _prepare_sync(self, evidence: Sequence[Chunk], facts: Sequence[Fact]) -> PreparedSpans:
        records = build_spans(evidence, facts)
        if records:
            matrix = self._embed([r.text for r in records])
        else:
            matrix = np.zeros((0, getattr(self.embedder, "dim", 0)), dtype=np.float32)
        return PreparedSpans(records, matrix, frozenset(r.source for r in records))

    def _analyse_many_sync(
        self, sentences: Sequence[Sentence], prepared: PreparedSpans
    ) -> list[Analysis]:
        """Analyse several sentences with one embedding call and one NLI batch per stage."""
        records = prepared.records
        if not records:
            return [Analysis(candidates=[], missing=[]) for _ in sentences]
        vecs = self._embed([s.text for s in sentences])
        work: list[_Work] = []
        for s, vec in zip(sentences, vecs, strict=True):
            idx = select_candidates(
                vec,
                prepared.matrix,
                records,
                s.citations,
                nearest_spans=self.thresholds.nearest_spans,
                candidate_spans=self.thresholds.candidate_spans,
            )
            cands = [records[i] for i in idx]
            missing = missing_quantities(s.text, (r.text for r in cands))
            # A number found in no candidate span is final; such a sentence skips the NLI model.
            work.append(_Work(s, cands, Analysis([r.id for r in cands], missing), bool(missing)))
        for kinds in _STAGES[self.context]:
            active = [w for w in work if not w.settled]
            if not active:
                break
            pairs: list[tuple[str, str]] = []
            owners: list[tuple[_Work, str, str]] = []  # (sentence, span id, premise body)
            for w in active:
                hypothesis = fold_tallman(w.sentence.text)
                for kind in kinds:
                    for span_id, premise, body in self._premises(w.cands, kind):
                        pairs.append((fold_tallman(premise), hypothesis))
                        owners.append((w, span_id, body))
            entail, contra = self.nli.score(pairs)  # one batch per stage, for every sentence
            for (w, span_id, body), e, c in zip(owners, entail, contra, strict=True):
                ok = not missing_quantities(w.sentence.text, [body])
                w.analysis.premises.append(PremiseScore(span_id, float(e), float(c), ok))
            if self.context == "cascade":
                # a supporting premise settles the sentence; the rest take the next stage
                for w in active:
                    w.settled = bool(_supporting(w.analysis, self.thresholds))
        return [w.analysis for w in work]

    # --- Verifier protocol ---

    async def prepare(self, evidence: list[Chunk], facts: list[Fact]) -> PreparedSpans:
        return await asyncio.to_thread(self._prepare_sync, evidence, facts)

    async def analyse_many(
        self, sentences: Sequence[Sentence], prepared: PreparedSpans
    ) -> list[Analysis]:
        if not sentences:
            return []
        return await asyncio.to_thread(self._analyse_many_sync, list(sentences), prepared)

    async def analyse_sentence(self, sentence: Sentence, prepared: PreparedSpans) -> Analysis:
        return (await self.analyse_many([sentence], prepared))[0]

    @staticmethod
    def _precheck(sentence: Sentence) -> Sentence | None:
        """The verdict for a sentence that needs no model, or None when it does."""
        if sentence.verdict == "unsupported":
            return sentence  # already rejected upstream, e.g. reason "no_citation"
        if not sentence.citations:
            # FR-5 defence in depth: the writer already marks these, but a caller that did not
            # must not get an uncited sentence past the verifier.
            return sentence.model_copy(
                update={"verdict": "unsupported", "reason": REASON_NO_CITATION}
            )
        return None

    async def verify_many(
        self, sentences: list[Sentence], prepared: PreparedSpans
    ) -> list[Sentence]:
        """Same results and order as `verify_sentence` on each, with one NLI batch per stage."""
        out: list[Sentence | None] = [self._precheck(s) for s in sentences]
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            analyses = await self.analyse_many([sentences[i] for i in todo], prepared)
            for i, a in zip(todo, analyses, strict=True):
                out[i] = apply_analysis(sentences[i], a, self.thresholds)
        return [o for o in out if o is not None]

    async def verify_sentence(self, sentence: Sentence, prepared: PreparedSpans) -> Sentence:
        return (await self.verify_many([sentence], prepared))[0]

    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        prepared = await self.prepare(evidence, facts)
        return [await self.verify_sentence(s, prepared) for s in sentences]

"""Grounding verifier (spec section 9, "Verifier, per sentence").

For each cited sentence: pick candidate spans, run the deterministic number check, then score
(span, sentence) pairs with an NLI cross-encoder in one batch and decide a verdict.

    prepare(evidence, facts)         embed every span once per generation
    verify_sentence(sentence, prep)  one verdict; safe to call concurrently
    verify(sentences, evidence, facts)  sequential convenience

All model work runs in worker threads (`asyncio.to_thread`), so the event loop is never blocked.
Tokenizers and torch modules are not safe to enter from several threads at once, so each model
sits behind its own lock: two sentences can overlap an embedding with an NLI batch, but never two
NLI batches.
"""

from __future__ import annotations

import asyncio
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

    def __init__(self, model_name: str, *, max_length: int = 512, model=None) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device="cpu", max_length=max_length)
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


@dataclass
class PreparedSpans:
    """What `prepare` builds: span records and their embeddings, immutable afterwards."""

    records: list[SpanRecord]
    matrix: np.ndarray  # (n_spans, dim), L2-normalised
    sources: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Analysis:
    """Everything the verifier learned about one sentence, before thresholds are applied.

    The tuning code sweeps thresholds over saved analyses without re-running the models.
    """

    candidates: list[str]  # span ids, best cosine first
    missing: list[Quantity]
    best_entail: float | None = None
    best_entail_span: str | None = None
    best_contra: float | None = None
    best_contra_span: str | None = None


def decide(a: Analysis, thresholds: GroundingThresholds) -> tuple[Verdict, str | None]:
    """Verdict and reason from an analysis (spec section 9, step 4)."""
    if not a.candidates:
        return "unsupported", REASON_NO_EVIDENCE
    if a.missing:
        return "unsupported", REASON_NUMERIC
    entail = a.best_entail or 0.0
    contra = a.best_contra or 0.0
    if entail >= thresholds.entail_min:
        return "supported", None
    if contra >= thresholds.contradict_min:
        return "contradicted", REASON_CONTRADICTED
    return "unsupported", REASON_LOW_ENTAILMENT


def apply_analysis(sentence: Sentence, a: Analysis, thresholds: GroundingThresholds) -> Sentence:
    verdict, reason = decide(a, thresholds)
    span = a.best_contra_span if verdict == "contradicted" else a.best_entail_span
    return sentence.model_copy(
        update={"verdict": verdict, "score": a.best_entail, "span": span, "reason": reason}
    )


# What the NLI model reads as the premise for one candidate span:
#   "neighbour"  the span plus one neighbour sentence (the spec's design)
#   "span"       the span alone
#   "both"       both premises in one batch, so twice the pairs; the best of the two counts
#   "cascade"    span plus neighbour first; the span alone only if that does not entail
# "cascade" gives the same verdict as "both" at the same entail_min (supported iff either premise
# entails), and costs a single batch for sentences the first premise already supports. It is not
# a valid mode for threshold tuning, because its second stage depends on entail_min: tune with
# "both".
ContextMode = Literal["neighbour", "span", "both", "cascade"]
# Whether the label heading leads the premise: none, the subsection title (or the section title
# when there is none), or "section > subsection".
HeadingMode = Literal["none", "sub", "full"]


class GroundingVerifier:
    def __init__(
        self,
        embedder: Embedder,
        nli: NLIScorer,
        thresholds: GroundingThresholds,
        *,
        context: ContextMode = "neighbour",
        heading: HeadingMode = "none",
    ) -> None:
        self.embedder = embedder
        self.nli = nli
        self.thresholds = thresholds
        self.context = context
        self.heading = heading
        self._embed_lock = threading.Lock()

    # --- model work (runs in worker threads) ---

    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._embed_lock:
            return np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)

    def _premise(self, r: SpanRecord, kind: str) -> str:
        """The premise text for one candidate: the span or span-plus-neighbour, optionally led by
        the label heading ("2.1 Adult Dosage: ..."). A writer says "in adults" because the heading
        does; without it the NLI model sees the claim as adding a fact."""
        body = r.text if kind == "span" else r.context
        if self.heading == "none":
            return body
        head = r.subsection or r.section
        if self.heading == "full" and r.subsection and r.section:
            head = f"{r.section} > {r.subsection}"
        return f"{head}: {body}" if head else body

    def _prepare_sync(self, evidence: Sequence[Chunk], facts: Sequence[Fact]) -> PreparedSpans:
        records = build_spans(evidence, facts)
        if records:
            matrix = self._embed([r.text for r in records])
        else:
            matrix = np.zeros((0, getattr(self.embedder, "dim", 0)), dtype=np.float32)
        return PreparedSpans(records, matrix, frozenset(r.source for r in records))

    def _analyse_sync(self, sentence: Sentence, prepared: PreparedSpans) -> Analysis:
        records = prepared.records
        if not records:
            return Analysis(candidates=[], missing=[])
        vec = self._embed([sentence.text])[0]
        idx = select_candidates(
            vec,
            prepared.matrix,
            records,
            sentence.citations,
            nearest_spans=self.thresholds.nearest_spans,
            candidate_spans=self.thresholds.candidate_spans,
        )
        cands = [records[i] for i in idx]
        missing = missing_quantities(sentence.text, (r.text for r in cands))
        analysis = Analysis(candidates=[r.id for r in cands], missing=missing)
        if missing:
            return analysis  # a wrong number is final; no NLI pass needed
        stages = {
            "neighbour": [("neighbour",)],
            "span": [("span",)],
            "both": [("span", "neighbour")],  # one batch with both premises per candidate
            "cascade": [("neighbour",), ("span",)],
        }[self.context]
        entail = np.zeros(0)
        contra = np.zeros(0)
        owner: list[SpanRecord] = []
        for kinds in stages:
            pairs: list[tuple[str, str]] = []
            for r in cands:
                for kind in kinds:
                    pairs.append((self._premise(r, kind), sentence.text))
                    owner.append(r)
            e_new, c_new = self.nli.score(pairs)  # one batch per stage
            entail, contra = np.concatenate([entail, e_new]), np.concatenate([contra, c_new])
            # Cascade: a premise that entails the sentence settles it as supported, so the second
            # pass (the span alone) runs only when the first (span plus neighbour) did not.
            if self.context == "cascade" and float(entail.max()) >= self.thresholds.entail_min:
                break
        e, c = int(np.argmax(entail)), int(np.argmax(contra))
        analysis.best_entail, analysis.best_entail_span = float(entail[e]), owner[e].id
        analysis.best_contra, analysis.best_contra_span = float(contra[c]), owner[c].id
        return analysis

    # --- Verifier protocol ---

    async def prepare(self, evidence: list[Chunk], facts: list[Fact]) -> PreparedSpans:
        return await asyncio.to_thread(self._prepare_sync, evidence, facts)

    async def analyse_sentence(self, sentence: Sentence, prepared: PreparedSpans) -> Analysis:
        return await asyncio.to_thread(self._analyse_sync, sentence, prepared)

    async def verify_sentence(self, sentence: Sentence, prepared: PreparedSpans) -> Sentence:
        if sentence.verdict == "unsupported":
            return sentence  # already rejected upstream, e.g. reason "no_citation"
        if not sentence.citations:
            # FR-5 defence in depth: the writer already marks these, but a caller that did not
            # must not get an uncited sentence past the verifier.
            return sentence.model_copy(
                update={"verdict": "unsupported", "reason": REASON_NO_CITATION}
            )
        analysis = await self.analyse_sentence(sentence, prepared)
        return apply_analysis(sentence, analysis, self.thresholds)

    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        prepared = await self.prepare(evidence, facts)
        return [await self.verify_sentence(s, prepared) for s in sentences]

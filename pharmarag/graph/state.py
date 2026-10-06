"""Graph state and the value types every lane shares (spec section 6).

Contract file: owned by the orchestrator. Lanes read it, never edit it.
"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field

Intent = Literal["label_qa", "interaction", "class_question", "patient_advice", "out_of_scope"]

Status = Literal[
    "answered",
    "needs_clarification",
    "drug_not_recognised",
    "not_covered",
    "insufficient_evidence",
    "declined",
]

MatchKind = Literal["exact", "brand", "salt", "approximate", "pinned"]

Verdict = Literal["supported", "unsupported", "contradicted", "unverified"]


class Flags(BaseModel):
    """Eval-only switches (FR-9). Honoured only when EVAL_MODE=true."""

    resolver: bool = True
    verifier: bool = True


class Candidate(BaseModel):
    ingredient_key: str
    name: str
    score: float


class Resolution(BaseModel):
    """Outcome of resolving one drug mention (spec section 7)."""

    mention: str
    status: Literal["resolved", "ambiguous", "unknown", "not_covered"]
    match: MatchKind | None = None
    ingredient_key: str | None = None  # sorted ingredient RxCUIs joined by '+'
    name: str | None = None  # display name of the resolved ingredient(s)
    ingredient_rxcuis: list[str] = Field(default_factory=list)
    product_hint: str | None = None  # brand/salt RxCUI: a tie-break, never a filter
    score: float | None = None
    candidates: list[Candidate] = Field(default_factory=list)  # up to 3 when ambiguous


class Span(BaseModel):
    """A sentence or table row inside a chunk, by character offsets into chunk.text."""

    i: int
    start: int
    end: int


class Hit(BaseModel):
    """One ranked chunk from a retrieval branch. Every hit carries its attempt number."""

    chunk_id: int
    setid: UUID
    spl_version: int
    ingredient_key: str
    section_loinc: str
    section_title: str
    subsection_title: str | None = None
    text: str
    spans: list[Span]
    effective_date: date | None = None
    source: Literal["dense", "lexical"]
    rank: int  # 1-based, within its branch
    score: float
    retrieval_attempt: int


class Fact(BaseModel):
    """One RxNorm fact line (brands, dose forms, strengths). Citable as r1, r2, ..."""

    id: str  # "r1", "r2", ...
    rxcui: str
    text: str
    retrieval_attempt: int


class Chunk(BaseModel):
    """A fused evidence chunk passed to the writer. Citable as c1..c8."""

    id: str  # "c1", "c2", ...
    chunk_id: int
    setid: UUID
    spl_version: int
    ingredient_key: str
    section_loinc: str
    section_title: str
    subsection_title: str | None = None
    text: str
    spans: list[Span]
    effective_date: date | None = None
    fused_score: float


class Sentence(BaseModel):
    text: str
    citations: list[str]  # e.g. ["c2", "r1"]
    verdict: Verdict = "unverified"
    score: float | None = None  # best entailment probability
    span: str | None = None  # best supporting span, e.g. "c2:4"
    reason: str | None = None  # e.g. "numeric_mismatch", "no_citation"


def merge_dicts(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {**left, **right}


class RAGState(TypedDict, total=False):
    question: str
    flags: Flags
    pinned_rxcuis: list[str]
    request_id: str
    intent: Intent
    mentions: list[str]
    sections: list[str]
    resolutions: list[Resolution]
    ingredient_keys: list[str]
    search_query: str
    retrieval_attempt: int  # starts at 1
    hits: Annotated[list[Hit], operator.add]
    rx_facts: Annotated[list[Fact], operator.add]
    evidence: list[Chunk]
    evidence_ok: bool
    generation_attempt: int  # starts at 1
    sentences: list[Sentence]
    removed_sentences: list[Sentence]
    support_ratio: float
    status: Status
    response: dict
    timings: Annotated[dict[str, float], merge_dicts]

"""Claim sets for the grounding eval (spec section 11.2): schema, JSONL IO and file hashes.

A claim is one writer sentence plus the evidence snapshot it was written from, so the verifier
can be re-run later without the LLM, the retriever or the database.

Files in pharmarag/eval/datasets/:

    grounding_dev.jsonl             synthetic dev claims, label S or H by construction
    grounding_questions.jsonl       the 14 fixed test questions
    grounding_claims_test.jsonl     40 natural writer sentences, frozen (SHA pinned in a test)
    grounding_labels_test.jsonl     the human labels, imported from the worksheet
    grounding_topup_test.jsonl      perturbed replacements, only if the top-up rule fired
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pharmarag.app.settings import REPO_ROOT
from pharmarag.graph.state import Chunk, Fact

DATASETS = REPO_ROOT / "pharmarag" / "eval" / "datasets"
DEV_CLAIMS = DATASETS / "grounding_dev.jsonl"
QUESTIONS = DATASETS / "grounding_questions.jsonl"
TEST_CLAIMS = DATASETS / "grounding_claims_test.jsonl"
TEST_LABELS = DATASETS / "grounding_labels_test.jsonl"
TEST_TOPUP = DATASETS / "grounding_topup_test.jsonl"

Label = Literal["S", "H"]  # supported / hallucinated


class Claim(BaseModel):
    """One sentence to verify, with the evidence the writer had when it wrote it."""

    id: str
    question: str | None = None
    sentence: str
    citations: list[str]
    evidence: list[Chunk]
    facts: list[Fact] = Field(default_factory=list)
    # Dev claims carry their by-construction label. Test claims never do: their labels live in
    # grounding_labels_test.jsonl, written by the human.
    label: Label | None = None
    kind: str = "natural"  # dev: verbatim, rewrite, numeric, negation, ...; test: natural
    meta: dict = Field(default_factory=dict)


class Question(BaseModel):
    id: str
    question: str
    answerable: bool  # False: the label cannot answer it (the writer should say INSUFFICIENT)
    drug: str


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict | BaseModel]) -> None:
    lines = []
    for r in rows:
        d = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
        lines.append(json.dumps(d, ensure_ascii=False, sort_keys=False))
    path.write_text("\n".join(lines) + "\n")


def load_claims(path: Path) -> list[Claim]:
    return [Claim.model_validate(r) for r in load_jsonl(path)]


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [Question.model_validate(r) for r in load_jsonl(path)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

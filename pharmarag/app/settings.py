"""Environment configuration (spec section 12) and frozen thresholds (config/thresholds.yaml).

Contract file: owned by the orchestrator. Lanes read it, never edit it.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/pharmarag"
    redis_url: str = "redis://localhost:6379/0"

    llm_provider: Literal["anthropic", "fake"] = "anthropic"
    llm_model: str = ""  # pinned model ID, never an alias
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 500

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    verify_mode: Literal["off", "sequential", "stream"] = "stream"

    rxnav_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    dailymed_base_url: str = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    labels_per_ingredient: int = 1

    eval_mode: bool = False
    log_questions: bool = False
    rate_limit_per_minute: int = 30

    rxnav_timeout_s: float = 2.0
    llm_timeout_s: float = 30.0
    db_timeout_s: float = 5.0

    thresholds_path: Path = REPO_ROOT / "config" / "thresholds.yaml"


class ResolverThresholds(BaseModel):
    resolved_min: float
    margin: float
    ambiguous_min: float
    trigram_min: float
    trigram_k: int
    max_candidates: int


class RetrievalThresholds(BaseModel):
    top_k_per_branch: int
    rrf_k: int
    evidence_top_k: int
    evidence_top_k_per_drug: int
    dense_min: float
    overlap_min: int
    max_facts: int


class GroundingThresholds(BaseModel):
    entail_min: float
    contradict_min: float
    support_ratio_min: float
    candidate_spans: int
    nearest_spans: int


class GraphLimits(BaseModel):
    max_retrieval_attempts: int
    max_generation_attempts: int
    recursion_limit: int


class Thresholds(BaseModel):
    resolver: ResolverThresholds
    retrieval: RetrievalThresholds
    grounding: GroundingThresholds
    graph: GraphLimits


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_thresholds(path: Path | None = None) -> Thresholds:
    path = path or get_settings().thresholds_path
    return Thresholds.model_validate(yaml.safe_load(path.read_text()))

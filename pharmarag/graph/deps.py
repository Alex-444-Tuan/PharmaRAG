"""Runtime dependencies shared by graph nodes. Built once at startup, never per request."""

from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.llm.base import LLMClient
from pharmarag.models.embedder import Embedder


@dataclass
class Deps:
    llm: LLMClient
    embedder: Embedder
    pool: AsyncConnectionPool
    settings: Settings
    thresholds: Thresholds
    # Cycle 1 has no resolver node and no verifier. `run_ask` forces the matching request flag
    # off when a capability is missing, so a missing stage can never be mistaken for a passed
    # one (FR-3, FR-6). Cycles 2 and 3 flip these to True when they wire the real stages.
    resolver_available: bool = False
    verifier_available: bool = False

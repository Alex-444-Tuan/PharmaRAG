"""Runtime dependencies shared by graph nodes. Built once at startup, never per request."""

from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

from pharmarag.app.observability import Metrics
from pharmarag.app.settings import Settings, Thresholds
from pharmarag.cache import Cache, InMemoryCache
from pharmarag.grounding.base import Verifier
from pharmarag.llm.base import LLMClient
from pharmarag.models.embedder import Embedder
from pharmarag.ops import QueryEmbeddingCache
from pharmarag.resolver.base import Resolver, RxFactsSource


@dataclass
class Deps:
    llm: LLMClient
    embedder: Embedder
    pool: AsyncConnectionPool
    settings: Settings
    thresholds: Thresholds
    # A stage that is None is missing from this build (the app degrades when a factory is not
    # implemented yet). `run_ask` forces the matching request flag off, so a missing stage can
    # never be mistaken for a passed one (FR-3, FR-6).
    resolver: Resolver | None = None
    rx_facts: RxFactsSource | None = None
    verifier: Verifier | None = None
    cache: Cache = field(default_factory=InMemoryCache)
    # None: embed with `embedder` directly (no cache in this build, or EVAL_MODE without one).
    query_embeddings: QueryEmbeddingCache | None = None
    # None: nothing is counted (unit tests that do not look at metrics).
    metrics: Metrics | None = None

    @property
    def resolver_available(self) -> bool:
        return self.resolver is not None

    @property
    def verifier_available(self) -> bool:
        return self.verifier is not None

"""Runtime dependencies shared by graph nodes. Built once at startup, never per request."""

from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.cache import Cache, InMemoryCache
from pharmarag.grounding.base import Verifier
from pharmarag.llm.base import LLMClient
from pharmarag.models.embedder import Embedder
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

    @property
    def resolver_available(self) -> bool:
        return self.resolver is not None

    @property
    def verifier_available(self) -> bool:
        return self.verifier is not None

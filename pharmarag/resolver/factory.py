"""Builders for the request-time resolver and RxNorm facts source.

The signatures are the contract: the app lifespan calls these. Lane A owns the
bodies (cycle 2) and replaces the NotImplementedError stubs.
"""

from psycopg_pool import AsyncConnectionPool

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.cache import Cache
from pharmarag.resolver.base import Resolver, RxFactsSource


def build_resolver(
    settings: Settings, thresholds: Thresholds, pool: AsyncConnectionPool, cache: Cache
) -> Resolver:
    raise NotImplementedError("cycle 2, lane A")


def build_rx_facts_source(
    settings: Settings, thresholds: Thresholds, cache: Cache
) -> RxFactsSource:
    raise NotImplementedError("cycle 2, lane A")

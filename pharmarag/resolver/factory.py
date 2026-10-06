"""Builders for the request-time resolver and RxNorm facts source.

The signatures are the contract: the app lifespan calls these. Lane A owns the bodies.
"""

from psycopg_pool import AsyncConnectionPool

from pharmarag.app.settings import Settings, Thresholds
from pharmarag.cache import Cache
from pharmarag.resolver.base import Resolver, RxFactsSource
from pharmarag.resolver.cascade import CascadeResolver
from pharmarag.resolver.facts import RxNavFactsSource
from pharmarag.resolver.local import PgLocalStore
from pharmarag.resolver.rxnav_client import RxNavClient

# One RxNav client per Settings object, shared by the resolver and the facts source, so both
# draw from one 20 requests/s budget, one circuit breaker and one concept memo. Settings is
# not hashable, hence the id key; the value keeps the Settings alive so the id is never reused.
_clients: dict[int, tuple[Settings, RxNavClient]] = {}


def _rxnav_for(settings: Settings) -> RxNavClient:
    held = _clients.get(id(settings))
    if held is None:
        held = (settings, RxNavClient(settings.rxnav_base_url, timeout_s=settings.rxnav_timeout_s))
        _clients[id(settings)] = held
    return held[1]


def build_resolver(
    settings: Settings, thresholds: Thresholds, pool: AsyncConnectionPool, cache: Cache
) -> Resolver:
    return CascadeResolver(PgLocalStore(pool), _rxnav_for(settings), thresholds.resolver, cache)


def build_rx_facts_source(
    settings: Settings, thresholds: Thresholds, cache: Cache
) -> RxFactsSource:
    return RxNavFactsSource(_rxnav_for(settings), cache, thresholds.retrieval.max_facts)


async def close_rxnav(settings: Settings) -> None:
    """Close the shared RxNav client of `settings`, if any. Call from the app shutdown."""
    held = _clients.pop(id(settings), None)
    if held is not None:
        await held[1].aclose()

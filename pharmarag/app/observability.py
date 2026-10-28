"""Structured JSON logs (NFR-5) and Prometheus metrics (spec section 12, "Operations").

Logs. One JSON line per graph node and one per HTTP request, both through the
`pharmarag.events` logger so a test can capture them. The question is never logged: only a hash
of it, unless `LOG_QUESTIONS=true`.

Metrics. A `Metrics` object owns its own `CollectorRegistry`, so two apps in one process (tests)
never collide on the global registry. It is carried on `Deps` and read by the nodes.
"""

import contextlib
import contextvars
import hashlib
import json
import logging
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Histogram

EVENT_LOGGER = "pharmarag.events"
RXNAV_LOGGER = "pharmarag.resolver.rxnav_client"
FACTS_LOGGER = "pharmarag.resolver.facts"
LOG_CONFIG_PATH = Path(__file__).with_name("log_config.json")

# What a client may send as X-Request-ID. Anything else is replaced, so a header cannot inject
# newlines or megabytes into the logs.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def valid_request_id(value: str | None) -> str | None:
    return value if value and _REQUEST_ID.fullmatch(value) else None


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def question_fields(question: str, *, log_questions: bool) -> dict:
    """The only way a question enters a log line (spec section 12, "Privacy")."""
    fields: dict = {"question_hash": question_hash(question)}
    if log_questions:
        fields["question"] = question
    return fields


# --- logs ------------------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Event records carry `fields`; plain records get a `message`."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
        }
        fields = getattr(record, "fields", None)
        if fields:
            out.update(fields)
        else:
            out["message"] = record.getMessage()
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Send every log record to stdout as JSON. Idempotent; called from the app lifespan, never
    at import, so importing the app does not rewire a test run's logging."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _configured = True


def log_event(event: str, **fields) -> None:
    logging.getLogger(EVENT_LOGGER).info(event, extra={"fields": {"event": event, **fields}})


# --- metrics ---------------------------------------------------------------------------------

_NODE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
# End to end. The NFR-1 targets (p50 4 s, p95 8 s) sit on bucket edges so they can be read off.
_REQUEST_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 30, 60)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.http_requests = Counter(
            "pharmarag_http_requests_total",
            "HTTP requests by route template and status code",
            ["path", "code"],
            registry=r,
        )
        self.ask = Counter(
            "pharmarag_ask_total",
            "Answered /v1/ask requests by response status",
            ["status"],
            registry=r,
        )
        self.node_duration = Histogram(
            "pharmarag_node_duration_seconds",
            "Wall time of one graph node run",
            ["node"],
            buckets=_NODE_BUCKETS,
            registry=r,
        )
        self.request_duration = Histogram(
            "pharmarag_request_duration_seconds",
            "End-to-end HTTP request time (to the last byte) by route template",
            ["path"],
            buckets=_REQUEST_BUCKETS,
            registry=r,
        )
        self.rxnav_errors = Counter(
            "pharmarag_rxnav_errors_total",
            "RxNav calls that failed after the client's own retries",
            registry=r,
        )
        self.cache = Counter(
            "pharmarag_cache_requests_total",
            "Cache lookups by cache and result",
            ["cache", "result"],
            registry=r,
        )
        self.rate_limited = Counter(
            "pharmarag_rate_limited_total",
            "Requests rejected with 429",
            registry=r,
        )
        # Touch every labelled series that has a known value set, so the exposition lists them
        # from the first scrape and `rate(...)` has a zero to start from.
        for result in ("hit", "miss"):
            for cache in ("answer", "query_embedding"):
                self.cache.labels(cache, result)

    def cache_result(self, cache: str, *, hit: bool) -> None:
        self.cache.labels(cache, "hit" if hit else "miss").inc()


@dataclass
class RxNavScope:
    """RxNav failures seen while a block ran (see `watch_rxnav`)."""

    errors: int = 0


_rxnav_scope: contextvars.ContextVar[RxNavScope | None] = contextvars.ContextVar(
    "rxnav_scope", default=None
)


@contextlib.contextmanager
def watch_rxnav() -> Iterator[RxNavScope]:
    """Within the block, RxNav failures logged by the resolver package are tallied on the scope.

    The facts source swallows RxNav outages on purpose (NFR-4) and returns no lines, so the
    `fetch_rx_facts` node cannot tell "no facts exist" from "RxNav was down". This is how it
    learns. The scope is a mutable object in a ContextVar: tasks started inside the block (the
    facts source gathers one per ingredient) copy the context but share the object, and two
    requests in flight have separate scopes.
    """
    scope = RxNavScope()
    token = _rxnav_scope.set(scope)
    try:
        yield scope
    finally:
        _rxnav_scope.reset(token)


class RxNavErrorCounter(logging.Handler):
    """Counts RxNav failures from the resolver package's own log lines.

    Neither the RxNav client nor the facts source (resolver/, not this lane's code) has a metrics
    hook, and both swallow outages by design (NFR-4), so their warnings are the one place that
    sees every failure. Two lines matter:

    * "rxnav unavailable": the client, once per call whose retries are spent. Feeds
      `pharmarag_rxnav_errors_total` and the active `watch_rxnav` scope.
    * "rx facts unavailable": the facts source, which also covers the circuit-breaker short
      circuit that the client does not log. Feeds the scope only, so a breaker-open outage still
      marks the answer as degraded without double counting the metric.

    The wording is a coupling to Lane A's code. `test_ops_graph.py` runs the real client and facts
    source against a 503, so a reworded message fails a test instead of silently zeroing the
    metric.
    """

    def __init__(self, metrics: Metrics) -> None:
        super().__init__(logging.WARNING)
        self._metrics = metrics

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        counted = message.startswith("rxnav unavailable")
        if counted:
            self._metrics.rxnav_errors.inc()
        if counted or message.startswith("rx facts unavailable"):
            scope = _rxnav_scope.get()
            if scope is not None:
                scope.errors += 1


def attach_rxnav_error_counter(metrics: Metrics) -> logging.Handler:
    handler = RxNavErrorCounter(metrics)
    for name in (RXNAV_LOGGER, FACTS_LOGGER):
        logging.getLogger(name).addHandler(handler)
    return handler


def detach_rxnav_error_counter(handler: logging.Handler) -> None:
    for name in (RXNAV_LOGGER, FACTS_LOGGER):
        logging.getLogger(name).removeHandler(handler)


# --- query-embedding cache hit detection -----------------------------------------------------


@dataclass
class CacheProbe:
    missed: bool = False


_probe: contextvars.ContextVar[CacheProbe | None] = contextvars.ContextVar("probe", default=None)


@contextlib.contextmanager
def probe_cache() -> Iterator[CacheProbe]:
    """Within the block, `MissProbeEmbedder` marks the probe when the model really runs.

    The query-embedding cache (caching/, not this lane's code) returns a vector and does not say
    whether it came from Redis. The cache only calls the embedder on a miss, so a flag set by the
    embedder tells hit from miss. The probe is a mutable object in a ContextVar: `to_thread`
    copies the context but not the object, so a flag set in the worker thread is seen here, and
    two concurrent requests have separate probes.
    """
    probe = CacheProbe()
    token = _probe.set(probe)
    try:
        yield probe
    finally:
        _probe.reset(token)


class MissProbeEmbedder:
    """Delegates to `inner`; `embed_query` marks the active probe, if any."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self.dim = inner.dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        probe = _probe.get()
        if probe is not None:
            probe.missed = True
        return self._inner.embed_query(text)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)


def elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)

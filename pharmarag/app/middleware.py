"""Request middleware: request ID, rate limit on /v1/*, request metrics and the request log line.

Written as a pure ASGI middleware, not `BaseHTTPMiddleware`: the latter wraps the response body
in a task and has known problems with streaming responses, and `/v1/ask` streams server-sent
events.

The runtime (rate limiter, metrics) is read from `app.state.runtime` at request time. It does not
exist until the lifespan has run, and a test that never runs the lifespan sets it by hand. A
request with no runtime is served without a limit and without metrics, never failed.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

from pharmarag.app.observability import log_event, valid_request_id
from pharmarag.ops import RateDecision

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

RATE_LIMITED_PREFIX = "/v1/"


def client_ip(scope: Scope, *, trust_forwarded_for: bool) -> str:
    """The address the rate limit is keyed on.

    SPEC: section 12 says "per client IP" and is silent on proxies. Behind a reverse proxy the
    socket peer is the proxy, so every user would share one bucket. Behind none, `X-Forwarded-For`
    is a header the client writes, so honouring it lets anyone dodge the limit by rotating it.
    The default is therefore OFF (socket peer). With TRUST_FORWARDED_FOR=true the header is used,
    and the RIGHTMOST entry is taken: that is the one the nearest trusted proxy appended, and the
    entries to its left are whatever the client or earlier hops claimed.
    """
    if trust_forwarded_for:
        forwarded = Headers(scope=scope).get("x-forwarded-for")
        if forwarded:
            last = forwarded.split(",")[-1].strip()
            if last:
                return last
    client = scope.get("client")
    return client[0] if client else "unknown"


class OpsMiddleware:
    def __init__(self, app, *, trust_forwarded_for: bool = False) -> None:
        self.app = app
        self.trust_forwarded_for = trust_forwarded_for

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        t0 = time.perf_counter()
        request_id = valid_request_id(Headers(scope=scope).get("x-request-id")) or uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["log_fields"] = {}
        runtime = getattr(scope["app"].state, "runtime", None)
        metrics = runtime.deps.metrics if runtime is not None else None
        seen = {"status": 500}

        async def send_with_id(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                seen["status"] = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            decision = await self._rate_decision(scope, runtime)
            if decision is not None and not decision.allowed:
                if metrics is not None:
                    metrics.rate_limited.inc()
                response = JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(max(1, decision.retry_after_s))},
                )
                await response(scope, receive, send_with_id)
            else:
                await self.app(scope, receive, send_with_id)
        finally:
            route = scope.get("route")
            # "unrouted": the request never reached a route (a 404, or the 429 answered here).
            path = getattr(route, "path", None) or "unrouted"
            elapsed = time.perf_counter() - t0
            if metrics is not None:
                metrics.http_requests.labels(path, str(seen["status"])).inc()
                metrics.request_duration.labels(path).observe(elapsed)
            log_event(
                "request",
                request_id=request_id,
                method=scope["method"],
                path=path,
                http_status=seen["status"],
                duration_ms=round(elapsed * 1000, 2),
                **state["log_fields"],
            )

    async def _rate_decision(self, scope: Scope, runtime) -> RateDecision | None:
        """None when this request is not limited (other path, or no limiter in this build)."""
        if runtime is None or runtime.rate_limiter is None:
            return None
        if not scope["path"].startswith(RATE_LIMITED_PREFIX):
            return None
        ip = client_ip(scope, trust_forwarded_for=self.trust_forwarded_for)
        try:
            return await runtime.rate_limiter.hit(ip)
        except Exception:
            # Fail open: a broken limiter must not take the API down with it (the Redis cache
            # is already fail-soft; this covers a limiter bug or a non-Redis store).
            logger.warning("rate limiter failed; allowing the request", exc_info=True)
            return None

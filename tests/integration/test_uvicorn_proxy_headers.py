"""The reviewer's rate-limit bypass, against a real uvicorn server on localhost.

Uvicorn trusts `X-Forwarded-For` from 127.0.0.1 by default and rewrites the client address
before our middleware sees it. With `TRUST_FORWARDED_FOR` off, a client on localhost could pick
a new bucket per request. Our deployment runs uvicorn with `--no-proxy-headers`
(`Config(proxy_headers=False)` here); this test shows both what the default does and that the
flag closes it, over a real socket.
"""

import asyncio

import httpx
import pytest
import uvicorn

from pharmarag.app.main import create_app
from pharmarag.app.observability import Metrics
from pharmarag.app.runtime import Runtime
from pharmarag.graph.deps import Deps
from tests.graph.fakes import FakeRateLimiter


class ClosablePool:
    async def close(self) -> None:  # close_runtime closes the pool on shutdown
        pass


class OkGraph:
    async def ainvoke(self, state, config=None):
        return {
            "response": {
                "status": "declined",
                "meta": {"corpus_version": "v", "latency_ms": 0},
            },
            "timings": {},
        }


async def serve(settings, thresholds, limiter, *, proxy_headers: bool):
    deps = Deps(
        llm=None,
        embedder=None,
        pool=ClosablePool(),
        settings=settings,
        thresholds=thresholds,
        metrics=Metrics(),
    )

    async def factory(_settings):
        return Runtime(deps=deps, graph=OkGraph(), rate_limiter=limiter)

    app = create_app(settings, factory)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, proxy_headers=proxy_headers, log_config=None, lifespan="on"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started, "uvicorn did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"http://127.0.0.1:{port}"


async def hit_with_rotating_xff(base: str, n: int) -> list[int]:
    codes = []
    async with httpx.AsyncClient(base_url=base) as client:
        for i in range(n):
            r = await client.post(
                "/v1/ask",
                json={"question": "q"},
                headers={"X-Forwarded-For": f"198.51.100.{i}"},
            )
            codes.append(r.status_code)
    return codes


@pytest.mark.parametrize(
    ("proxy_headers", "expected_buckets"),
    [
        # Uvicorn's default: the header from a 127.0.0.1 peer becomes the client address, so every
        # request lands in its own bucket and the limit is never reached. This is the bug.
        (True, 3),
        # What we deploy (--no-proxy-headers): one bucket, keyed on the socket peer.
        (False, 1),
    ],
)
async def test_uvicorn_proxy_headers_decide_whether_xff_can_pick_the_bucket(
    settings, thresholds, proxy_headers, expected_buckets
):
    limiter = FakeRateLimiter(limit=1)
    server, task, base = await serve(settings, thresholds, limiter, proxy_headers=proxy_headers)
    try:
        codes = await hit_with_rotating_xff(base, 3)
    finally:
        server.should_exit = True
        await task
    assert len(limiter.counts) == expected_buckets
    assert codes == ([200, 200, 200] if proxy_headers else [200, 429, 429])
    if not proxy_headers:
        assert list(limiter.counts) == ["127.0.0.1"]

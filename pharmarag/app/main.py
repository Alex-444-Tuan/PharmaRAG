"""FastAPI app (spec section 10): /v1/ask, /v1/drugs/resolve, /v1/labels/{setid}, /healthz,
/readyz and /metrics.

`POST /v1/ask` with `stream: true` answers with server-sent events (spec section 10). The JSON
path goes through the answer cache (spec section 5); the SSE path never does."""

import asyncio
import copy
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from psycopg.rows import dict_row
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from pharmarag.app.middleware import OpsMiddleware
from pharmarag.app.observability import configure_logging, question_fields
from pharmarag.app.runtime import Runtime, build_runtime, close_runtime
from pharmarag.app.settings import Settings, get_settings
from pharmarag.graph.nodes.finalize import corpus_version
from pharmarag.graph.run import effective_flags, run_ask, run_ask_stream
from pharmarag.graph.state import Flags
from pharmarag.grounding.stream import resolve_verify_mode
from pharmarag.llm.errors import LLMError
from pharmarag.ops import answer_key
from pharmarag.sections import SECTIONS

logger = logging.getLogger(__name__)

DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"

# The statement the RxNav terms of service require (spec section 12, "Attribution"). Also in the
# README.
NLM_ATTRIBUTION = (
    "This product uses publicly available data from the U.S. National Library of Medicine "
    "(NLM), National Institutes of Health, Department of Health and Human Services; NLM is not "
    "responsible for the product and does not endorse or recommend this or any other product."
)

_ON = {"on", "true", "1"}
_OFF = {"off", "false", "0"}


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    pinned_rxcuis: list[str] = Field(default_factory=list, max_length=10)
    stream: bool = False
    debug: bool = False


def parse_eval_flags(header: str) -> Flags:
    """Parse `resolver=off,verifier=off`. Malformed input is a client error (422)."""
    values: dict[str, bool] = {}
    for part in header.split(","):
        if not part.strip():
            continue
        key, _, raw = part.partition("=")
        key, raw = key.strip().lower(), raw.strip().lower()
        if key not in Flags.model_fields or raw not in _ON | _OFF:
            raise HTTPException(422, f"invalid X-Eval-Flags entry: {part.strip()!r}")
        values[key] = raw in _ON
    return Flags(**values)


_DEPENDENCY_ERRORS = (LLMError, psycopg.OperationalError, PoolTimeout)


def sse_frame(event: dict) -> bytes:
    """One server-sent event: `event: <name>` and one `data:` line of JSON."""
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], default=str)}\n\n".encode()


async def sse_stream(
    events: AsyncIterator[dict],
    first: dict,
    on_done: Callable[[dict], None] | None = None,
) -> AsyncIterator[bytes]:
    """Frames for `first` and the rest of `events`.

    The status line is already sent by now, so a later fault cannot become a 503. It becomes an
    `error` event, the last one, and no `done` follows. A client treats a stream without `done`
    as failed and shows nothing as final.

    `on_done` receives the response carried by the `done` event, for the request log and metrics.
    """

    def seen(event: dict) -> bytes:
        if on_done is not None and event["event"] == "done":
            on_done(event["data"])
        return sse_frame(event)

    try:
        yield seen(first)
        async for event in events:
            yield seen(event)
    except _DEPENDENCY_ERRORS:
        logger.warning("dependency failed mid-stream", exc_info=True)
        yield sse_frame(
            {"event": "error", "data": {"status": 503, "detail": "a dependency is unavailable"}}
        )
    except Exception:
        # A bug (a crashed verifier, say) after sentences went out. Same shape as a 503: the
        # client sees an `error` event and no `done`, never a silently cut stream.
        logger.exception("request failed mid-stream")
        yield sse_frame({"event": "error", "data": {"status": 500, "detail": "internal error"}})
    finally:
        await events.aclose()  # type: ignore[attr-defined]


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(503, "the service is starting")
    return runtime


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _ON


def flags_key(flags: Flags, verify_mode: str) -> str:
    """The canonical flags part of an answer-cache key (spec section 5): what the request really
    runs with, so an eval-mode request with the verifier off never reads a verified answer."""
    resolver, verifier = ("on" if b else "off" for b in (flags.resolver, flags.verifier))
    return f"resolver={resolver},verifier={verifier},verify={verify_mode}"


def ask_log_fields(response: dict, *, cache_hit: bool) -> dict:
    """What the request log line says about an answer (NFR-5). Never the answer text."""
    meta = response.get("meta", {})
    return {
        "status": response.get("status"),
        "retrieval_attempts": meta.get("retrieval_attempts"),
        "generation_attempts": meta.get("generation_attempts"),
        "removed_sentences": meta.get("removed_sentences"),
        "cache_hit": cache_hit,
    }


def record_ask(request: Request, runtime: Runtime, response: dict, *, cache_hit: bool) -> None:
    request.state.log_fields.update(ask_log_fields(response, cache_hit=cache_hit))
    metrics = runtime.deps.metrics
    if metrics is not None:
        metrics.ask.labels(str(response.get("status"))).inc()


async def _check_database(deps) -> str:
    try:
        async with asyncio.timeout(deps.settings.db_timeout_s), deps.pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        logger.warning("readyz: database check failed: %s", type(exc).__name__)
        return "unreachable"
    return "ok"


async def _check_redis(deps) -> str:
    ping = getattr(deps.cache, "ping", None)
    if ping is None:
        return "ok"  # a cache that cannot be pinged (an in-process fake) cannot be down
    return "ok" if await ping() else "unreachable"


def _models_loaded(deps) -> bool:
    """The embedder is always loaded before the app serves. The NLI verifier must be too,
    except in EVAL_MODE, where a build without it is the AT-2 baseline on purpose."""
    return deps.embedder is not None and (deps.verifier is not None or deps.settings.eval_mode)


async def _ask_json(
    body: AskRequest, flags: Flags, request_id: str, runtime: Runtime, t0: float
) -> dict:
    """The non-streaming answer, through the answer cache when there is one (spec section 5).

    The key holds `corpus_version`, the flags the request really runs with, the question and the
    pinned RxCUIs. Only `answered` responses are stored: a refusal, a clarification or an
    `insufficient_evidence` may come from a transient fault (RxNav down, a weak retrieval that a
    reworded index would fix) and must be recomputed next time. `debug` requests bypass the cache
    because their payload is per request. The cache is consulted here and not in `run_ask`, so
    the eval harness, which calls `run_ask`, never sees it.
    """
    deps = runtime.deps
    use_cache = runtime.answer_cache is not None and deps.settings.answer_cache and not body.debug
    key = version = None
    if use_cache:
        version = await corpus_version(deps)
        key = answer_key(
            version,
            flags_key(effective_flags(flags, deps), resolve_verify_mode(deps.settings)),
            body.question,
            body.pinned_rxcuis,
        )
        cached = await runtime.answer_cache.get(key)
        if cached is not None:
            hit = copy.deepcopy(cached)
            hit["meta"]["cache_hit"] = True
            hit["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            if deps.metrics is not None:
                deps.metrics.cache_result("answer", hit=True)
            return hit
    result = await run_ask(
        runtime.graph,
        deps,
        question=body.question,
        pinned_rxcuis=body.pinned_rxcuis,
        flags=flags,
        request_id=request_id,
        debug=body.debug,
    )
    if use_cache:
        if deps.metrics is not None:
            deps.metrics.cache_result("answer", hit=False)
        # The version guard: a corpus refresh between the key lookup and the graph's own read
        # would file a newer answer under the older key.
        if result["status"] == "answered" and result["meta"]["corpus_version"] == version:
            await runtime.answer_cache.put(key, copy.deepcopy(result))
    result["meta"]["cache_hit"] = False
    return result


def create_app(
    settings: Settings | None = None,
    runtime_factory: Callable[[Settings], Awaitable[Runtime]] = build_runtime,
    *,
    trust_forwarded_for: bool | None = None,
) -> FastAPI:
    """`trust_forwarded_for` defaults to the TRUST_FORWARDED_FOR environment variable (off)."""
    settings = settings or get_settings()
    if trust_forwarded_for is None:
        # SPEC: read straight from the environment because Settings is a contract file. See the
        # report: the contract change is one `trust_forwarded_for: bool = False` field.
        trust_forwarded_for = _env_flag("TRUST_FORWARDED_FOR")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.runtime = await runtime_factory(settings)
        try:
            yield
        finally:
            await close_runtime(app.state.runtime)

    app = FastAPI(
        title="PharmaRAG",
        description=(
            "Answers prescription-drug questions from FDA DailyMed labels, citing a label span "
            "for every sentence. Not medical advice.\n\n" + NLM_ATTRIBUTION
        ),
        lifespan=lifespan,
    )
    app.add_middleware(OpsMiddleware, trust_forwarded_for=trust_forwarded_for)

    @app.get("/healthz")
    async def healthz() -> dict:
        """The process is up. Checks nothing else, so an orchestrator can tell "restart me" from
        "do not send me traffic yet" (`/readyz`)."""
        return {"status": "ok"}

    @app.get("/readyz", response_model=None)
    async def readyz(runtime: Annotated[Runtime, Depends(get_runtime)]) -> JSONResponse:
        """Database, Redis (when it is the cache backend) and both models are ready.

        503 with the failing dependencies named in `failing`.
        """
        deps = runtime.deps
        checks = {
            "database": await _check_database(deps),
            "models": "ok" if _models_loaded(deps) else "not loaded",
        }
        if deps.settings.cache_backend == "redis":
            checks["redis"] = await _check_redis(deps)
        failing = [name for name, state in checks.items() if state != "ok"]
        body = {"status": "unavailable" if failing else "ready", "checks": checks}
        if failing:
            body["failing"] = failing
        return JSONResponse(body, status_code=503 if failing else 200)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint(runtime: Annotated[Runtime, Depends(get_runtime)]) -> Response:
        metrics = runtime.deps.metrics
        if metrics is None:
            raise HTTPException(503, "metrics are not enabled")
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/ask", response_model=None)
    async def ask(
        body: AskRequest,
        request: Request,
        response: Response,
        runtime: Annotated[Runtime, Depends(get_runtime)],
        x_eval_flags: Annotated[str | None, Header()] = None,
    ) -> dict | StreamingResponse:
        t0 = time.perf_counter()
        deps = runtime.deps
        # The header is honoured only in eval mode (FR-9, spec section 10).
        flags = Flags()
        if x_eval_flags is not None and deps.settings.eval_mode:
            flags = parse_eval_flags(x_eval_flags)
        request_id = request.state.request_id  # set (and echoed as X-Request-ID) by the middleware
        request.state.log_fields.update(
            question_fields(body.question, log_questions=deps.settings.log_questions)
        )
        if body.stream:
            events = run_ask_stream(
                runtime.graph,
                deps,
                question=body.question,
                pinned_rxcuis=body.pinned_rxcuis,
                flags=flags,
                request_id=request_id,
                debug=body.debug,
            )
            # Wait for the first event before committing to a 200: a fault before it (the
            # analyze call) can still be a proper 503. Declined and other early exits send
            # `done` first, which is fine.
            try:
                first = await anext(events)
            except _DEPENDENCY_ERRORS as exc:
                await events.aclose()
                raise HTTPException(503, "a dependency is unavailable") from exc

            # SPEC: SSE bypasses the answer cache (brief for cycle 4). A cached answer would have
            # to be replayed as sentence events, which would skip the live verifier path that
            # the stream design (spec section 9) exists to show.
            def on_done(done: dict) -> None:
                done["meta"]["cache_hit"] = False  # same meta shape as the JSON path
                record_ask(request, runtime, done, cache_hit=False)

            return StreamingResponse(
                sse_stream(events, first, on_done=on_done),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await _ask_json(body, flags, request_id, runtime, t0)
        except _DEPENDENCY_ERRORS as exc:
            raise HTTPException(503, "a dependency is unavailable") from exc
        record_ask(request, runtime, result, cache_hit=bool(result["meta"].get("cache_hit")))
        return result

    @app.get("/v1/drugs/resolve")
    async def resolve_drug(
        name: Annotated[str, Query(min_length=1, max_length=200)],
        runtime: Annotated[Runtime, Depends(get_runtime)],
    ) -> dict:
        """Run the resolver alone, for demos and debugging."""
        resolver = runtime.deps.resolver
        if resolver is None:
            raise HTTPException(503, "the drug resolver is not available in this build")
        mention = name.strip()
        if not mention:
            raise HTTPException(422, "name must not be blank")
        try:
            (resolution,) = await resolver.resolve([mention])
        except (httpx.HTTPError, psycopg.OperationalError, PoolTimeout) as exc:
            raise HTTPException(503, "a dependency is unavailable") from exc
        return resolution.model_dump()

    @app.get("/v1/labels/{setid}")
    async def label_metadata(
        setid: uuid.UUID, runtime: Annotated[Runtime, Depends(get_runtime)]
    ) -> dict:
        """Metadata and section list of the newest stored version of one label."""
        try:
            async with (
                runtime.deps.pool.connection() as conn,
                conn.cursor(row_factory=dict_row) as cur,
            ):
                await cur.execute(
                    "SELECT setid, spl_version, title, labeler, marketing_category, "
                    "effective_date, ingredient_key, ingredient_rxcuis, product_rxcuis "
                    "FROM label WHERE setid = %s ORDER BY spl_version DESC LIMIT 1",
                    (setid,),
                )
                label = await cur.fetchone()
                if label is None:
                    raise HTTPException(404, "label not found")
                await cur.execute(
                    "SELECT section_loinc, min(section_title) AS title, count(*) AS chunks "
                    "FROM chunk WHERE setid = %s AND spl_version = %s "
                    "GROUP BY section_loinc ORDER BY min(ordinal)",
                    (setid, label["spl_version"]),
                )
                sections = await cur.fetchall()
        except (psycopg.OperationalError, PoolTimeout) as exc:
            raise HTTPException(503, "a dependency is unavailable") from exc
        return {
            "setid": str(label["setid"]),
            "spl_version": label["spl_version"],
            "title": label["title"],
            "labeler": label["labeler"],
            "marketing_category": label["marketing_category"],
            "effective_date": label["effective_date"].isoformat()
            if label["effective_date"]
            else None,
            "ingredient_key": label["ingredient_key"],
            "ingredient_rxcuis": label["ingredient_rxcuis"],
            "product_rxcuis": label["product_rxcuis"],
            "url": DAILYMED_URL.format(setid=label["setid"]),
            "sections": [
                {
                    "loinc": r["section_loinc"],
                    "title": SECTIONS.get(r["section_loinc"], r["title"]),
                    "chunks": r["chunks"],
                }
                for r in sections
            ],
        }

    return app


app = create_app()

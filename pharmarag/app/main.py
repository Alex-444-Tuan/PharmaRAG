"""FastAPI app (spec section 10). Cycle 1: /v1/ask (non-streaming) and /healthz."""

import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from pharmarag.app.runtime import Runtime, build_runtime, close_runtime
from pharmarag.app.settings import Settings, get_settings
from pharmarag.graph.run import run_ask
from pharmarag.graph.state import Flags
from pharmarag.llm.errors import LLMError

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


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def create_app(
    settings: Settings | None = None,
    runtime_factory: Callable[[Settings], Awaitable[Runtime]] = build_runtime,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = await runtime_factory(settings)
        try:
            yield
        finally:
            await close_runtime(app.state.runtime)

    app = FastAPI(
        title="PharmaRAG",
        description=(
            "Answers prescription-drug questions from FDA DailyMed labels, citing a label span "
            "for every sentence. Not medical advice."
        ),
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask")
    async def ask(
        body: AskRequest,
        response: Response,
        runtime: Annotated[Runtime, Depends(get_runtime)],
        x_eval_flags: Annotated[str | None, Header()] = None,
    ) -> dict:
        if body.stream:
            raise HTTPException(501, "streaming is not implemented yet")
        # The header is honoured only in eval mode (FR-9, spec section 10).
        flags = Flags()
        if x_eval_flags is not None and runtime.deps.settings.eval_mode:
            flags = parse_eval_flags(x_eval_flags)
        request_id = uuid.uuid4().hex
        response.headers["X-Request-ID"] = request_id
        try:
            return await run_ask(
                runtime.graph,
                runtime.deps,
                question=body.question,
                pinned_rxcuis=body.pinned_rxcuis,
                flags=flags,
                request_id=request_id,
                debug=body.debug,
            )
        except (LLMError, psycopg.OperationalError, PoolTimeout) as exc:
            raise HTTPException(503, "a dependency is unavailable") from exc

    return app


app = create_app()

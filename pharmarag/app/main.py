"""FastAPI app (spec section 10): /v1/ask (non-streaming), /v1/drugs/resolve, /v1/labels/{setid},
/healthz."""

import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from psycopg.rows import dict_row
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from pharmarag.app.runtime import Runtime, build_runtime, close_runtime
from pharmarag.app.settings import Settings, get_settings
from pharmarag.graph.run import run_ask
from pharmarag.graph.state import Flags
from pharmarag.llm.errors import LLMError
from pharmarag.sections import SECTIONS

DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"

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

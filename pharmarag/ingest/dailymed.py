"""DailyMed v2 REST client and the `rxnorm_mappings.zip` product index (spec section 4)."""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx

from pharmarag.ingest.http import RateLimiter, get_with_retry
from pharmarag.sections import PRESCRIPTION_LABEL_LOINC

log = logging.getLogger(__name__)

# SPEC: the spec names DailyMed's mapping file but not its URL, and settings has no
# dailymed timeout. The file lives on a separate host from the REST API.
RXNORM_MAPPINGS_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/rxnorm_mappings.zip"
DEFAULT_TIMEOUT_S = 30.0
# SPEC: DailyMed publishes no rate limit. 10 requests per second is a polite default.
DEFAULT_MAX_RPS = 10.0

# FDA marketing category codes accepted by the `marketing_category_code` filter.
MARKETING_CATEGORY_NDA = "C73594"
MARKETING_CATEGORY_BLA = "C73585"


@dataclass(frozen=True)
class LabelRef:
    """One row of `/spls.json`. `published_date` is when DailyMed published this version."""

    setid: str
    spl_version: int
    title: str
    published_date: date | None


def _parse_published(value: str | None) -> date | None:
    try:
        return datetime.strptime(value or "", "%b %d, %Y").date()
    except ValueError:
        return None


class DailyMedClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        retries: int = 2,
        backoff_s: float = 1.0,
        max_rps: float = DEFAULT_MAX_RPS,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=True)
        self._owns_client = client is None
        self._retries = retries
        self._backoff_s = backoff_s
        self._limiter = RateLimiter(max_rps)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, url: str, **kwargs) -> httpx.Response:
        return await get_with_retry(
            self._client,
            url,
            retries=self._retries,
            backoff_s=self._backoff_s,
            before_request=self._limiter.wait,
            **kwargs,
        )

    async def list_labels(
        self,
        rxcui: str,
        *,
        doctype: str = PRESCRIPTION_LABEL_LOINC,
        marketing_category_code: str | None = None,
        pagesize: int = 100,
        max_pages: int = 3,
    ) -> list[LabelRef]:
        """Labels whose products map to `rxcui`, in DailyMed's order (newest first)."""
        params: dict[str, str | int] = {"rxcui": rxcui, "doctype": doctype, "pagesize": pagesize}
        if marketing_category_code:
            params["marketing_category_code"] = marketing_category_code
        out: list[LabelRef] = []
        for page in range(1, max_pages + 1):
            resp = await self._get(f"{self._base}/spls.json", params={**params, "page": page})
            body = resp.json()
            for row in body.get("data") or []:
                out.append(
                    LabelRef(
                        setid=row["setid"].lower(),
                        spl_version=int(row["spl_version"]),
                        title=row.get("title", ""),
                        published_date=_parse_published(row.get("published_date")),
                    )
                )
            nxt = (body.get("metadata") or {}).get("next_page")
            if not isinstance(nxt, int):
                break
        return out

    async def fetch_spl_xml(self, setid: str) -> bytes:
        resp = await self._get(f"{self._base}/spls/{setid}.xml")
        return resp.content

    async def download_rxnorm_mappings(self, dest: Path, url: str = RXNORM_MAPPINGS_URL) -> Path:
        """Stream `rxnorm_mappings.zip` to `dest` (written atomically)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        async with self._client.stream("GET", url, timeout=300.0) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                async for part in resp.aiter_bytes():
                    f.write(part)
        tmp.replace(dest)
        return dest


def load_rxnorm_mappings(path: Path) -> dict[str, set[str]]:
    """Read `rxnorm_mappings.zip` into {setid: product RxCUIs}.

    The file is pipe-delimited: SETID|SPL_VERSION|RXCUI|RXSTRING|RXTTY. If a setid appears
    with several versions only the highest version's RxCUIs are kept.
    """
    best: dict[str, tuple[int, set[str]]] = {}
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(name) as raw:
            reader = csv.reader(
                io.TextIOWrapper(raw, encoding="utf-8"), delimiter="|", quoting=csv.QUOTE_NONE
            )
            next(reader, None)  # header
            for row in reader:
                if len(row) < 3:
                    continue
                setid, rxcui = row[0].lower(), row[2]
                if not row[1].isdigit():
                    continue
                version = int(row[1])
                cur = best.get(setid)
                if cur is None or version > cur[0]:
                    best[setid] = (version, {rxcui})
                elif version == cur[0]:
                    cur[1].add(rxcui)
    return {setid: rxcuis for setid, (_, rxcuis) in best.items()}

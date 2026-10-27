"""Fake Resolver, RxFactsSource and Verifier for graph tests. No network, no models."""

import json
from collections.abc import Callable, Sequence

from pharmarag.graph.state import Candidate, Chunk, Fact, Resolution, Sentence, Verdict
from pharmarag.resolver.base import RxFact


def resolved(
    mention: str,
    key: str,
    name: str,
    match: str = "exact",
    product_hint: str | None = None,
) -> Resolution:
    return Resolution(
        mention=mention,
        status="resolved",
        match=match,  # type: ignore[arg-type]
        ingredient_key=key,
        name=name,
        ingredient_rxcuis=key.split("+"),
        product_hint=product_hint,
        score=1.0,
    )


def ambiguous(mention: str, candidates: Sequence[tuple[str, str, float]]) -> Resolution:
    return Resolution(
        mention=mention,
        status="ambiguous",
        candidates=[Candidate(ingredient_key=k, name=n, score=s) for k, n, s in candidates],
    )


def unknown(mention: str) -> Resolution:
    return Resolution(mention=mention, status="unknown", score=0.3)


def not_covered(mention: str, key: str = "999", name: str = "rarexamab") -> Resolution:
    return Resolution(
        mention=mention,
        status="not_covered",
        match="exact",
        ingredient_key=key,
        name=name,
        ingredient_rxcuis=[key],
    )


class FakeResolver:
    """Answers from a table keyed by lower-cased mention. Records every call.

    Like the real one, an ambiguous mention whose candidates include a pinned ingredient
    resolves to it with match="pinned".
    """

    def __init__(self, table: dict[str, Resolution]) -> None:
        self.table = {k.lower(): v for k, v in table.items()}
        self.calls: list[dict] = []

    async def resolve(
        self, mentions: list[str], *, pinned_rxcuis: list[str] | None = None
    ) -> list[Resolution]:
        self.calls.append({"mentions": list(mentions), "pinned": list(pinned_rxcuis or [])})
        out: list[Resolution] = []
        for m in mentions:
            r = self.table.get(m.lower(), unknown(m))
            if r.status == "ambiguous":
                pick = next(
                    (c for c in r.candidates if c.ingredient_key in (pinned_rxcuis or [])), None
                )
                if pick:
                    r = resolved(m, pick.ingredient_key, pick.name, match="pinned")
            out.append(r)
        return out


class FakeRxFacts:
    def __init__(self, lines: dict[str, list[str]] | None = None, error: Exception | None = None):
        self.lines = lines or {}
        self.error = error
        self.calls: list[list[str]] = []

    async def facts(self, ingredient_rxcuis: list[str]) -> list[RxFact]:
        self.calls.append(list(ingredient_rxcuis))
        if self.error:
            raise self.error
        return [
            RxFact(rxcui=rxcui, text=line)
            for rxcui in ingredient_rxcuis
            for line in self.lines.get(rxcui, [])
        ]


class FakeVerifier:
    """Scripted verdicts, one entry per generation attempt.

    An entry is a list of verdicts (one per sentence, in order) or a callable
    (sentences) -> list of verdicts. Sentences that already carry a verdict other than
    "unverified" (the writer's no_citation ones) come back unchanged, as the contract says.
    The last entry repeats when the script runs out.

    It speaks both verification paths: `verify` (sequential mode) is one attempt; `prepare`
    followed by `verify_sentence` calls (stream mode) is one attempt too. `calls` has one record
    per attempt that verified something, so tests can count attempts in either mode. `prepares`
    counts `prepare` calls, which stream mode makes before it knows whether the writer will
    produce any sentence at all.
    """

    def __init__(
        self, script: Sequence[Sequence[Verdict] | Callable[[list[Sentence]], list[Verdict]]]
    ):
        self.script = list(script)
        self.calls: list[dict] = []
        self.prepares: list[dict] = []
        self.batches: list[list[str]] = []  # texts of each verify_many call
        self._attempts = 0
        self._cursor: dict[int, int] = {}

    def _entry(self, attempt: int):
        return self.script[min(attempt, len(self.script) - 1)]

    def _verdict_for(self, s: Sentence, v: Verdict) -> Sentence:
        if s.verdict != "unverified":
            return s
        return s.model_copy(
            update={
                "verdict": v,
                "score": 0.93 if v == "supported" else 0.1,
                "span": f"{s.citations[0]}:0" if v == "supported" and s.citations else None,
            }
        )

    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        attempt = self._attempts
        self._attempts += 1
        self.calls.append({"sentences": list(sentences), "evidence": evidence, "facts": facts})
        entry = self._entry(attempt)
        verdicts = entry(sentences) if callable(entry) else list(entry)
        assert len(verdicts) == len(sentences), "FakeVerifier script does not match sentence count"
        return [self._verdict_for(s, v) for s, v in zip(sentences, verdicts, strict=True)]

    async def prepare(self, evidence: list[Chunk], facts: list[Fact]):
        attempt = self._attempts
        self._attempts += 1
        self.prepares.append({"evidence": evidence, "facts": facts})
        return {"attempt": attempt, "evidence": evidence, "facts": facts}

    async def verify_sentence(self, sentence: Sentence, prepared) -> Sentence:
        attempt = prepared["attempt"]
        if attempt not in self._cursor:
            self._cursor[attempt] = 0
            self.calls.append({"sentences": [], "evidence": prepared["evidence"]})
        entry = self._entry(attempt)
        i = self._cursor[attempt]
        self._cursor[attempt] += 1
        self.calls[-1]["sentences"].append(sentence)
        if sentence.verdict != "unverified":
            return sentence
        # The script lists a verdict for every sentence of the attempt, including the ones the
        # writer already marked (those are not passed here), so index by the attempt's position.
        verdicts = entry([sentence]) if callable(entry) else list(entry)
        return self._verdict_for(sentence, verdicts[min(i, len(verdicts) - 1)])

    async def verify_many(self, sentences: list[Sentence], prepared) -> list[Sentence]:
        """One batch: same results, same order as `verify_sentence` on each. The stream path
        passes only the sentences the writer left unverified, so in stream mode a list script is
        indexed over those (a writer-marked sentence in the middle does not take an index)."""
        self.batches.append([s.text for s in sentences])
        return [await self.verify_sentence(s, prepared) for s in sentences]


# --- scripted LLM replies shared by the route tests -------------------------------------------

DOSE_TEXT = "Hydroxyzine dosage for anxiety in adults is 50 to 100 mg four times daily."
HYDRALAZINE_DOSE_TEXT = "Hydralazine dosage for hypertension in adults is 10 mg four times daily."
GIBBERISH = "zzzz qqqq"  # shares no word with any fixture chunk: no lexical hit, cosine 0
HYDROXYZIN_Q = "What is the max daily dose of hydroxyzin?"


def analysis(**kw) -> str:
    """A scripted `analyze` reply. The default search query is a fixture chunk verbatim."""
    base = {"intent": "label_qa", "mentions": [], "sections": [], "search_query": DOSE_TEXT}
    return json.dumps({**base, **kw})


def rewrite(query: str = DOSE_TEXT) -> str:
    return json.dumps({"search_query": query})


# --- cycle 4 operational fakes (the real ones are Lane A's, pharmarag/caching/) ----------------


class FakeRateLimiter:
    """`limit` allowed hits per client id, then denials with a 17 s Retry-After."""

    def __init__(self, limit: int = 30, error: Exception | None = None):
        self.limit = limit
        self.error = error
        self.counts: dict[str, int] = {}

    async def hit(self, client_id: str):
        from pharmarag.ops import RateDecision

        if self.error:
            raise self.error
        n = self.counts[client_id] = self.counts.get(client_id, 0) + 1
        if n > self.limit:
            return RateDecision(allowed=False, remaining=0, retry_after_s=17)
        return RateDecision(allowed=True, remaining=self.limit - n, retry_after_s=0)


class FakeAnswerCache:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.gets: list[str] = []
        self.puts: list[str] = []

    async def get(self, key: str) -> dict | None:
        self.gets.append(key)
        return self.data.get(key)

    async def put(self, key: str, response: dict) -> None:
        self.puts.append(key)
        self.data[key] = response


class FakeQueryEmbeddingCache:
    """Embeds through `embedder` on a miss, like the real cache; remembers by query text."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.vectors: dict[str, object] = {}

    async def embed_query(self, text: str):
        if text not in self.vectors:
            self.vectors[text] = self.embedder.embed_query(text)
        return self.vectors[text]

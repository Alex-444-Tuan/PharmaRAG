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
    """Scripted verdicts, one entry per `verify` call.

    An entry is a list of verdicts (one per sentence, in order) or a callable
    (sentences) -> list of verdicts. Sentences that already carry a verdict other than
    "unverified" (the writer's no_citation ones) come back unchanged, as the contract says.
    The last entry repeats when the script runs out.
    """

    def __init__(
        self, script: Sequence[Sequence[Verdict] | Callable[[list[Sentence]], list[Verdict]]]
    ):
        self.script = list(script)
        self.calls: list[dict] = []

    async def verify(
        self, sentences: list[Sentence], evidence: list[Chunk], facts: list[Fact]
    ) -> list[Sentence]:
        self.calls.append({"sentences": list(sentences), "evidence": evidence, "facts": facts})
        entry = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        verdicts = entry(sentences) if callable(entry) else list(entry)
        assert len(verdicts) == len(sentences), "FakeVerifier script does not match sentence count"
        out: list[Sentence] = []
        for s, v in zip(sentences, verdicts, strict=True):
            if s.verdict != "unverified":
                out.append(s)
                continue
            out.append(
                s.model_copy(
                    update={
                        "verdict": v,
                        "score": 0.93 if v == "supported" else 0.1,
                        "span": f"{s.citations[0]}:0" if v == "supported" and s.citations else None,
                    }
                )
            )
        return out


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

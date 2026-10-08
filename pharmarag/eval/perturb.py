"""Rule-based rewrites and perturbations of label sentences (spec section 11.2).

Used to build the synthetic dev set (supported rewrites and hallucinated perturbations) and the
top-up claims for the test set. Every function takes a sentence and the text of the evidence it
must NOT be supported by, and returns None when the rule does not apply. A perturbation is only
returned when the changed fact is absent from the evidence, so the label H is true by construction.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from decimal import Decimal

from pharmarag.grounding.numeric import _QUANTITY, missing_quantities, parse_number

# --- supported rewrites (label S) ---

_INLINE_REF = re.compile(r"\s*\((?:see\s+)?\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*\)", re.IGNORECASE)

_PARAPHRASES: list[tuple[str, str]] = [
    (r"\btwice daily\b", "two times a day"),
    (r"\bonce daily\b", "once a day"),
    (r"\bthree times daily\b", "three times a day"),
    (r"\bshould be administered\b", "should be given"),
    (r"\bis contraindicated in\b", "must not be used in"),
    (r"\bare contraindicated in\b", "must not be used in"),
    (r"\bis not recommended\b", "is not advised"),
    (r"\bDiscontinue\b", "Stop"),
    (r"\bdiscontinue\b", "stop"),
    (r"\bmay be increased\b", "can be increased"),
    (r"\bin patients with\b", "for patients with"),
    (r"\bpatients\b", "people"),
    (r"\bAdminister\b", "Give"),
    (r"\bshould be monitored\b", "should be watched closely"),
]


def clean_sentence(text: str) -> str:
    """The sentence as a careful writer would phrase it: no inline section references."""
    return " ".join(_INLINE_REF.sub("", text).split())


def lower_first(text: str) -> str:
    """Lower-case the first letter unless the first word is an acronym or contains a capital."""
    first = text.split(" ", 1)[0]
    if len(first) > 1 and any(c.isupper() for c in first[1:]):
        return text
    return text[:1].lower() + text[1:]


def rewrite_verbatim(sentence: str) -> str:
    return clean_sentence(sentence)


def rewrite_light(sentence: str, rng: random.Random) -> str | None:
    """A light, meaning-preserving rewrite: first applicable of several rules in random order."""
    s = clean_sentence(sentence)
    rules: list[Callable[[str], str | None]] = [
        _paraphrase,
        _first_clause,
        _spacing,
        _drop_discourse,
        _drop_parenthetical,
    ]
    rng.shuffle(rules)
    for rule in rules:
        out = rule(s)
        if out and out != s:
            return out
    return None


def _paraphrase(s: str) -> str | None:
    for pat, repl in _PARAPHRASES:
        if re.search(pat, s):
            return re.sub(pat, repl, s, count=1)
    return None


_DISCOURSE = re.compile(r"^(?:However|Therefore|In addition|Additionally|Also|Thus),?\s+")


def _drop_discourse(s: str) -> str | None:
    """Drop a leading connective: a writer does not carry "However," into a stand-alone claim."""
    out = _DISCOURSE.sub("", s)
    return out[:1].upper() + out[1:] if out != s else None


def _drop_parenthetical(s: str) -> str | None:
    """Drop a parenthetical aside. The shorter claim says less, so it is still supported."""
    out = " ".join(re.sub(r"\s*\([^()]*\)", "", s).split())
    return out if out != s and len(out.split()) >= 5 else None


def _first_clause(s: str) -> str | None:
    if ";" in s:
        head = s.split(";", 1)[0].strip()
        if len(head.split()) >= 5:
            return head + "."
    return None


def _spacing(s: str) -> str | None:
    out = re.sub(r"(\d) (mg|mcg|mL|g)\b", r"\1\2", s)
    return out if out != s else None


def join_sentences(first: str, second: str) -> str:
    """Two label sentences as one claim: both facts are stated, so the claim is supported."""
    return f"{clean_sentence(first).rstrip('.')}; {lower_first(clean_sentence(second))}"


# --- hallucination perturbations (label H) ---

_NEGATIONS: list[tuple[str, str]] = [
    (r"\bis contraindicated\b", "is not contraindicated"),
    (r"\bare contraindicated\b", "are not contraindicated"),
    (r"\bshould not be\b", "should be"),
    (r"\bshould be\b", "should not be"),
    (r"\bis not recommended\b", "is recommended"),
    (r"\bis recommended\b", "is not recommended"),
    (r"\bcan cause\b", "does not cause"),
    (r"\bmay cause\b", "does not cause"),
    (r"\bmay increase\b", "may decrease"),
    (r"\bincreases\b", "decreases"),
    (r"\bdecreases\b", "increases"),
    (r"\bmay decrease\b", "may increase"),
    (r"\bis excreted\b", "is not excreted"),
    (r"\bis metabolized\b", "is not metabolized"),
    (r"\bcan be\b", "cannot be"),
    (r"\bmust not\b", "must"),
    (r"\bAvoid\b", "Use"),
    (r"\bavoid\b", "use"),
    (r"\bDo not\b", "Do"),
    (r"\bdo not\b", "do"),
    (r"\bhave been reported\b", "have not been reported"),
    (r"\bwas observed\b", "was not observed"),
    (r"\bwere observed\b", "were not observed"),
]

_FREQUENCY: list[tuple[str, str]] = [
    (r"\bonce daily\b", "three times daily"),
    (r"\bonce a day\b", "three times a day"),
    (r"\btwice daily\b", "once daily"),
    (r"\bthree times daily\b", "once daily"),
    (r"\bdaily\b", "weekly"),
    (r"\bweekly\b", "daily"),
    (r"\bin the morning\b", "at bedtime"),
    (r"\bat bedtime\b", "in the morning"),
    (r"\bwith food\b", "on an empty stomach"),
    (r"\bon an empty stomach\b", "with food"),
    (r"\bas needed\b", "on a fixed schedule"),
]

_POPULATION: list[tuple[str, str]] = [
    (r"\bpediatric patients\b", "adult patients"),
    (r"\badult patients\b", "pediatric patients"),
    (r"\badults\b", "children"),
    (r"\bchildren\b", "adults"),
    (r"\belderly patients\b", "pediatric patients"),
    (r"\bgeriatric patients\b", "pediatric patients"),
    (r"\bpregnant women\b", "nursing mothers"),
    (r"\bnursing mothers\b", "pregnant women"),
    (r"\bwomen\b", "men"),
    (r"\bmen\b", "women"),
    (r"\brenal impairment\b", "hepatic impairment"),
    (r"\bhepatic impairment\b", "renal impairment"),
    (r"\bpregnancy\b", "lactation"),
]


# A claim stronger than the evidence: subtle for NLI, because most of the words are the same.
_OVERCLAIM: list[tuple[str, str]] = [
    (r"\bmay cause\b", "always causes"),
    (r"\bcan cause\b", "always causes"),
    (r"\bmay be\b", "is always"),
    (r"\bmay\b", "will"),
    (r"\bsome patients\b", "all patients"),
    (r"\bis recommended\b", "is required"),
    (r"\bshould be considered\b", "is required"),
    (r"\boccasionally\b", "routinely"),
    (r"\brare\b", "common"),
    (r"\bcommon\b", "rare"),
    (r"\bcan reduce\b", "eliminates"),
    (r"\bincreased risk of\b", "certain development of"),
]


def overclaim(sentence: str, evidence_texts: Sequence[str]) -> str | None:
    return _swap(sentence, _OVERCLAIM, evidence_texts)


def _absent(evidence_text: str, replacement: str) -> bool:
    """The replacement phrase does not occur in the evidence (case-insensitive)."""
    return replacement.lower() not in evidence_text.lower()


def _swap(
    sentence: str, table: Sequence[tuple[str, str]], evidence_texts: Sequence[str]
) -> str | None:
    evidence = "\n".join(evidence_texts)
    s = clean_sentence(sentence)
    for pat, repl in table:
        if re.search(pat, s) and _absent(evidence, repl):
            return re.sub(pat, repl, s, count=1)
    return None


_NEGATOR = re.compile(r"\b(?:no|not|never|without|cannot|none|neither|nor)\b|n't", re.IGNORECASE)


def negate(sentence: str, evidence_texts: Sequence[str]) -> str | None:
    """Flip the sentence's polarity. A rule that adds a negator is skipped when the sentence
    already has one: stacking a second gives a double negative whose truth value is not clear, so
    the label H would not be sure. Rules that remove a negator ("should not be" -> "should be")
    always apply."""
    has_negator = bool(_NEGATOR.search(sentence))
    rules = [(p, r) for p, r in _NEGATIONS if _NEGATOR.search(p) or not has_negator]
    return _swap(sentence, rules, evidence_texts)


def swap_frequency(sentence: str, evidence_texts: Sequence[str]) -> str | None:
    return _swap(sentence, _FREQUENCY, evidence_texts)


def swap_population(sentence: str, evidence_texts: Sequence[str]) -> str | None:
    return _swap(sentence, _POPULATION, evidence_texts)


def change_number(sentence: str, evidence_texts: Sequence[str], rng: random.Random) -> str | None:
    """Replace one number that has a unit by a different one that the evidence does not state."""
    s = clean_sentence(sentence)
    matches = list(_QUANTITY.finditer(s))
    rng.shuffle(matches)
    for m in matches:
        value = parse_number(m.group("num"))
        if value is None or value == 0:
            continue
        options = [value * 2, value / 2, value * 10, value + (1 if value < 10 else 10)]
        rng.shuffle(options)
        for new in options:
            text = _format_number(new)
            candidate = s[: m.start("num")] + text + s[m.end("num") :]
            # The new quantity must be absent from the evidence, or the claim might be true.
            if new != value and missing_quantities(candidate, evidence_texts):
                return candidate
    return None


def _format_number(d: Decimal) -> str:
    d = d.normalize()
    text = format(d, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text

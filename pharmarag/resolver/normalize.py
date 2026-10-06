"""Mention normalisation (spec section 7, cascade step 1).

Lowercase, strip dose and dose-form words, collapse whitespace. Salt words (hydrochloride,
besylate, ...) are kept on purpose: they decide between an ingredient match and a salt match,
and the alias table lists salts under their own names. Only the abbreviation "hcl" is expanded,
because the alias table spells it out.
"""

from __future__ import annotations

import re
import unicodedata

# Separators that join or wrap names ("zoloft (sertraline)", "a/b", "name, 25 mg"). Hyphens
# stay: brand names such as "toprol-xl" contain them.
_SEPARATORS = re.compile(r"[,;:()\[\]{}/\\\"'*]+")

_UNITS = "mg|mcg|ug|µg|g|gm|kg|ml|l|iu|unit|units|meq|mmol|%|mg/ml|mcg/ml|u"
# 25, 12.5, 25mg, 0.1%, 500mcg. A bare number is a dose too: "hydroxyzine 25".
_DOSE_TOKEN = re.compile(rf"^\d+(?:[.,]\d+)?(?:{_UNITS})?$")

# SPEC: the spec says "dose and form words (mg, tablet, ER, and so on)" without a list.
# These are the lists. Every entry is a word that is never a drug name on its own.
_DOSE_FORM_WORDS = frozenset(
    {
        "mg", "mcg", "ug", "µg", "g", "gm", "ml", "iu", "unit", "units", "meq", "mmol", "%",
        "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "caplet",
        "caplets", "pill", "pills", "injection", "injectable", "inj", "solution", "suspension",
        "syrup", "elixir", "cream", "ointment", "gel", "lotion", "patch", "spray", "inhaler",
        "drops", "oral", "topical", "sublingual", "chewable", "dispersible", "disintegrating",
        "film", "coated", "enteric", "dose", "doses", "dosage", "strength", "generic", "brand",
        "pack", "vial", "vials", "ampule", "ampules", "prefilled", "kit", "rx",
    }
)  # fmt: skip
# Release-type words. They are part of some brand names ("glucotrol xl", "toprol-xl"), so the
# alias table is tried once with them kept (`collapse`) before they are stripped (`normalize`).
_RELEASE_WORDS = frozenset(
    {
        "extended", "delayed", "sustained", "immediate", "controlled", "modified", "release",
        "er", "xr", "xl", "sr", "cr", "dr", "la", "odt", "ir",
    }
)  # fmt: skip

_ABBREVIATIONS = {"hcl": "hydrochloride"}


def _tokens(mention: str, strip: frozenset[str]) -> str:
    text = unicodedata.normalize("NFKC", mention).lower()
    text = _SEPARATORS.sub(" ", text)
    tokens: list[str] = []
    for tok in text.split():
        tok = tok.strip(".")
        if not tok or tok in strip or _DOSE_TOKEN.match(tok):
            continue
        tokens.append(_ABBREVIATIONS.get(tok, tok))
    return " ".join(tokens)


_ALL_FORM_WORDS = _DOSE_FORM_WORDS | _RELEASE_WORDS


def normalize(mention: str) -> str:
    """The cascade's working form of a mention. Empty if nothing but dose/form words remain."""
    return _tokens(mention, _ALL_FORM_WORDS)


def collapse(mention: str) -> str:
    """Like `normalize` but keeps release-type words (ER, XL, ...). The alias table is also
    tried with this form: "Glucotrol XL 10 mg" is the brand "glucotrol xl"."""
    return _tokens(mention, _DOSE_FORM_WORDS)

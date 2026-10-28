"""Plain-language `message` for every response that is not an answer.

A bare status (`declined`, `not_covered`, ...) tells a program what happened and tells a person
nothing. The message says why, and what to do next, in one or two sentences. `notice` (the
not-medical-advice line) is separate and unchanged.

Messages are built from data the graph already holds (resolution names, candidates, mentions).
They never include text from the question itself beyond a drug name the user typed, and they are
fixed English: the strings are the contract that clients and tests read.
"""

from pharmarag.graph.state import Resolution, Status

PATIENT_ADVICE = (
    "I can't give personal medical advice \N{EM DASH} please ask a pharmacist or doctor. "
    "I can tell you what a drug's FDA label says, e.g. "
    "'What does the acetaminophen label say about fever?'"
)
OUT_OF_SCOPE = (
    "That is outside what I can answer: I only report what FDA prescription drug labels say. "
    "Try a question about a specific drug's label, e.g. "
    "'What does the lisinopril label say about side effects?'"
)
INSUFFICIENT_EVIDENCE = (
    "The drug's label doesn't answer this question, or no answer could be checked against it, "
    "so I'm not showing one."
)
NOT_RECOGNISED = "I didn't recognise {names}. Please check the spelling and try again."
NO_DRUG_NAMED = (
    "I couldn't tell which drug you mean. Please name the drug (and check its spelling) "
    "and try again."
)


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _not_covered(resolutions: list[Resolution]) -> str:
    names = list(
        dict.fromkeys(r.name or r.mention for r in resolutions if r.status == "not_covered")
    )
    if len(names) == 1:
        subject, its = _capitalise(names[0]), "its label isn't"
    else:
        subject, its = _capitalise(_join(names) if names else "that drug"), "their labels aren't"
    return (
        f"{subject} resolved, but {its} in this service's corpus "
        "(prescription labels only; over-the-counter drugs are not covered)."
    )


def _clarification(resolutions: list[Resolution]) -> str:
    ambiguous = next((r for r in resolutions if r.status == "ambiguous"), None)
    options = (
        ", ".join(f"{c.name} ({c.ingredient_key})" for c in ambiguous.candidates[:3])
        if ambiguous
        else ""
    )
    what = f"'{ambiguous.mention}' could be {options}." if ambiguous else "That name is ambiguous."
    return (
        f"{what} Resend your question with pinned_rxcuis set to the ingredient key "
        "of the drug you mean."
    )


def _not_recognised(resolutions: list[Resolution]) -> str:
    unknown = [f"'{r.mention}'" for r in resolutions if r.status == "unknown"]
    return NOT_RECOGNISED.format(names=_join(unknown)) if unknown else NO_DRUG_NAMED


def status_message(
    status: Status, *, intent: str | None, resolutions: list[Resolution]
) -> str | None:
    """The message for a response with this status, or None for `answered`."""
    if status == "answered":
        return None
    if status == "declined":
        return PATIENT_ADVICE if intent == "patient_advice" else OUT_OF_SCOPE
    if status == "not_covered":
        return _not_covered(resolutions)
    if status == "needs_clarification":
        return _clarification(resolutions)
    if status == "drug_not_recognised":
        return _not_recognised(resolutions)
    return INSUFFICIENT_EVIDENCE

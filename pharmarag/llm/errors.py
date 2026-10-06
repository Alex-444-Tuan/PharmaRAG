"""Errors raised by LLM providers. The API maps all of them to HTTP 503 (spec section 10),
except where a graph node handles them first (a refusal in `analyze` or `answer`).
"""

# Appended by a provider to `complete()` / `stream()` output that stopped at max_tokens. The
# LLMClient protocol returns plain text, so truncation travels in-band. The writer removes the
# marker and drops the last, possibly incomplete, line.
TRUNCATION_MARKER = "\n<<TRUNCATED_AT_MAX_TOKENS>>"


class LLMError(RuntimeError):
    """Base class for provider-independent LLM failures."""


class LLMUnavailableError(LLMError):
    """Timeout, rate limit, 5xx, or connection failure after the SDK's own retries."""


class LLMOutputError(LLMError):
    """The model answered but not in a usable form (truncation, invalid JSON)."""


class LLMRefusalError(LLMOutputError):
    """The provider's safety layer declined the request (`stop_reason == "refusal"`)."""

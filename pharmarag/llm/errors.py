"""Errors raised by LLM providers. The API maps all of them to HTTP 503 (spec section 10)."""


class LLMError(RuntimeError):
    """Base class for provider-independent LLM failures."""


class LLMUnavailableError(LLMError):
    """Timeout, rate limit, 5xx, or connection failure after the SDK's own retries."""


class LLMOutputError(LLMError):
    """The model answered but not in a usable form (refusal, truncation, invalid JSON)."""

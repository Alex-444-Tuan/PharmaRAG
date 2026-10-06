"""Pick the LLM provider from settings."""

from pharmarag.app.settings import Settings
from pharmarag.llm.anthropic import AnthropicLLM
from pharmarag.llm.base import FakeLLM, LLMClient


def get_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM(model_id=settings.llm_model or "fake-llm")
    model = settings.llm_model
    if not model or model.endswith("-latest"):
        raise ValueError("LLM_MODEL must be a pinned model ID, not empty or a '-latest' alias")
    return AnthropicLLM(model, api_key=settings.llm_api_key, timeout_s=settings.llm_timeout_s)

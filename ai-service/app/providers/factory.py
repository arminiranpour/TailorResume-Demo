from functools import lru_cache

from app.config import get_config
from app.providers.api_provider import ApiProvider
from app.providers.base import LLMProvider
from app.providers.errors import ProviderConfigurationError
from app.providers.local_provider import LocalProvider


@lru_cache
def get_provider() -> LLMProvider:
    config = get_config()
    provider_name = config.llm_provider.strip().lower()
    if provider_name == "local":
        return LocalProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout_seconds=config.llm_timeout_seconds,
        )
    if provider_name == "api":
        return ApiProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout_seconds=config.llm_timeout_seconds,
        )
    raise ProviderConfigurationError(f"Unsupported LLM provider: {config.llm_provider}")

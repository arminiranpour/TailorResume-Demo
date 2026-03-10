from typing import Dict, List, Optional

from .base import LLMProvider
from .errors import ProviderConfigurationError


class ApiProvider(LLMProvider):
    """Stub for cloud API providers (OpenAI/Anthropic/etc.)."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        _ = base_url, model, timeout_seconds

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, object]] = None,
        temperature: float = 0,
        seed: int = 0,
        timeout: Optional[int] = None,
    ) -> str:
        _ = messages, json_schema, temperature, seed, timeout
        raise ProviderConfigurationError(
            "ApiProvider selected successfully, but cloud API transport is not "
            "implemented yet. This is an intentional stub for Phase 6; set "
            "LLM_PROVIDER=local to use the local provider."
        )

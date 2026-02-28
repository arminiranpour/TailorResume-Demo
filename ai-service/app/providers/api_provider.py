from typing import Any, Dict

from .base import LLMProvider


class ApiProvider(LLMProvider):
    """Stub for cloud API providers (OpenAI/Anthropic/etc.)."""

    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("ApiProvider.generate_structured is not implemented")

    def rewrite_text(self, prompt: str) -> str:
        raise NotImplementedError("ApiProvider.rewrite_text is not implemented")

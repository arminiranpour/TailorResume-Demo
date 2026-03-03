from typing import Dict, List, Optional

from .base import LLMProvider


class ApiProvider(LLMProvider):
    """Stub for cloud API providers (OpenAI/Anthropic/etc.)."""

    def generate(self, messages: List[Dict[str, str]], *, timeout_seconds: Optional[int] = None) -> str:
        raise NotImplementedError("ApiProvider.generate is not implemented")

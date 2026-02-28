from typing import Any, Dict

from .base import LLMProvider


class LocalProvider(LLMProvider):
    """Stub for local model runtime (Ollama)."""

    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider.generate_structured is not implemented")

    def rewrite_text(self, prompt: str) -> str:
        raise NotImplementedError("LocalProvider.rewrite_text is not implemented")

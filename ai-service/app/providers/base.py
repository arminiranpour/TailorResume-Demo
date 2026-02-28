from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMProvider(ABC):
    """Provider interface for LLM-backed structured extraction and rewriting."""

    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return JSON that conforms to the provided schema."""
        raise NotImplementedError

    @abstractmethod
    def rewrite_text(self, prompt: str) -> str:
        """Return rewritten text based on a deterministic prompt."""
        raise NotImplementedError

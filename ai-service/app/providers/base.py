from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Provider interface for LLM-backed structured extraction and rewriting."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], *, timeout_seconds: Optional[int] = None) -> str:
        """Return raw model output content."""
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Provider interface for LLM-backed structured extraction and rewriting.

    The contract is intentionally small and deterministic:
    - `messages` is the canonical chat payload used across pipelines.
    - `json_schema` is optional and may be ignored by providers that do not support it.
    - `temperature`, `seed`, and `timeout` must be accepted to preserve deterministic behavior.
    - Returns a raw string model output (no provider-specific wrapper objects).
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, object]] = None,
        temperature: float = 0,
        seed: int = 0,
        timeout: Optional[int] = None,
    ) -> str:
        """Return raw model output content."""
        raise NotImplementedError

from typing import Dict, List, Optional

import requests

from .base import LLMProvider
from .errors import ProviderConnectionError, ProviderResponseError, ProviderTimeoutError


class LocalProvider(LLMProvider):
    """Ollama-backed local provider."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, object]] = None,
        temperature: float = 0,
        seed: int = 0,
        timeout: Optional[int] = None,
    ) -> str:
        _ = json_schema
        timeout_value = timeout if timeout is not None else self._timeout_seconds
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
            },
        }
        url = f"{self._base_url}/api/chat"
        try:
            response = requests.post(url, json=payload, timeout=timeout_value)
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {timeout_value} seconds"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderConnectionError(f"Ollama connection error: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderResponseError(
                f"Ollama error {response.status_code}: {response.text.strip()}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError("Ollama response was not valid JSON") from exc
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ProviderResponseError("Ollama response missing message content")
        return content

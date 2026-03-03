from typing import Dict, List, Optional

import requests

from .base import LLMProvider


class LLMConnectionError(RuntimeError):
    pass


class LLMTimeoutError(RuntimeError):
    pass


class LocalProvider(LLMProvider):
    """Ollama-backed local provider."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(self, messages: List[Dict[str, str]], *, timeout_seconds: Optional[int] = None) -> str:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 0,
            },
        }
        url = f"{self._base_url}/api/chat"
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(f"Ollama request timed out after {timeout} seconds") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(f"Ollama connection error: {exc}") from exc
        if response.status_code >= 400:
            raise LLMConnectionError(
                f"Ollama error {response.status_code}: {response.text.strip()}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMConnectionError("Ollama response was not valid JSON") from exc
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMConnectionError("Ollama response missing message content")
        return content

import os
import sys
import inspect

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import get_config
from app.providers.api_provider import ApiProvider
from app.providers.base import LLMProvider
from app.providers.errors import ProviderConfigurationError
from app.providers.factory import get_provider
from app.providers.local_provider import LocalProvider


def _reset_provider(monkeypatch, provider_name: str) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider_name)
    get_config.cache_clear()
    get_provider.cache_clear()


def test_factory_returns_local_provider(monkeypatch):
    _reset_provider(monkeypatch, "local")
    provider = get_provider()
    assert isinstance(provider, LocalProvider)


def test_factory_returns_api_provider(monkeypatch):
    _reset_provider(monkeypatch, "api")
    provider = get_provider()
    assert isinstance(provider, ApiProvider)


def test_factory_invalid_provider_raises(monkeypatch):
    _reset_provider(monkeypatch, "invalid")
    with pytest.raises(ProviderConfigurationError):
        get_provider()


def test_api_provider_generate_raises_stub_error():
    provider = ApiProvider(
        base_url="http://example.com",
        model="stub-model",
        timeout_seconds=1,
    )
    with pytest.raises(ProviderConfigurationError) as excinfo:
        provider.generate([{"role": "user", "content": "ping"}])
    message = str(excinfo.value)
    assert "ApiProvider" in message
    assert "not implemented" in message.lower()


def test_provider_interface_signatures_match():
    assert issubclass(LocalProvider, LLMProvider)
    assert issubclass(ApiProvider, LLMProvider)
    assert inspect.signature(LocalProvider.generate) == inspect.signature(
        ApiProvider.generate
    )
    assert inspect.signature(ApiProvider.generate) == inspect.signature(
        LLMProvider.generate
    )

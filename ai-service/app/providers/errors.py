class ProviderError(RuntimeError):
    """Base error for all provider failures."""


class ProviderConnectionError(ProviderError):
    """Raised when a provider cannot be reached or a transport error occurs."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns a malformed or unexpected response."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid or unsupported."""

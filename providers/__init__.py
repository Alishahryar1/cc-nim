"""Providers package - implement your own provider by extending BaseProvider."""

from .base import BaseProvider, ProviderConfig
from .nvidia_nim import NvidiaNimProvider
from .exceptions import (
    ProviderError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    OverloadedError,
    APIError,
)
from .registry import (
    register_provider,
    get_provider_class,
    list_providers,
    create_provider,
)

# Auto-register built-in providers
register_provider("nvidia_nim", NvidiaNimProvider)

__all__ = [
    # Base classes
    "BaseProvider",
    "ProviderConfig",
    # Built-in providers
    "NvidiaNimProvider",
    # Exceptions
    "ProviderError",
    "AuthenticationError",
    "InvalidRequestError",
    "RateLimitError",
    "OverloadedError",
    "APIError",
    # Registry functions
    "register_provider",
    "get_provider_class",
    "list_providers",
    "create_provider",
]

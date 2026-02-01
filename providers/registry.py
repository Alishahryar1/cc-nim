"""Provider registry for dynamic provider selection."""

from typing import Dict, Type, Optional
from .base import BaseProvider, ProviderConfig


# Global registry mapping provider names to classes
_PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {}


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a provider class with a name.

    Args:
        name: Unique identifier for the provider (e.g., "nvidia_nim", "openai")
        provider_class: The provider class that extends BaseProvider
    """
    _PROVIDER_REGISTRY[name] = provider_class


def get_provider_class(name: str) -> Optional[Type[BaseProvider]]:
    """Get a provider class by name.

    Args:
        name: The provider name to look up

    Returns:
        The provider class if found, None otherwise
    """
    return _PROVIDER_REGISTRY.get(name)


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        List of registered provider names
    """
    return list(_PROVIDER_REGISTRY.keys())


def create_provider(name: str, config: ProviderConfig) -> BaseProvider:
    """Create a provider instance by name.

    Args:
        name: The provider name to instantiate
        config: Configuration for the provider

    Returns:
        An instance of the requested provider

    Raises:
        ValueError: If the provider name is not registered
    """
    provider_class = get_provider_class(name)
    if provider_class is None:
        available = ", ".join(list_providers()) or "none"
        raise ValueError(f"Unknown provider: '{name}'. Available providers: {available}")
    return provider_class(config)

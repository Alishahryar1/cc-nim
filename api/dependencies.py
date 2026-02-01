"""Dependency injection for FastAPI."""

from typing import Optional
from config.settings import Settings, get_settings as _get_settings, NVIDIA_NIM_BASE_URL
from providers.base import BaseProvider, ProviderConfig
from providers import create_provider


# Global provider instance (singleton)
_provider: Optional[BaseProvider] = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def get_provider() -> BaseProvider:
    """Get or create the provider instance using registry."""
    global _provider
    if _provider is None:
        settings = get_settings()
        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            rate_limit=settings.nvidia_nim_rate_limit,
            rate_window=settings.nvidia_nim_rate_window,
        )
        _provider = create_provider(settings.provider, config)
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        client = getattr(_provider, "_client", None)
        if client and hasattr(client, "aclose"):
            await client.aclose()
    _provider = None

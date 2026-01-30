"""Dependency injection for FastAPI."""

from typing import Optional
from config.settings import Settings, get_settings as _get_settings, NVIDIA_NIM_BASE_URL
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider


# Global provider instance (singleton)
_provider: Optional[NvidiaNimProvider] = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def get_provider() -> NvidiaNimProvider:
    """Get or create the NvidiaNimProvider instance."""
    global _provider
    if _provider is None:
        settings = get_settings()
        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,  # Use constant, not from settings
            rate_limit=settings.nvidia_nim_rate_limit,
            rate_window=settings.nvidia_nim_rate_window,
        )
        _provider = NvidiaNimProvider(config)
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider and hasattr(_provider, "_client"):
        await _provider._client.aclose()
    _provider = None

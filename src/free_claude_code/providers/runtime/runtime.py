"""One closable generation of lazily constructed provider clients."""

from collections.abc import MutableMapping

from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider

from .cache import ProviderCache


class ProviderRuntime:
    """Own provider instances for one immutable settings snapshot."""

    def __init__(
        self,
        settings: Settings,
        providers: MutableMapping[str, BaseProvider] | None = None,
    ) -> None:
        self.settings = settings
        self._provider_cache = ProviderCache(settings, providers)

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already cached."""
        return self._provider_cache.is_cached(provider_id)

    def resolve_provider(self, provider_id: str) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        return self._provider_cache.get(provider_id)

    async def cleanup(self) -> None:
        """Release every provider client constructed by this generation."""
        await self._provider_cache.cleanup()

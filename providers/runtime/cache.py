"""Provider instance cache and cleanup."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping

from config.settings import Settings
from providers.base import BaseProvider

from .factory import create_provider

ProviderCreator = Callable[[str, Settings], BaseProvider]


class ProviderCache:
    """Cache provider instances for one settings snapshot."""

    def __init__(
        self,
        settings: Settings,
        providers: MutableMapping[str, BaseProvider] | None = None,
        *,
        creator: ProviderCreator = create_provider,
    ) -> None:
        self._settings = settings
        self._providers = providers if providers is not None else {}
        self._creator = creator

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already cached."""
        return provider_id in self._providers

    def get(self, provider_id: str) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        if provider_id not in self._providers:
            self._providers[provider_id] = self._creator(provider_id, self._settings)
        return self._providers[provider_id]

    async def cleanup(self) -> None:
        """Clean up every cached provider, then clear the cache."""
        items = list(self._providers.items())
        errors: list[Exception] = []
        try:
            for _provider_id, provider in items:
                try:
                    await provider.cleanup()
                except Exception as exc:
                    errors.append(exc)
        finally:
            self._providers.clear()
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("One or more provider cleanups failed", errors)

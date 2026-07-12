"""One closable generation of lazily constructed provider clients."""

import asyncio
from collections.abc import MutableMapping

from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider

from .factory import create_provider


class ProviderRuntime:
    """Own provider instances for one immutable settings snapshot."""

    def __init__(
        self,
        settings: Settings,
        providers: MutableMapping[str, BaseProvider] | None = None,
    ) -> None:
        self.settings = settings
        self._providers = providers if providers is not None else {}

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already cached."""
        return provider_id in self._providers

    def resolve_provider(self, provider_id: str) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        if provider_id not in self._providers:
            self._providers[provider_id] = create_provider(provider_id, self.settings)
        return self._providers[provider_id]

    async def cleanup(self) -> None:
        """Release every provider client constructed by this generation."""
        errors: list[Exception] = []
        for provider_id, provider in list(self._providers.items()):
            try:
                await provider.cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(exc)
            else:
                self._providers.pop(provider_id, None)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("One or more provider cleanups failed", errors)

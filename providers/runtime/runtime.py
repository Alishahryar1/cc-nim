"""App-scoped provider runtime orchestration."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping

from config.settings import Settings
from providers.base import BaseProvider
from providers.model_listing import ProviderModelInfo

from .cache import ProviderCache
from .discovery import ProviderModelDiscovery
from .model_cache import ProviderModelCache
from .validation import ConfiguredModelValidator


class ProviderRuntime:
    """Own provider instances, model discovery, validation, and cleanup."""

    def __init__(
        self,
        settings: Settings,
        providers: MutableMapping[str, BaseProvider] | None = None,
    ) -> None:
        self.settings = settings
        self._provider_cache = ProviderCache(settings, providers)
        self._model_cache = ProviderModelCache()
        self._discovery = ProviderModelDiscovery(
            settings,
            self.resolve_provider,
            self._model_cache,
        )
        self._validator = ConfiguredModelValidator(
            settings,
            self.resolve_provider,
            self._model_cache,
        )

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already cached."""
        return self._provider_cache.is_cached(provider_id)

    def resolve_provider(self, provider_id: str) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        return self._provider_cache.get(provider_id)

    def cache_model_ids(self, provider_id: str, model_ids: Iterable[str]) -> None:
        """Store raw provider model ids for later instant API responses."""
        self._model_cache.cache_model_ids(provider_id, model_ids)

    def cache_model_infos(
        self, provider_id: str, model_infos: Iterable[ProviderModelInfo]
    ) -> None:
        """Store provider model metadata for later instant API responses."""
        self._model_cache.cache_model_infos(provider_id, model_infos)

    def cached_model_ids(self) -> dict[str, frozenset[str]]:
        """Return cached raw provider model ids by provider."""
        return self._model_cache.cached_model_ids()

    def cached_model_supports_thinking(
        self, provider_id: str, model_id: str
    ) -> bool | None:
        """Return cached thinking support when a provider exposes it."""
        return self._model_cache.cached_model_supports_thinking(provider_id, model_id)

    def cached_prefixed_model_refs(self) -> tuple[str, ...]:
        """Return cached provider models in user-selectable ``provider/model`` form."""
        return self._model_cache.cached_prefixed_model_refs()

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        """Return cached provider models with user-selectable prefixed ids."""
        return self._model_cache.cached_prefixed_model_infos()

    async def refresh_model_list_cache(self, *, only_missing: bool = False) -> None:
        """Best-effort refresh of model lists for usable providers."""
        await self._discovery.refresh_model_list_cache(only_missing=only_missing)

    def start_model_list_refresh(self) -> None:
        """Start a non-blocking cache warmup for missing eligible provider lists."""
        self._discovery.start_model_list_refresh()

    async def validate_configured_models(self) -> None:
        """Fail unless every configured chat model exists upstream."""
        await self._validator.validate_configured_models()

    async def cleanup(self) -> None:
        """Cancel discovery, clean provider instances, and clear model metadata."""
        try:
            await self._discovery.cleanup()
            await self._provider_cache.cleanup()
        finally:
            self._model_cache.clear()

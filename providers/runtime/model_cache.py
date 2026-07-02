"""Provider model-list metadata cache."""

from __future__ import annotations

from collections.abc import Iterable

from config.provider_catalog import SUPPORTED_PROVIDER_IDS
from providers.model_listing import ProviderModelInfo, model_infos_from_ids


class ProviderModelCache:
    """Store provider model metadata for instant model-list responses."""

    def __init__(self) -> None:
        self._model_infos_by_provider: dict[str, dict[str, ProviderModelInfo]] = {}

    def cache_model_ids(self, provider_id: str, model_ids: Iterable[str]) -> None:
        """Store raw provider model ids with unknown capability metadata."""
        self.cache_model_infos(provider_id, model_infos_from_ids(model_ids))

    def cache_model_infos(
        self, provider_id: str, model_infos: Iterable[ProviderModelInfo]
    ) -> None:
        """Store provider model metadata by raw provider model id."""
        clean_infos = {
            info.model_id: info for info in model_infos if info.model_id.strip()
        }
        self._model_infos_by_provider[provider_id] = clean_infos

    def cached_model_ids(self) -> dict[str, frozenset[str]]:
        """Return cached raw provider model ids by provider."""
        return {
            provider_id: frozenset(infos)
            for provider_id, infos in self._model_infos_by_provider.items()
        }

    def has_provider(self, provider_id: str) -> bool:
        """Return whether this provider has any cached model-list result."""
        return provider_id in self._model_infos_by_provider

    def cached_model_supports_thinking(
        self, provider_id: str, model_id: str
    ) -> bool | None:
        """Return cached thinking support when a provider exposes it."""
        info = self._model_infos_by_provider.get(provider_id, {}).get(model_id)
        if info is None:
            return None
        return info.supports_thinking

    def cached_prefixed_model_refs(self) -> tuple[str, ...]:
        """Return cached provider models in user-selectable ``provider/model`` form."""
        return tuple(info.model_id for info in self.cached_prefixed_model_infos())

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        """Return cached provider models with user-selectable prefixed ids."""
        infos: list[ProviderModelInfo] = []
        for provider_id in SUPPORTED_PROVIDER_IDS:
            provider_infos = self._model_infos_by_provider.get(provider_id, {})
            infos.extend(
                ProviderModelInfo(
                    model_id=f"{provider_id}/{info.model_id}",
                    supports_thinking=info.supports_thinking,
                )
                for info in sorted(
                    provider_infos.values(), key=lambda item: item.model_id
                )
            )
        return tuple(infos)

    def clear(self) -> None:
        """Clear all cached model metadata."""
        self._model_infos_by_provider.clear()

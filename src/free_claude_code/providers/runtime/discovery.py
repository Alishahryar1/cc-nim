"""Provider model-list discovery and background refresh."""

import asyncio
from collections.abc import Callable

from loguru import logger

from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider

from .config import has_provider_configuration
from .model_cache import ProviderModelCache
from .validation import provider_query_failure_reason

ProviderResolver = Callable[[str], BaseProvider]


def referenced_provider_ids(settings: Settings) -> frozenset[str]:
    """Return provider ids referenced by configured chat model refs."""
    return frozenset(ref.provider_id for ref in configured_chat_model_refs(settings))


def model_cache_provider_ids_for_settings(settings: Settings) -> tuple[str, ...]:
    """Return providers whose model metadata is valid for these settings."""
    return tuple(
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if has_provider_configuration(descriptor, settings)
    )


def model_list_provider_ids_for_settings(settings: Settings) -> tuple[str, ...]:
    """Return providers worth discovering for this process configuration."""
    referenced_ids = referenced_provider_ids(settings)
    return tuple(
        provider_id
        for provider_id in model_cache_provider_ids_for_settings(settings)
        if not PROVIDER_CATALOG[provider_id].local or provider_id in referenced_ids
    )


class ProviderModelDiscovery:
    """Refresh provider model-list metadata for one provider runtime."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        model_cache: ProviderModelCache,
    ) -> None:
        self._settings = settings
        self._provider_resolver = provider_resolver
        self._model_cache = model_cache

    async def refresh_model_list_cache(
        self, *, only_missing: bool = False
    ) -> ProviderModelRefreshResult:
        """Best-effort refresh of model lists for usable providers."""
        provider_ids = model_list_provider_ids_for_settings(self._settings)
        if only_missing:
            provider_ids = tuple(
                provider_id
                for provider_id in provider_ids
                if not self._model_cache.has_provider(provider_id)
            )
        return await self._refresh_model_infos(provider_ids)

    async def _refresh_model_infos(
        self, provider_ids: tuple[str, ...]
    ) -> ProviderModelRefreshResult:
        failed_provider_ids: list[str] = []
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id in provider_ids:
            try:
                provider = self._provider_resolver(provider_id)
            except Exception as exc:
                self._log_discovery_failure(provider_id, exc)
                failed_provider_ids.append(provider_id)
                continue
            tasks[provider_id] = asyncio.create_task(provider.list_model_infos())

        refreshed_provider_ids: list[str] = []
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (provider_id, _task), result in zip(
                tasks.items(), results, strict=True
            ):
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    self._log_discovery_failure(provider_id, result)
                    failed_provider_ids.append(provider_id)
                    continue
                self._model_cache.cache_model_infos(provider_id, result)
                refreshed_provider_ids.append(provider_id)
                logger.info(
                    "Provider model discovery cached: provider={} models={}",
                    provider_id,
                    len(result),
                )

        return ProviderModelRefreshResult(
            refreshed_provider_ids=tuple(refreshed_provider_ids),
            failed_provider_ids=tuple(failed_provider_ids),
        )

    def _log_discovery_failure(self, provider_id: str, exc: BaseException) -> None:
        logger.warning(
            "Provider model discovery skipped: provider={} reason={}",
            provider_id,
            provider_query_failure_reason(exc, self._settings),
        )

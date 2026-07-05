"""Provider model-list discovery and background refresh."""

import asyncio
from collections.abc import Callable
from contextlib import suppress

from loguru import logger

from config.model_refs import configured_chat_model_refs
from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from providers.base import BaseProvider
from providers.model_listing import ProviderModelInfo

from .config import provider_credential
from .model_cache import ProviderModelCache
from .validation import provider_query_failure_reason

ProviderResolver = Callable[[str], BaseProvider]


def referenced_provider_ids(settings: Settings) -> frozenset[str]:
    """Return provider ids referenced by configured chat model refs."""
    return frozenset(ref.provider_id for ref in configured_chat_model_refs(settings))


def model_list_provider_ids_for_settings(settings: Settings) -> tuple[str, ...]:
    """Return providers worth discovering for this process configuration."""
    referenced_ids = referenced_provider_ids(settings)
    provider_ids: list[str] = []
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if descriptor.static_credential is not None:
            if provider_id in referenced_ids:
                provider_ids.append(provider_id)
            continue
        if (
            descriptor.credential_env is not None
            and provider_credential(descriptor, settings).strip()
        ):
            provider_ids.append(provider_id)
    return tuple(provider_ids)


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
        self._refresh_task: asyncio.Task[None] | None = None

    async def refresh_model_list_cache(self, *, only_missing: bool = False) -> None:
        """Best-effort refresh of model lists for usable providers."""
        provider_ids = model_list_provider_ids_for_settings(self._settings)
        if only_missing:
            provider_ids = tuple(
                provider_id
                for provider_id in provider_ids
                if not self._model_cache.has_provider(provider_id)
            )
        await self._refresh_model_infos(provider_ids)

    def start_model_list_refresh(self) -> None:
        """Start a non-blocking cache warmup for missing eligible provider lists."""
        if self._refresh_task is not None and not self._refresh_task.done():
            return

        provider_ids = tuple(
            provider_id
            for provider_id in model_list_provider_ids_for_settings(self._settings)
            if not self._model_cache.has_provider(provider_id)
        )
        if not provider_ids:
            logger.info(
                "Provider model discovery cache already warm: providers={}",
                len(self._model_cache.cached_model_ids()),
            )
            return

        self._refresh_task = asyncio.create_task(self._run_refresh(provider_ids))

    async def cleanup(self) -> None:
        """Cancel any background model-list refresh."""
        if self._refresh_task is None or self._refresh_task.done():
            return
        self._refresh_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._refresh_task

    async def _run_refresh(self, provider_ids: tuple[str, ...]) -> None:
        try:
            await self._refresh_model_infos(provider_ids)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Provider model discovery task failed: exc_type={}",
                type(exc).__name__,
            )

    async def _refresh_model_infos(self, provider_ids: tuple[str, ...]) -> None:
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id in provider_ids:
            try:
                provider = self._provider_resolver(provider_id)
            except Exception as exc:
                self._log_discovery_failure(provider_id, exc)
                continue
            tasks[provider_id] = asyncio.create_task(provider.list_model_infos())

        if not tasks:
            return

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for (provider_id, _task), result in zip(tasks.items(), results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                self._log_discovery_failure(provider_id, result)
                continue
            self._model_cache.cache_model_infos(provider_id, result)
            logger.info(
                "Provider model discovery cached: provider={} models={}",
                provider_id,
                len(result),
            )

    def _log_discovery_failure(self, provider_id: str, exc: BaseException) -> None:
        logger.warning(
            "Provider model discovery skipped: provider={} reason={}",
            provider_id,
            provider_query_failure_reason(exc, self._settings),
        )

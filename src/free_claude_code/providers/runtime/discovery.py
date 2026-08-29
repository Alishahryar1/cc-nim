"""Provider model-list discovery and background refresh."""

import asyncio
from collections.abc import Callable

import httpx
from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.model_listing import ModelListResponseError

from .config import has_provider_configuration
from .model_cache import ProviderModelCache

ProviderResolver = Callable[[str], BaseProvider]

# A local provider that is actually running answers over loopback in milliseconds.
OPPORTUNISTIC_LOCAL_DISCOVERY_TIMEOUT_SECONDS = 5.0


def _provider_query_failure_reason(exc: BaseException, settings: Settings) -> str:
    """Return a concise model-list query failure reason for user-facing logs."""
    if isinstance(exc, ModelListResponseError):
        return f"malformed model-list response: {exc.message}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"query failure: HTTP {exc.response.status_code}"
    if isinstance(exc, ApplicationUnavailableError):
        return f"query failure: {exc.message}"
    if isinstance(exc, ExecutionFailure) and settings.log_api_error_tracebacks:
        return f"query failure: {exc.message}"
    return f"query failure: {type(exc).__name__}"


def referenced_provider_ids(settings: Settings) -> tuple[str, ...]:
    """Return unique provider ids referenced by configured chat models."""
    return tuple(
        dict.fromkeys(ref.provider_id for ref in configured_chat_model_refs(settings))
    )


def model_cache_provider_ids_for_settings(
    settings: Settings,
    connected_provider_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return providers whose model metadata is valid for these settings."""
    configured = tuple(
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if has_provider_configuration(descriptor, settings)
    )
    available = set(configured) | set(connected_provider_ids)
    return tuple(
        provider_id for provider_id in PROVIDER_CATALOG if provider_id in available
    )


class ProviderModelDiscovery:
    """Refresh provider model-list metadata for one provider runtime."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        model_cache: ProviderModelCache,
        connected_provider_ids: tuple[str, ...] = (),
    ) -> None:
        self._settings = settings
        self._provider_resolver = provider_resolver
        self._model_cache = model_cache
        self._connected_provider_ids = connected_provider_ids

    async def warm_referenced_model_cache(self) -> ProviderModelRefreshResult:
        """Synchronously cache model metadata for routed providers."""
        return await self._refresh_model_infos(referenced_provider_ids(self._settings))

    async def refresh_model_list_cache(
        self, *, only_missing: bool = False
    ) -> ProviderModelRefreshResult:
        """Best-effort refresh of model lists for usable providers."""
        provider_ids = model_cache_provider_ids_for_settings(
            self._settings, self._connected_provider_ids
        )
        if only_missing:
            provider_ids = tuple(
                provider_id
                for provider_id in provider_ids
                if not self._model_cache.has_provider(provider_id)
            )
        return await self._refresh_model_infos(provider_ids)

    async def refresh_provider(self, provider_id: str) -> ProviderModelRefreshResult:
        """Refresh exactly one dynamically changed provider."""

        return await self._refresh_model_infos((provider_id,))

    async def _refresh_model_infos(
        self, provider_ids: tuple[str, ...]
    ) -> ProviderModelRefreshResult:
        failed_provider_ids: list[str] = []
        opportunistic_ids = self._opportunistic_local_ids()
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id in provider_ids:
            try:
                provider = self._provider_resolver(provider_id)
            except Exception as exc:
                self._record_discovery_failure(
                    provider_id, exc, failed_provider_ids, opportunistic_ids
                )
                continue
            query = provider.list_model_infos()
            if provider_id in opportunistic_ids:
                # Provider retry policy assumes a transient upstream fault. A local
                # endpoint nobody is running refuses instantly and stays refused, so
                # cap the whole opportunistic query instead of paying that backoff.
                query = asyncio.wait_for(
                    query, OPPORTUNISTIC_LOCAL_DISCOVERY_TIMEOUT_SECONDS
                )
            tasks[provider_id] = asyncio.create_task(query)

        refreshed_provider_ids: list[str] = []
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (provider_id, _task), result in zip(
                tasks.items(), results, strict=True
            ):
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    self._record_discovery_failure(
                        provider_id, result, failed_provider_ids, opportunistic_ids
                    )
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

    def _opportunistic_local_ids(self) -> frozenset[str]:
        """Return local providers queried on the chance one is actually running.

        A local provider carries a default base URL, so it is always configured
        even for the many installs that never run one. Querying it anyway is what
        lets its models reach the client model picker, but an unreachable endpoint
        is the expected state rather than a fault, so it must not be reported.
        A local provider the customer has routed to is excluded here: there a
        failure is real and keeps its existing warning.
        """

        referenced = set(referenced_provider_ids(self._settings))
        return frozenset(
            provider_id
            for provider_id, descriptor in PROVIDER_CATALOG.items()
            if descriptor.local and provider_id not in referenced
        )

    def _record_discovery_failure(
        self,
        provider_id: str,
        exc: BaseException,
        failed_provider_ids: list[str],
        opportunistic_ids: frozenset[str],
    ) -> None:
        """Log one failed model-list query and record only actionable failures."""
        if provider_id in opportunistic_ids:
            logger.debug(
                "Local provider model discovery unavailable: provider={} reason={}",
                provider_id,
                _provider_query_failure_reason(exc, self._settings),
            )
            return
        logger.warning(
            "Provider model discovery skipped: provider={} reason={}",
            provider_id,
            _provider_query_failure_reason(exc, self._settings),
        )
        failed_provider_ids.append(provider_id)

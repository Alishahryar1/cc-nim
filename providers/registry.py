"""Provider descriptors, factory, and runtime registry."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable, MutableMapping
from contextlib import suppress

import httpx
from loguru import logger

from config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderDescriptor,
)
from config.settings import ConfiguredChatModelRef, Settings
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import (
    AuthenticationError,
    ModelListResponseError,
    ProviderError,
    ServiceUnavailableError,
    UnknownProviderTypeError,
)
from providers.model_listing import ProviderModelInfo, model_infos_from_ids

ProviderFactory = Callable[[ProviderConfig, Settings], BaseProvider]

# Backwards-compatible name for the catalog (single source: ``config.provider_catalog``).
PROVIDER_DESCRIPTORS: dict[str, ProviderDescriptor] = PROVIDER_CATALOG


def _create_nvidia_nim(config: ProviderConfig, settings: Settings) -> BaseProvider:
    from providers.nvidia_nim import NvidiaNimProvider

    nim_settings = settings.nim
    if settings.nim_strip_tools:
        nim_settings = nim_settings.model_copy(
            update={
                "strip_tools": [
                    t.strip()
                    for t in settings.nim_strip_tools.split(",")
                    if t.strip()
                ]
            }
        )
    return NvidiaNimProvider(config, nim_settings=nim_settings)


def _create_open_router(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config)


def _create_mistral(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.mistral import MistralProvider

    return MistralProvider(config)


def _create_mistral_codestral(
    config: ProviderConfig, _settings: Settings
) -> BaseProvider:
    from providers.codestral import CodestralProvider

    return CodestralProvider(config)


def _create_deepseek(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config)


def _create_lmstudio(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config)


def _create_llamacpp(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.llamacpp import LlamaCppProvider

    return LlamaCppProvider(config)


def _create_ollama(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.ollama import OllamaProvider

    return OllamaProvider(config)


def _create_kimi(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.kimi import KimiProvider

    return KimiProvider(config)


def _create_wafer(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.wafer import WaferProvider

    return WaferProvider(config)


def _create_opencode(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.opencode import OpenCodeProvider

    return OpenCodeProvider(config)


def _create_opencode_go(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.opencode import OpenCodeProvider

    return OpenCodeProvider(config, provider_name="OPENCODE_GO")


def _create_zai(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.zai import ZaiProvider

    return ZaiProvider(config)


def _create_fireworks(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.fireworks import FireworksProvider

    return FireworksProvider(config)


def _create_gemini(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.gemini import GeminiProvider

    return GeminiProvider(config)


def _create_groq(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.groq import GroqProvider

    return GroqProvider(config)


def _create_cerebras(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.cerebras import CerebrasProvider

    return CerebrasProvider(config)


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "gemini": _create_gemini,
    "deepseek": _create_deepseek,
    "mistral": _create_mistral,
    "mistral_codestral": _create_mistral_codestral,
    "opencode": _create_opencode,
    "opencode_go": _create_opencode_go,
    "wafer": _create_wafer,
    "kimi": _create_kimi,
    "cerebras": _create_cerebras,
    "groq": _create_groq,
    "fireworks": _create_fireworks,
    "zai": _create_zai,
    "lmstudio": _create_lmstudio,
    "llamacpp": _create_llamacpp,
    "ollama": _create_ollama,
}

if set(PROVIDER_DESCRIPTORS) != set(SUPPORTED_PROVIDER_IDS) or set(
    PROVIDER_FACTORIES
) != set(SUPPORTED_PROVIDER_IDS):
    raise AssertionError(
        "PROVIDER_DESCRIPTORS, PROVIDER_FACTORIES, and SUPPORTED_PROVIDER_IDS are out of sync: "
        f"descriptors={set(PROVIDER_DESCRIPTORS)!r} factories={set(PROVIDER_FACTORIES)!r} "
        f"ids={set(SUPPORTED_PROVIDER_IDS)!r}"
    )


def _string_attr(settings: Settings, attr_name: str | None, default: str = "") -> str:
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def _credential_for(descriptor: ProviderDescriptor, settings: Settings) -> str:
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    if descriptor.credential_attr:
        return _string_attr(settings, descriptor.credential_attr)
    return ""


def _require_credential(descriptor: ProviderDescriptor, credential: str) -> None:
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    message = f"{descriptor.credential_env} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise AuthenticationError(message)


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings
) -> ProviderConfig:
    credential = _credential_for(descriptor, settings)
    _require_credential(descriptor, credential)
    base_url = _string_attr(
        settings, descriptor.base_url_attr, descriptor.default_base_url or ""
    )
    proxy = _string_attr(settings, descriptor.proxy_attr)
    return ProviderConfig(
        api_key=credential,
        base_url=base_url or descriptor.default_base_url,
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        enable_thinking=settings.enable_model_thinking,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )


def create_provider(provider_id: str, settings: Settings) -> BaseProvider:
    descriptor = PROVIDER_DESCRIPTORS.get(provider_id)
    if descriptor is None:
        supported = "', '".join(PROVIDER_DESCRIPTORS)
        raise UnknownProviderTypeError(
            f"Unknown provider_type: '{provider_id}'. Supported: '{supported}'"
        )

    config = build_provider_config(descriptor, settings)
    factory = PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        raise AssertionError(f"Unhandled provider descriptor: {provider_id}")
    return factory(config, settings)


def _format_provider_query_failures(
    refs: list[ConfiguredChatModelRef],
    exc: BaseException,
    settings: Settings,
) -> list[str]:
    reason = _provider_query_failure_reason(exc, settings)
    return [_format_model_validation_failure(ref, reason) for ref in refs]


def _format_missing_model_failure(ref: ConfiguredChatModelRef) -> str:
    return _format_model_validation_failure(ref, "missing model")


def _format_model_validation_failure(ref: ConfiguredChatModelRef, problem: str) -> str:
    return (
        f"sources={','.join(ref.sources)} provider={ref.provider_id} "
        f"model={ref.model_id} problem={problem}"
    )


def _provider_query_failure_reason(
    exc: BaseException,
    settings: Settings,
) -> str:
    if isinstance(exc, ModelListResponseError):
        return f"malformed model-list response: {exc.message}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"query failure: HTTP {exc.response.status_code}"
    if isinstance(exc, AuthenticationError):
        return f"query failure: {exc.message}"
    if isinstance(exc, ProviderError) and settings.log_api_error_tracebacks:
        return f"query failure: {exc.message}"
    return f"query failure: {type(exc).__name__}"


def _referenced_provider_ids(settings: Settings) -> frozenset[str]:
    return frozenset(ref.provider_id for ref in settings.configured_chat_model_refs())


def _model_list_provider_ids_for_settings(settings: Settings) -> tuple[str, ...]:
    """Return providers worth discovering for this process configuration."""
    referenced_provider_ids = _referenced_provider_ids(settings)
    provider_ids: list[str] = []
    for provider_id, descriptor in PROVIDER_DESCRIPTORS.items():
        if descriptor.static_credential is not None:
            if provider_id in referenced_provider_ids:
                provider_ids.append(provider_id)
            continue
        if (
            descriptor.credential_env is not None
            and _credential_for(descriptor, settings).strip()
        ):
            provider_ids.append(provider_id)
    return tuple(provider_ids)


def _log_model_discovery_failure(
    provider_id: str, exc: BaseException, settings: Settings
) -> None:
    logger.warning(
        "Provider model discovery skipped: provider={} reason={}",
        provider_id,
        _provider_query_failure_reason(exc, settings),
    )


class ProviderRegistry:
    """Cache and clean up provider instances by provider id."""

    def __init__(self, providers: MutableMapping[str, BaseProvider] | None = None):
        self._providers = providers if providers is not None else {}
        self._model_ids_by_provider: dict[str, frozenset[str]] = {}
        self._model_infos_by_provider: dict[str, dict[str, ProviderModelInfo]] = {}
        self._model_list_refresh_task: asyncio.Task[None] | None = None

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already in the cache."""
        return provider_id in self._providers

    def get(self, provider_id: str, settings: Settings) -> BaseProvider:
        if provider_id not in self._providers:
            self._providers[provider_id] = create_provider(provider_id, settings)
        return self._providers[provider_id]

    def cache_model_ids(self, provider_id: str, model_ids: Iterable[str]) -> None:
        """Store a provider model-list result for later instant API responses."""
        self.cache_model_infos(provider_id, model_infos_from_ids(model_ids))

    def cache_model_infos(
        self, provider_id: str, model_infos: Iterable[ProviderModelInfo]
    ) -> None:
        """Store provider model metadata for later instant API responses."""
        clean_infos = {
            info.model_id: info for info in model_infos if info.model_id.strip()
        }
        self._model_infos_by_provider[provider_id] = clean_infos
        self._model_ids_by_provider[provider_id] = frozenset(clean_infos)

    def cached_model_ids(self) -> dict[str, frozenset[str]]:
        """Return a copy of cached raw provider model ids."""
        return dict(self._model_ids_by_provider)

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

    async def refresh_model_list_cache(
        self, settings: Settings, *, only_missing: bool = False
    ) -> None:
        """Best-effort refresh of model lists for providers usable in this process."""
        provider_ids = _model_list_provider_ids_for_settings(settings)
        if only_missing:
            provider_ids = tuple(
                provider_id
                for provider_id in provider_ids
                if provider_id not in self._model_ids_by_provider
            )
        await self._refresh_model_ids(settings, provider_ids)

    def start_model_list_refresh(self, settings: Settings) -> None:
        """Start a non-blocking cache warmup for missing eligible provider lists."""
        if (
            self._model_list_refresh_task is not None
            and not self._model_list_refresh_task.done()
        ):
            return

        provider_ids = tuple(
            provider_id
            for provider_id in _model_list_provider_ids_for_settings(settings)
            if provider_id not in self._model_ids_by_provider
        )
        if not provider_ids:
            logger.info(
                "Provider model discovery cache already warm: providers={}",
                len(self._model_ids_by_provider),
            )
            return

        self._model_list_refresh_task = asyncio.create_task(
            self._run_model_list_refresh(settings, provider_ids)
        )

    async def _run_model_list_refresh(
        self, settings: Settings, provider_ids: tuple[str, ...]
    ) -> None:
        try:
            await self._refresh_model_ids(settings, provider_ids)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Provider model discovery task failed: exc_type={}",
                type(exc).__name__,
            )

    async def _refresh_model_ids(
        self, settings: Settings, provider_ids: tuple[str, ...]
    ) -> None:
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id in provider_ids:
            try:
                provider = self.get(provider_id, settings)
            except Exception as exc:
                _log_model_discovery_failure(provider_id, exc, settings)
                continue
            tasks[provider_id] = asyncio.create_task(provider.list_model_infos())

        if not tasks:
            return

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for (provider_id, _task), result in zip(tasks.items(), results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                _log_model_discovery_failure(provider_id, result, settings)
                continue
            self.cache_model_infos(provider_id, result)
            logger.info(
                "Provider model discovery cached: provider={} models={}",
                provider_id,
                len(result),
            )

    async def validate_configured_models(self, settings: Settings) -> None:
        """Fail fast unless every configured chat model exists upstream."""
        refs = settings.configured_chat_model_refs()
        refs_by_provider: dict[str, list[ConfiguredChatModelRef]] = defaultdict(list)
        for ref in refs:
            refs_by_provider[ref.provider_id].append(ref)

        failures: list[str] = []
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id, provider_refs in refs_by_provider.items():
            try:
                provider = self.get(provider_id, settings)
            except Exception as exc:
                failures.extend(
                    _format_provider_query_failures(provider_refs, exc, settings)
                )
                continue
            tasks[provider_id] = asyncio.create_task(provider.list_model_infos())

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (provider_id, _task), result in zip(
                tasks.items(), results, strict=True
            ):
                provider_refs = refs_by_provider[provider_id]
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    failures.extend(
                        _format_provider_query_failures(provider_refs, result, settings)
                    )
                    continue
                self.cache_model_infos(provider_id, result)
                model_ids = self._model_ids_by_provider[provider_id]
                failures.extend(
                    _format_missing_model_failure(ref)
                    for ref in provider_refs
                    if ref.model_id not in model_ids
                )

        if failures:
            message = "Configured model validation failed:\n" + "\n".join(
                f"- {failure}" for failure in failures
            )
            raise ServiceUnavailableError(message)

        logger.info(
            "Configured provider models validated: models={} providers={}",
            len(refs),
            len(refs_by_provider),
        )

    async def cleanup(self) -> None:
        """Call ``cleanup`` on every cached provider, then clear the cache.

        Attempts all providers even if one fails. A single failure is re-raised
        as-is; multiple failures are wrapped in :exc:`ExceptionGroup`.
        """
        if (
            self._model_list_refresh_task is not None
            and not self._model_list_refresh_task.done()
        ):
            self._model_list_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._model_list_refresh_task

        items = list(self._providers.items())
        errors: list[Exception] = []
        try:
            for _pid, provider in items:
                try:
                    await provider.cleanup()
                except Exception as e:
                    errors.append(e)
        finally:
            self._providers.clear()
            self._model_ids_by_provider.clear()
            self._model_infos_by_provider.clear()
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            msg = "One or more provider cleanups failed"
            raise ExceptionGroup(msg, errors)

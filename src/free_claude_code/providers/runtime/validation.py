"""Configured provider model validation."""

import asyncio
from collections import defaultdict
from collections.abc import Callable

import httpx
from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.model_refs import (
    ConfiguredChatModelRef,
    configured_chat_model_refs,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.model_listing import ModelListResponseError

from .model_cache import ProviderModelCache

ProviderResolver = Callable[[str], BaseProvider]


def provider_query_failure_reason(exc: BaseException, settings: Settings) -> str:
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


class ConfiguredModelValidator:
    """Validate configured provider/model refs against upstream model lists."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        model_cache: ProviderModelCache,
    ) -> None:
        self._settings = settings
        self._provider_resolver = provider_resolver
        self._model_cache = model_cache

    async def validate_configured_models(self) -> None:
        """Fail unless every configured chat model exists upstream."""
        refs = configured_chat_model_refs(self._settings)
        refs_by_provider: dict[str, list[ConfiguredChatModelRef]] = defaultdict(list)
        for ref in refs:
            refs_by_provider[ref.provider_id].append(ref)

        failures: list[str] = []
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id, provider_refs in refs_by_provider.items():
            try:
                provider = self._provider_resolver(provider_id)
            except Exception as exc:
                failures.extend(
                    self._format_provider_query_failures(provider_refs, exc)
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
                        self._format_provider_query_failures(provider_refs, result)
                    )
                    continue
                model_ids = frozenset(info.model_id for info in result)
                self._model_cache.cache_model_infos(provider_id, result)
                failures.extend(
                    self._format_missing_model_failure(ref)
                    for ref in provider_refs
                    if ref.model_id not in model_ids
                )

        if failures:
            message = "Configured model validation failed:\n" + "\n".join(
                f"- {failure}" for failure in failures
            )
            raise ApplicationUnavailableError(message)

        logger.info(
            "Configured provider models validated: models={} providers={}",
            len(refs),
            len(refs_by_provider),
        )

    def _format_provider_query_failures(
        self,
        refs: list[ConfiguredChatModelRef],
        exc: BaseException,
    ) -> list[str]:
        reason = provider_query_failure_reason(exc, self._settings)
        return [self._format_model_validation_failure(ref, reason) for ref in refs]

    def _format_missing_model_failure(self, ref: ConfiguredChatModelRef) -> str:
        return self._format_model_validation_failure(ref, "missing model")

    @staticmethod
    def _format_model_validation_failure(
        ref: ConfiguredChatModelRef, problem: str
    ) -> str:
        return (
            f"sources={','.join(ref.sources)} provider={ref.provider_id} "
            f"model={ref.model_id} problem={problem}"
        )

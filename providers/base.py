"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

from config.constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from providers.key_rotation import ApiKeyRotationMode, ApiKeyRotationPool
from providers.model_listing import ProviderModelInfo, model_infos_from_ids


class ProviderConfig(BaseModel):
    """Configuration for a provider.

    Base fields apply to all providers. Provider-specific parameters
    (e.g. NIM temperature, top_p) are passed by the provider constructor.
    """

    api_key: str
    api_key_rotation_mode: ApiKeyRotationMode = ApiKeyRotationMode.ROUND_ROBIN
    base_url: str | None = None
    rate_limit: int | None = None
    rate_window: int = 60
    max_concurrency: int = 5
    http_read_timeout: float = 300.0
    http_write_timeout: float = 10.0
    http_connect_timeout: float = HTTP_CONNECT_TIMEOUT_DEFAULT
    enable_thinking: bool = True
    proxy: str = ""
    log_raw_sse_events: bool = False
    log_api_error_tracebacks: bool = False


class BaseProvider(ABC):
    """Base class for all providers. Extend this to add your own."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._api_key_context: ContextVar[str | None] = ContextVar(
            f"{type(self).__name__}_api_key_context",
            default=None,
        )
        self._api_key_pool: ApiKeyRotationPool | None = None
        if config.api_key.strip():
            self._api_key_pool = ApiKeyRotationPool(
                config.api_key,
                config.api_key_rotation_mode,
            )

    def _key_for_new_request(self) -> str:
        if self._api_key_pool is None:
            return self._config.api_key
        return self._api_key_pool.key_for_new_request()

    def _next_key_after_limit(self, current_key: str) -> str | None:
        if self._api_key_pool is None:
            return None
        if self._api_key_pool.mode != ApiKeyRotationMode.FAILOVER_ON_LIMIT:
            return None
        return self._api_key_pool.next_key_after_limit(current_key)

    def _uses_api_key_failover(self) -> bool:
        return (
            self._api_key_pool is not None
            and self._api_key_pool.mode == ApiKeyRotationMode.FAILOVER_ON_LIMIT
        )

    def _current_api_key(self) -> str:
        key = self._api_key_context.get()
        if key is not None:
            return key
        return self._key_for_new_request()

    @contextmanager
    def _using_api_key(self, key: str):
        token = self._api_key_context.set(key)
        try:
            yield
        finally:
            self._api_key_context.reset(token)

    def _is_thinking_enabled(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> bool:
        """Return whether thinking should be enabled for this request."""
        thinking = getattr(request, "thinking", None)
        config_enabled = (
            self._config.enable_thinking
            if thinking_enabled is None
            else thinking_enabled
        )
        request_enabled = True
        if thinking is not None:
            thinking_type = (
                thinking.get("type")
                if isinstance(thinking, dict)
                else getattr(thinking, "type", None)
            )
            if thinking_type == "disabled":
                request_enabled = False

            enabled = (
                thinking.get("enabled")
                if isinstance(thinking, dict)
                else getattr(thinking, "enabled", None)
            )
            if enabled is not None:
                request_enabled = bool(enabled)
        return config_enabled and request_enabled

    def preflight_stream(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> None:
        """Eagerly validate/build the upstream request before opening an SSE stream.

        Subclasses with ``_build_request_body`` (OpenAI and native) raise
        :class:`providers.exceptions.InvalidRequestError` on conversion failures.
        """
        build = getattr(self, "_build_request_body", None)
        if build is None:
            return
        build(request, thinking_enabled=thinking_enabled)

    def _log_stream_transport_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> None:
        """Log streaming transport failures (metadata-only unless verbose is enabled)."""
        from loguru import logger

        from core.trace import trace_event

        response = getattr(error, "response", None)
        http_status = (
            getattr(response, "status_code", None) if response is not None else None
        )
        trace_event(
            stage="provider",
            event="provider.response.transport_error",
            source="provider",
            provider=tag,
            request_id=request_id,
            exc_type=type(error).__name__,
            http_status=http_status,
        )

        if self._config.log_api_error_tracebacks:
            logger.error(
                "{}_ERROR:{} {}: {}", tag, req_tag, type(error).__name__, error
            )
            return
        logger.error(
            "{}_ERROR:{} exc_type={} http_status={}",
            tag,
            req_tag,
            type(error).__name__,
            http_status,
        )

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources held by this provider."""

    @abstractmethod
    async def list_model_ids(self) -> frozenset[str]:
        """Return the model ids currently advertised by this provider."""

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return advertised model ids with optional provider capability metadata."""
        return model_infos_from_ids(await self.list_model_ids())

    @abstractmethod
    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        # Typing: abstract async generators need a yield for AsyncIterator[str]
        # inference; this branch is never executed.
        if False:
            yield ""

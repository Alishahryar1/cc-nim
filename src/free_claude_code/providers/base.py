"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from loguru import logger
from pydantic import BaseModel

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    redacted_exception_traceback,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.model_listing import model_infos_from_ids


class ProviderConfig(BaseModel):
    """Configuration for a provider.

    Base fields apply to all providers. Provider-specific parameters
    (e.g. NIM temperature, top_p) are passed by the provider constructor.
    """

    api_key: str
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

    def _is_thinking_enabled(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> bool:
        """Return whether thinking should be enabled for this request."""
        thinking = request.thinking
        config_enabled = (
            self._config.enable_thinking
            if thinking_enabled is None
            else thinking_enabled
        )
        request_enabled = True
        if thinking is not None:
            if "enabled" in thinking.model_fields_set and thinking.enabled is not None:
                request_enabled = thinking.enabled
            if thinking.type == "disabled":
                request_enabled = False
        return config_enabled and request_enabled

    @abstractmethod
    def preflight_stream(
        self, request: MessagesRequest, *, thinking_enabled: bool | None = None
    ) -> None:
        """Validate the upstream request before opening an SSE stream."""

    def _log_stream_transport_error(
        self,
        tag: str,
        req_tag: str,
        error: Exception,
        *,
        request_id: str | None = None,
    ) -> None:
        """Log streaming transport failures (metadata-only unless verbose is enabled)."""
        response = getattr(error, "response", None)
        http_status = (
            getattr(response, "status_code", None) if response is not None else None
        )
        cause_types = exception_cause_types(error)
        trace_event(
            stage="provider",
            event="provider.response.transport_error",
            source="provider",
            provider=tag,
            request_id=request_id,
            exc_type=type(error).__name__,
            http_status=http_status,
            cause_types=cause_types,
        )

        if self._config.log_api_error_tracebacks:
            logger.error(
                "{}_ERROR:{} exc_type={}\n{}",
                tag,
                req_tag,
                type(error).__name__,
                redacted_exception_traceback(error),
            )
            return
        logger.error(
            "{}_ERROR:{} exc_type={} http_status={} cause_types={}",
            tag,
            req_tag,
            type(error).__name__,
            http_status,
            ",".join(cause_types) if cause_types else None,
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
    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""

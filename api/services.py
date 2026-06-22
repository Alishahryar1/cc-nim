"""Application services for the Claude-compatible API."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import (
    get_token_count,
    get_user_facing_error_message,
    iter_provider_stream_error_sse_events,
)
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.exceptions import (
    InvalidRequestError,
    PreStreamProviderError,
    ProviderError,
)

from .model_router import ModelRouter, RoutedMessagesRequest
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim", "opencode", "zai"})


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)
            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.optimization.web_server_tool",
                    source="api",
                    model=routed.request.model,
                )
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                trace_event(
                    stage="routing",
                    event="api.optimization.short_circuit",
                    source="api",
                    model=routed.request.model,
                )
                return optimized
            logger.debug("No optimization matched, routing to provider")

            provider = self._provider_getter(routed.resolved.provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            trace_event(
                stage="routing",
                event="api.route.resolved",
                source="api",
                provider_id=routed.resolved.provider_id,
                provider_model=routed.resolved.provider_model,
                provider_model_ref=routed.resolved.provider_model_ref,
                gateway_model=routed.request.model,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            with logger.contextualize(request_id=request_id):
                trace_event(
                    stage="ingress",
                    event="api.request.received",
                    source="api",
                    message_count=len(routed.request.messages),
                    snapshot=api_messages_request_snapshot(routed.request),
                )

                if self._settings.log_raw_api_payloads:
                    logger.debug(
                        "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                    )

                input_tokens = self._token_counter(
                    routed.request.messages,
                    routed.request.system,
                    routed.request.tools,
                )

                # Build the primary stream eagerly so any synchronous
                # provider-construction error surfaces inside this try block
                # (→ HTTPException) instead of mid-stream. Real providers are
                # async generators (construction never raises); the retry
                # wrapper handles in-stream pre-stream failures and empty
                # completions.
                primary_stream = provider.stream_response(
                    routed.request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=routed.resolved.thinking_enabled,
                    raise_on_prestream_error=True,
                )
                base_stream = self._stream_with_failover(
                    routed,
                    primary_stream,
                    request_id=request_id,
                    input_tokens=input_tokens,
                )

                streamed = traced_async_stream(
                    base_stream,
                    stage="egress",
                    source="api",
                    complete_event="api.response.stream_completed",
                    interrupted_event="api.response.stream_interrupted",
                    chunk_event=None,
                    extra={
                        "request_id": request_id,
                        "provider_id": routed.resolved.provider_id,
                        "gateway_model": routed.request.model,
                    },
                )
                return anthropic_sse_streaming_response(streamed)

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    async def _stream_with_failover(
        self,
        routed: RoutedMessagesRequest,
        primary_stream: AsyncIterator[str],
        *,
        request_id: str,
        input_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream the primary model; on a pre-stream failure transparently retry
        the turn once, then guarantee a well-formed stream to the client.

        ``primary_stream`` is the already-constructed primary attempt (built by
        the caller so synchronous construction errors surface there).

        A "pre-stream failure" is any error — or an intermittent empty 200
        completion (no content blocks) — that occurs before a single content
        block reaches the client, surfaced by the provider as
        :class:`PreStreamProviderError` (only when ``raise_on_prestream_error``
        is set). Because nothing was emitted yet, the turn is safe to retry:

        * on the configured tier fallback model when one is set, otherwise
        * on the SAME model (empty completions are transient; a fresh attempt
          almost always succeeds).

        If the retry ALSO fails pre-stream, a clean Anthropic error stream is
        emitted so the client never receives an empty or malformed body.
        """
        try:
            async for chunk in primary_stream:
                yield chunk
            return
        except PreStreamProviderError as exc:
            if (
                routed.fallback_resolved is not None
                and routed.fallback_request is not None
            ):
                retry_resolved = routed.fallback_resolved
                retry_request = routed.fallback_request
                same_model = False
            else:
                retry_resolved = routed.resolved
                retry_request = routed.request
                same_model = True
            trace_event(
                stage="routing",
                event="api.route.failover",
                source="api",
                request_id=request_id,
                primary_provider=routed.resolved.provider_id,
                primary_model=routed.resolved.provider_model,
                fallback_provider=retry_resolved.provider_id,
                fallback_model=retry_resolved.provider_model,
                same_model=same_model,
                reason=exc.message,
            )
            logger.warning(
                "FAILOVER request_id={}: primary {}/{} failed pre-stream ({}); "
                "retrying on {}/{}{}",
                request_id,
                routed.resolved.provider_id,
                routed.resolved.provider_model,
                exc.message,
                retry_resolved.provider_id,
                retry_resolved.provider_model,
                " (same model)" if same_model else "",
            )

        retry_provider = self._provider_getter(retry_resolved.provider_id)
        try:
            async for chunk in retry_provider.stream_response(
                retry_request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=retry_resolved.thinking_enabled,
                raise_on_prestream_error=True,
            ):
                yield chunk
        except PreStreamProviderError as exc2:
            logger.error(
                "FAILOVER request_id={}: retry on {}/{} also failed pre-stream "
                "({}); emitting clean error stream to client",
                request_id,
                retry_resolved.provider_id,
                retry_resolved.provider_model,
                exc2.message,
            )
            trace_event(
                stage="routing",
                event="api.route.failover_exhausted",
                source="api",
                request_id=request_id,
                retry_provider=retry_resolved.provider_id,
                retry_model=retry_resolved.provider_model,
                reason=exc2.message,
            )
            for event in iter_provider_stream_error_sse_events(
                request=retry_request,
                input_tokens=input_tokens,
                error_message=exc2.message,
                sent_any_event=False,
                log_raw_sse_events=self._settings.log_raw_sse_events,
            ):
                yield event

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.route.resolved",
                    source="api",
                    kind="count_tokens",
                    provider_id=routed.resolved.provider_id,
                    provider_model=routed.resolved.provider_model,
                    provider_model_ref=routed.resolved.provider_model_ref,
                    gateway_model=routed.request.model,
                )
                trace_event(
                    stage="ingress",
                    event="api.count_tokens.completed",
                    source="api",
                    message_count=len(routed.request.messages),
                    input_tokens=tokens,
                    snapshot=api_messages_request_snapshot(routed.request),
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e

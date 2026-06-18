"""Application services for the Claude-compatible API."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from core.openai_responses import OpenAIResponsesAdapter
from core.trace import api_messages_request_snapshot, trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.openai_responses import OpenAIResponsesRequest
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
_OPENAI_CHAT_UPSTREAM_IDS = frozenset(
    provider_id
    for provider_id, descriptor in PROVIDER_CATALOG.items()
    if descriptor.transport_type == "openai_chat"
)


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def openai_responses_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a streaming response for OpenAI Responses-style SSE."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=OpenAIResponsesAdapter.sse_headers,
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
        responses_adapter: OpenAIResponsesAdapter | None = None,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._responses_adapter = responses_adapter or OpenAIResponsesAdapter()

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

                streamed = traced_async_stream(
                    provider.stream_response(
                        routed.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=routed.resolved.thinking_enabled,
                    ),
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

    async def create_response(self, request_data: OpenAIResponsesRequest) -> object:
        """Create an OpenAI Responses-compatible response through the provider router."""

        request_payload = request_data.model_dump(mode="json", exclude_none=True)
        if request_data.stream is False:
            invalid_request = InvalidRequestError(
                "FCC /v1/responses supports streaming only; omit stream or set stream=true."
            )
            return JSONResponse(
                status_code=invalid_request.status_code,
                content=self._responses_adapter.error_payload(
                    message=invalid_request.message,
                    error_type=invalid_request.error_type,
                ),
            )

        try:
            anthropic_payload = self._responses_adapter.to_anthropic_payload(
                request_payload
            )
            response_request = MessagesRequest(**anthropic_payload)
            _require_non_empty_messages(response_request.messages)
            routed = self._model_router.resolve_messages_request(response_request)

            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

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
                wire_api="responses",
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            with logger.contextualize(request_id=request_id):
                trace_event(
                    stage="ingress",
                    event="api.responses.request.received",
                    source="api",
                    message_count=len(routed.request.messages),
                    snapshot=api_messages_request_snapshot(routed.request),
                )

                if self._settings.log_raw_api_payloads:
                    logger.debug(
                        "FULL_RESPONSES_PAYLOAD [{}]: {}",
                        request_id,
                        request_payload,
                    )

                input_tokens = self._token_counter(
                    routed.request.messages,
                    routed.request.system,
                    routed.request.tools,
                )

                streamed = traced_async_stream(
                    provider.stream_response(
                        routed.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=routed.resolved.thinking_enabled,
                    ),
                    stage="egress",
                    source="api",
                    complete_event="api.responses.stream_completed",
                    interrupted_event="api.responses.stream_interrupted",
                    chunk_event=None,
                    extra={
                        "request_id": request_id,
                        "provider_id": routed.resolved.provider_id,
                        "gateway_model": routed.request.model,
                    },
                )
                return openai_responses_sse_streaming_response(
                    self._responses_adapter.iter_sse_from_anthropic(
                        streamed,
                        request_payload,
                    )
                )
        except OpenAIResponsesAdapter.ConversionError as exc:
            invalid_request = InvalidRequestError(str(exc))
            return JSONResponse(
                status_code=invalid_request.status_code,
                content=self._responses_adapter.error_payload(
                    message=invalid_request.message,
                    error_type=invalid_request.error_type,
                ),
            )
        except ProviderError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=self._responses_adapter.error_payload(
                    message=exc.message,
                    error_type=exc.error_type,
                ),
            )
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings,
                e,
                context="CREATE_RESPONSE_ERROR",
            )
            return JSONResponse(
                status_code=_http_status_for_unexpected_service_exception(e),
                content=self._responses_adapter.error_payload(
                    message=get_user_facing_error_message(e),
                    error_type="api_error",
                ),
            )

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

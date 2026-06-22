"""API request pipeline for routing, intercepts, and provider execution."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
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

from .detection import is_safety_classifier_request
from .model_router import ModelRouter, RoutedMessagesRequest
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
MessageIntercept = Callable[[RoutedMessagesRequest], object | None]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion.
_OPENAI_CHAT_UPSTREAM_IDS = frozenset(
    provider_id
    for provider_id, descriptor in PROVIDER_CATALOG.items()
    if descriptor.transport_type == "openai_chat"
)


def anthropic_sse_streaming_response(body: AsyncIterator[str]) -> StreamingResponse:
    """Return a streaming response for Anthropic-style SSE streams."""
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


def _http_status_for_unexpected_pipeline_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures."""
    return 500


def _log_unexpected_pipeline_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log API failures without echoing exception text unless opted in."""
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


class ApiRequestPipeline:
    """Coordinate API request intercepts, routing, and provider stream execution."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        responses_adapter: OpenAIResponsesAdapter | None = None,
    ) -> None:
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._responses_adapter = responses_adapter or OpenAIResponsesAdapter()
        self._message_intercepts: tuple[MessageIntercept, ...] = (
            self._intercept_web_server_tool,
            self._intercept_local_optimization,
        )

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create an Anthropic-compatible message response."""
        try:
            _require_non_empty_messages(request_data.messages)
            routed = self._model_router.resolve_messages_request(request_data)
            routed = self._apply_message_routing_policies(routed)
            self._reject_unsupported_server_tools(routed)

            intercepted = self._run_message_intercepts(routed)
            if intercepted is not None:
                return intercepted

            logger.debug("No optimization matched, routing to provider")
            return anthropic_sse_streaming_response(
                self._provider_stream(
                    routed,
                    wire_api="messages",
                    raw_log_label="FULL_PAYLOAD",
                    raw_log_payload=routed.request.model_dump(),
                )
            )
        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_pipeline_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_pipeline_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    async def create_response(self, request_data: OpenAIResponsesRequest) -> object:
        """Create a streaming OpenAI Responses-compatible response."""
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
            self._reject_unsupported_server_tools(routed)

            streamed = self._provider_stream(
                routed,
                wire_api="responses",
                raw_log_label="FULL_RESPONSES_PAYLOAD",
                raw_log_payload=request_payload,
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
            _log_unexpected_pipeline_exception(
                self._settings,
                e,
                context="CREATE_RESPONSE_ERROR",
            )
            return JSONResponse(
                status_code=_http_status_for_unexpected_pipeline_exception(e),
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
                _log_unexpected_pipeline_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_pipeline_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e

    def _reject_unsupported_server_tools(self, routed: RoutedMessagesRequest) -> None:
        if routed.resolved.provider_id not in _OPENAI_CHAT_UPSTREAM_IDS:
            return
        tool_err = openai_chat_upstream_server_tool_error(
            routed.request,
            web_tools_enabled=self._settings.enable_web_server_tools,
        )
        if tool_err is not None:
            raise InvalidRequestError(tool_err)

    def _apply_message_routing_policies(
        self, routed: RoutedMessagesRequest
    ) -> RoutedMessagesRequest:
        if not is_safety_classifier_request(routed.request):
            return routed
        changed = routed.resolved.thinking_enabled
        trace_event(
            stage="routing",
            event="api.optimization.safety_classifier_no_thinking",
            source="api",
            model=routed.request.model,
            changed=changed,
        )
        if not changed:
            return routed
        return RoutedMessagesRequest(
            request=routed.request,
            resolved=replace(routed.resolved, thinking_enabled=False),
        )

    def _run_message_intercepts(self, routed: RoutedMessagesRequest) -> object | None:
        for intercept in self._message_intercepts:
            result = intercept(routed)
            if result is not None:
                return result
        return None

    def _intercept_web_server_tool(
        self, routed: RoutedMessagesRequest
    ) -> object | None:
        if not self._settings.enable_web_server_tools:
            return None
        if not is_web_server_tool_request(routed.request):
            return None

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

    def _intercept_local_optimization(
        self, routed: RoutedMessagesRequest
    ) -> object | None:
        optimized = try_optimizations(routed.request, self._settings)
        if optimized is None:
            return None
        trace_event(
            stage="routing",
            event="api.optimization.short_circuit",
            source="api",
            model=routed.request.model,
        )
        return optimized

    def _provider_stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: str,
        raw_log_label: str,
        raw_log_payload: Any,
    ) -> AsyncIterator[str]:
        provider = self._provider_getter(routed.resolved.provider_id)
        provider.preflight_stream(
            routed.request,
            thinking_enabled=routed.resolved.thinking_enabled,
        )

        route_trace: dict[str, Any] = {
            "stage": "routing",
            "event": "api.route.resolved",
            "source": "api",
            "provider_id": routed.resolved.provider_id,
            "provider_model": routed.resolved.provider_model,
            "provider_model_ref": routed.resolved.provider_model_ref,
            "gateway_model": routed.request.model,
            "thinking_enabled": routed.resolved.thinking_enabled,
        }
        if wire_api == "responses":
            route_trace["wire_api"] = "responses"
        trace_event(**route_trace)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        trace_event(
            stage="ingress",
            event=(
                "api.responses.request.received"
                if wire_api == "responses"
                else "api.request.received"
            ),
            source="api",
            message_count=len(routed.request.messages),
            snapshot=api_messages_request_snapshot(routed.request),
            request_id=request_id,
        )

        if self._settings.log_raw_api_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )
        return traced_async_stream(
            provider.stream_response(
                routed.request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=routed.resolved.thinking_enabled,
            ),
            stage="egress",
            source="api",
            complete_event=(
                "api.responses.stream_completed"
                if wire_api == "responses"
                else "api.response.stream_completed"
            ),
            interrupted_event=(
                "api.responses.stream_interrupted"
                if wire_api == "responses"
                else "api.response.stream_interrupted"
            ),
            chunk_event=None,
            extra={
                "request_id": request_id,
                "provider_id": routed.resolved.provider_id,
                "gateway_model": routed.request.model,
            },
        )

"""Application services for the Claude-compatible API."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from core.openai_responses import (
    OPENAI_RESPONSES_SSE_HEADERS,
    ResponsesConversionError,
    anthropic_message_response_to_openai_response,
    collect_openai_response_from_anthropic_sse,
    iter_anthropic_sse_as_openai_responses,
    iter_message_response_as_openai_responses,
    openai_error_payload,
    responses_request_to_anthropic_payload,
)
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
        headers=OPENAI_RESPONSES_SSE_HEADERS,
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


def _has_image_block(messages: list[Any]) -> bool:
    """Check if any message contains an image block."""
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        return True
                else:
                    if getattr(block, "type", None) == "image":
                        return True
        elif isinstance(content, dict) and content.get("type") == "image":
            return True
    return False


def _sanitize_old_images(messages: list[Any]) -> None:
    """Keep only the very last image block in the entire history to comply with 1-image limits of provider models."""
    image_count = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        image_count += 1
                else:
                    if getattr(block, "type", None) == "image":
                        image_count += 1
        elif isinstance(content, dict) and content.get("type") == "image":
            image_count += 1

    if image_count <= 1:
        return

    images_seen = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            new_content = []
            for block in content:
                is_image = False
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        is_image = True
                else:
                    if getattr(block, "type", None) == "image":
                        is_image = True

                if is_image:
                    images_seen += 1
                    if images_seen < image_count:
                        from api.models.anthropic import ContentBlockText

                        new_content.append(
                            ContentBlockText(
                                type="text",
                                text="[Prior image block removed to comply with 1-image limit]",
                            )
                        )
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            msg.content = new_content
        elif isinstance(content, dict) and content.get("type") == "image":
            images_seen += 1
            if images_seen < image_count:
                from api.models.anthropic import ContentBlockText

                msg.content = ContentBlockText(
                    type="text",
                    text="[Prior image block removed to comply with 1-image limit]",
                )


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

            if _has_image_block(request_data.messages):
                provider_model_lower = routed.resolved.provider_model.lower()
                vision_keywords = (
                    "vision",
                    "vl",
                    "gemini",
                    "kimi",
                    "omni",
                    "claude-3",
                    "gpt-4",
                    "pixtral",
                    "clip",
                    "siglip",
                )
                is_vision_capable = any(
                    kw in provider_model_lower for kw in vision_keywords
                )
                if not is_vision_capable and self._settings.model_fallback_vision:
                    fallback_ref = self._settings.model_fallback_vision
                    fallback_provider = Settings.parse_provider_type(fallback_ref)
                    fallback_model = Settings.parse_model_name(fallback_ref)
                    fallback_thinking = self._settings.resolve_thinking(fallback_model)

                    logger.info(
                        "Image detected in request but model '{}' is not vision-capable. "
                        "Dynamically falling back to vision model: {}",
                        routed.resolved.provider_model,
                        fallback_ref,
                    )

                    # Re-route the request
                    routed.request.model = fallback_model
                    _sanitize_old_images(routed.request.messages)
                    from .model_router import ResolvedModel, RoutedMessagesRequest

                    new_resolved = ResolvedModel(
                        original_model=routed.resolved.original_model,
                        provider_id=fallback_provider,
                        provider_model=fallback_model,
                        provider_model_ref=fallback_ref,
                        thinking_enabled=fallback_thinking,
                    )
                    routed = RoutedMessagesRequest(
                        request=routed.request, resolved=new_resolved
                    )

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
        try:
            anthropic_payload = responses_request_to_anthropic_payload(request_payload)
            result = self.create_message(MessagesRequest(**anthropic_payload))
        except ResponsesConversionError as exc:
            invalid_request = InvalidRequestError(str(exc))
            return JSONResponse(
                status_code=invalid_request.status_code,
                content=openai_error_payload(
                    message=invalid_request.message,
                    error_type=invalid_request.error_type,
                ),
            )
        except ProviderError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=openai_error_payload(
                    message=exc.message,
                    error_type=exc.error_type,
                ),
            )

        if request_data.stream is False:
            if isinstance(result, StreamingResponse):
                return await collect_openai_response_from_anthropic_sse(
                    result.body_iterator,
                    request_payload,
                )
            return anthropic_message_response_to_openai_response(
                _model_dump_json(result),
                request_payload,
            )

        if isinstance(result, StreamingResponse):
            return openai_responses_sse_streaming_response(
                iter_anthropic_sse_as_openai_responses(
                    result.body_iterator,
                    request_payload,
                )
            )

        return openai_responses_sse_streaming_response(
            _iter_static_response_sse(
                iter_message_response_as_openai_responses(
                    _model_dump_json(result),
                    request_payload,
                )
            )
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


def _model_dump_json(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


async def _iter_static_response_sse(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk

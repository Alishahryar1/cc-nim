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
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, SystemContent, TokenCountRequest
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
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim"})

_AGENTIC_DIRECTIVE = (
    "\n\n<agentic_behavior_directive>"
    "\nYou are an autonomous agent executing multi-step tasks. Rules:"
    "\n1. NEVER end your turn mid-task. Complete ALL steps before stopping."
    "\n2. NEVER use run_in_background for tasks whose results you need — run them"
    " synchronously."
    "\n3. After each tool result, immediately determine and execute the next step."
    "\n4. Chain as many tool calls as needed within a single turn."
    "\n5. Only stop when the entire task is fully complete and verified."
    "\n</agentic_behavior_directive>"
)


def _message_starts_with_tool_result(m: Any) -> bool:
    """True if the message's first content block is a tool_result.

    A user message that opens with tool_result blocks is a follow-up to an
    assistant tool_use turn.  If that assistant turn was trimmed away, the
    tool_result blocks are orphaned and DeepSeek rejects the request with:
      "Each tool_result block must have a corresponding tool_use block in the
       previous message."
    """
    content = getattr(m, "content", None)
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return getattr(first, "type", None) == "tool_result"


def _truncate_messages(request: MessagesRequest, max_messages: int) -> MessagesRequest:
    """Trim the conversation history to the most recent ``max_messages`` messages.

    Takes the last ``max_messages`` messages, then advances the start pointer to
    the first ``user`` message whose first block is NOT a tool_result.  This
    prevents orphaned tool_result blocks (whose paired assistant tool_use was
    trimmed) from appearing at position 0, which DeepSeek rejects with HTTP 400.
    """
    msgs = request.messages
    if len(msgs) <= max_messages:
        return request

    kept = msgs[-max_messages:]
    # Find the first clean user message (not a tool_result continuation).
    for i, m in enumerate(kept):
        if m.role == "user" and not _message_starts_with_tool_result(m):
            kept = kept[i:]
            break

    if not kept:
        kept = msgs[-1:]

    original = len(msgs)
    logger.info(
        "CONTEXT_TRIM: truncated %d → %d messages (cap=%d)",
        original,
        len(kept),
        max_messages,
    )
    return request.model_copy(update={"messages": kept})


def _inject_agentic_directive(request: MessagesRequest) -> MessagesRequest:
    """Append the agentic directive to the system prompt (only for tool-bearing requests)."""
    if not request.tools:
        return request
    system = request.system
    if system is None:
        new_system: str | list[SystemContent] = _AGENTIC_DIRECTIVE.strip()
    elif isinstance(system, str):
        new_system = system + _AGENTIC_DIRECTIVE
    else:
        new_system = list(system) + [
            SystemContent(type="text", text=_AGENTIC_DIRECTIVE.strip())
        ]
    return request.model_copy(update={"system": new_system})


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

            if (
                self._settings.max_request_messages is not None
                and len(request_data.messages) > self._settings.max_request_messages
            ):
                request_data = _truncate_messages(
                    request_data, self._settings.max_request_messages
                )

            if self._settings.inject_agentic_directive:
                request_data = _inject_agentic_directive(request_data)

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
                logger.info("Optimization: Handling Anthropic web server tool")
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
                return optimized
            logger.debug("No optimization matched, routing to provider")

            provider = self._provider_getter(routed.resolved.provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            logger.info(
                "API_REQUEST: request_id={} model={} messages={}",
                request_id,
                routed.request.model,
                len(routed.request.messages),
            )
            if self._settings.log_raw_api_payloads:
                logger.debug(
                    "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                )

            input_tokens = self._token_counter(
                routed.request.messages, routed.request.system, routed.request.tools
            )
            return anthropic_sse_streaming_response(
                provider.stream_response(
                    routed.request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=routed.resolved.thinking_enabled,
                ),
            )

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
                logger.info(
                    "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                    request_id,
                    routed.request.model,
                    len(routed.request.messages),
                    tokens,
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

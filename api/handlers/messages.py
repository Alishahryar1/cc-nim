"""Claude Messages API product flow."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace

from fastapi.responses import JSONResponse
from loguru import logger

from api.detection import is_safety_classifier_request
from api.model_router import ModelRouter, RoutedMessagesRequest
from api.models.anthropic import MessagesRequest
from api.optimization_handlers import try_optimizations
from api.provider_execution import ProviderExecutionService, TokenCounter
from api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
    unexpected_http_exception,
)
from api.response_streams import (
    EmptyStreamError,
    anthropic_sse_streaming_response,
)
from api.web_tools.egress import WebFetchEgressPolicy, web_fetch_allowed_scheme_set
from api.web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from api.web_tools.streaming import stream_web_server_tool_response
from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from core.anthropic import (
    aggregate_anthropic_sse_to_message,
    get_token_count,
    get_user_facing_error_message,
)
from core.trace import trace_event
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

_OPENAI_CHAT_UPSTREAM_IDS = frozenset(
    provider_id
    for provider_id, descriptor in PROVIDER_CATALOG.items()
    if descriptor.transport_type == "openai_chat"
)


@dataclass(frozen=True)
class _MessagesStreamResult:
    body: AsyncIterator[str]


@dataclass(frozen=True)
class _MessagesCompleteResult:
    response: object


ProviderGetter = Callable[[str], BaseProvider]
_MessagesResult = _MessagesStreamResult | _MessagesCompleteResult
MessageIntercept = Callable[[RoutedMessagesRequest], _MessagesResult | None]


def _unexpected_stream_error_message(exc: BaseException) -> str:
    if isinstance(exc, Exception):
        return get_user_facing_error_message(exc)
    return str(exc).strip() or f"{type(exc).__name__} occurred."


class MessagesHandler:
    """Handle Anthropic-compatible Messages requests."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        *,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
        provider_execution: ProviderExecutionService | None = None,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        self._provider_execution = provider_execution or ProviderExecutionService(
            settings,
            provider_getter,
            token_counter=token_counter,
        )
        self._message_intercepts: tuple[MessageIntercept, ...] = (
            self._intercept_web_server_tool,
            self._intercept_local_optimization,
        )

    async def create(self, request_data: MessagesRequest) -> object:
        """Create an Anthropic-compatible message response."""
        try:
            require_non_empty_messages(request_data.messages)
            routed = self._model_router.resolve_messages_request(request_data)
            routed = self._apply_message_routing_policies(routed)
            self._reject_unsupported_server_tools(routed)

            result = self._run_message_intercepts(routed)
            if result is None:
                logger.debug("No optimization matched, routing to provider")
                result = _MessagesStreamResult(
                    self._provider_execution.stream(
                        routed,
                        wire_api="messages",
                        raw_log_label="FULL_PAYLOAD",
                        raw_log_payload=routed.request.model_dump(),
                    )
                )
            return await self._to_public_response(result, stream=request_data.stream)
        except ProviderError:
            raise
        except Exception as exc:
            raise unexpected_http_exception(
                self._settings, exc, context="CREATE_MESSAGE_ERROR"
            ) from exc

    async def _to_public_response(
        self, result: _MessagesResult, *, stream: bool | None
    ) -> object:
        if isinstance(result, _MessagesCompleteResult):
            return result.response
        if stream is False:
            # Non-streaming clients (e.g. Claude Code utility calls) need a
            # complete JSON Message; the internal pipeline is always SSE, so
            # serving that raw here breaks the client SDK's response parse.
            message, error = await aggregate_anthropic_sse_to_message(result.body)
            if error is not None and not message.get("content"):
                return JSONResponse(
                    status_code=502,
                    content={"type": "error", "error": error},
                )
            return JSONResponse(content=message)
        return await anthropic_sse_streaming_response(
            result.body,
            pre_start_error_response=self._pre_start_error_response,
        )

    def _pre_start_error_response(self, exc: BaseException) -> JSONResponse:
        if isinstance(exc, ProviderError):
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_anthropic_format(),
            )
        log_unexpected_api_exception(
            self._settings,
            exc,
            context=(
                "CREATE_MESSAGE_EMPTY_STREAM"
                if isinstance(exc, EmptyStreamError)
                else "CREATE_MESSAGE_STREAM_START_ERROR"
            ),
        )
        return JSONResponse(
            status_code=http_status_for_unexpected_api_exception(exc),
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": _unexpected_stream_error_message(exc),
                },
            },
        )

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

    def _run_message_intercepts(
        self, routed: RoutedMessagesRequest
    ) -> _MessagesResult | None:
        for intercept in self._message_intercepts:
            result = intercept(routed)
            if result is not None:
                return result
        return None

    def _intercept_web_server_tool(
        self, routed: RoutedMessagesRequest
    ) -> _MessagesResult | None:
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
            allowed_schemes=web_fetch_allowed_scheme_set(
                self._settings.web_fetch_allowed_schemes
            ),
        )
        return _MessagesStreamResult(
            stream_web_server_tool_response(
                routed.request,
                input_tokens=input_tokens,
                web_fetch_egress=egress,
                verbose_client_errors=self._settings.log_api_error_tracebacks,
            ),
        )

    def _intercept_local_optimization(
        self, routed: RoutedMessagesRequest
    ) -> _MessagesResult | None:
        optimized = try_optimizations(routed.request, self._settings)
        if optimized is None:
            return None
        trace_event(
            stage="routing",
            event="api.optimization.short_circuit",
            source="api",
            model=routed.request.model,
        )
        return _MessagesCompleteResult(optimized)

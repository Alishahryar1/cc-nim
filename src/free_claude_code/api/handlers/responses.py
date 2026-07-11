"""OpenAI Responses API product flow for Codex clients."""

from collections.abc import Callable

from fastapi.responses import JSONResponse

from free_claude_code.api.model_router import ModelRouter
from free_claude_code.api.provider_execution import ProviderExecutionService
from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import (
    openai_responses_sse_streaming_response,
    terminal_execution_error_response,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    MessagesRequest,
    get_user_facing_error_message,
)
from free_claude_code.core.openai_responses import (
    OpenAIResponsesAdapter,
    OpenAIResponsesRequest,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.exceptions import InvalidRequestError, ProviderError

ProviderGetter = Callable[[str], BaseProvider]


class ResponsesHandler:
    """Handle streaming OpenAI Responses-compatible requests."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        *,
        model_router: ModelRouter | None = None,
        responses_adapter: OpenAIResponsesAdapter | None = None,
        provider_execution: ProviderExecutionService | None = None,
        generation_id: int | None = None,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._responses_adapter = responses_adapter or OpenAIResponsesAdapter()
        self._provider_execution = provider_execution or ProviderExecutionService(
            settings,
            provider_getter,
            generation_id=generation_id,
        )

    async def create(
        self, request_data: OpenAIResponsesRequest, *, request_id: str | None = None
    ) -> object:
        """Create a streaming OpenAI Responses-compatible response."""
        request_id = request_id or new_request_id()
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
                request_data
            )
            response_request = MessagesRequest(**anthropic_payload)
            require_non_empty_messages(response_request.messages)
            routed = self._model_router.resolve_messages_request(response_request)

            streamed = self._provider_execution.stream(
                routed,
                wire_api="responses",
                raw_log_label="FULL_RESPONSES_PAYLOAD",
                raw_log_payload=request_payload,
                request_id=request_id,
            )
            return await openai_responses_sse_streaming_response(
                self._responses_adapter.iter_sse_from_anthropic(
                    streamed,
                    request_data,
                ),
                headers=self._responses_adapter.sse_headers,
                pre_start_error_response=lambda exc: self._pre_start_error_response(
                    exc, request_id=request_id
                ),
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
        except Exception as exc:
            log_unexpected_api_exception(
                self._settings,
                exc,
                context="CREATE_RESPONSE_ERROR",
            )
            return JSONResponse(
                status_code=http_status_for_unexpected_api_exception(exc),
                content=self._responses_adapter.error_payload(
                    message=get_user_facing_error_message(exc),
                    error_type="api_error",
                ),
            )

    def _pre_start_error_response(
        self, exc: BaseException, *, request_id: str
    ) -> JSONResponse:
        if isinstance(exc, ProviderError):
            self._trace_terminal_execution_error(
                request_id=request_id,
                status_code=exc.status_code,
                error_type=exc.error_type,
            )
            return terminal_execution_error_response(
                status_code=exc.status_code,
                content=self._responses_adapter.error_payload(
                    message=exc.message,
                    error_type=exc.error_type,
                ),
            )
        log_unexpected_api_exception(
            self._settings,
            exc,
            context="CREATE_RESPONSE_STREAM_START_ERROR",
            request_id=request_id,
        )
        status_code = http_status_for_unexpected_api_exception(exc)
        self._trace_terminal_execution_error(
            request_id=request_id,
            status_code=status_code,
            error_type="api_error",
            exc_type=type(exc).__name__,
        )
        return terminal_execution_error_response(
            status_code=status_code,
            content=self._responses_adapter.error_payload(
                message=get_user_facing_error_message(exc),
                error_type="api_error",
            ),
        )

    @staticmethod
    def _trace_terminal_execution_error(
        *,
        request_id: str,
        status_code: int,
        error_type: str,
        exc_type: str | None = None,
    ) -> None:
        fields: dict[str, object] = {
            "wire_api": "responses",
            "request_id": request_id,
            "status_code": status_code,
            "error_type": error_type,
            "client_should_retry": False,
        }
        if exc_type is not None:
            fields["exc_type"] = exc_type
        trace_event(
            stage="egress",
            event="free_claude_code.api.response.terminal_execution_error",
            source="api",
            **fields,
        )

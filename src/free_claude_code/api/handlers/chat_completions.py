"""OpenAI Chat Completions API product flow."""

import asyncio
from typing import Any

from fastapi.responses import JSONResponse, Response

from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import (
    openai_chat_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import ProviderResolver
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    MessagesRequest,
    aggregate_anthropic_sse_to_message,
    anthropic_status_for_error_type,
)
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
from free_claude_code.core.openai_chat import (
    ChatCompletionsConversionError,
    OpenAIChatAdapter,
    OpenAIChatCompletionsRequest,
)
from free_claude_code.core.openai_responses import openai_error_type_for_failure


class ChatCompletionsHandler:
    """Handle OpenAI Chat Completions-compatible requests (streaming and not)."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        model_router: ModelRouter | None = None,
        chat_adapter: OpenAIChatAdapter | None = None,
        provider_executor: ProviderExecutor | None = None,
        generation_id: int | None = None,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._chat_adapter = chat_adapter or OpenAIChatAdapter()
        self._provider_executor = provider_executor or ProviderExecutor(
            provider_resolver,
            generation_id=generation_id,
            log_raw_payloads=settings.log_raw_api_payloads,
        )

    async def create(
        self,
        request_data: OpenAIChatCompletionsRequest,
        *,
        request_id: str | None = None,
    ) -> object:
        """Create a Chat Completions response, streaming when ``stream`` is true."""
        request_id = request_id or new_request_id()
        request_payload = request_data.model_dump(mode="json", exclude_none=True)
        stream = bool(request_data.stream)
        try:
            anthropic_payload = self._chat_adapter.to_anthropic_payload(request_data)
            messages_request = MessagesRequest(**anthropic_payload)
            require_non_empty_messages(messages_request.messages)
            routed = self._model_router.resolve_messages_request(messages_request)

            streamed = self._provider_executor.stream(
                routed,
                wire_api="chat",
                raw_log_label="FULL_CHAT_PAYLOAD",
                raw_log_payload=request_payload,
                request_id=request_id,
            )
            if stream:
                return await openai_chat_sse_streaming_response(
                    self._chat_adapter.iter_sse_from_anthropic(
                        streamed,
                        request_data,
                        on_post_start_terminal_failure=lambda exc: (
                            self._trace_post_start_terminal_failure(
                                exc, request_id=request_id
                            )
                        ),
                    ),
                    headers=self._chat_adapter.sse_headers,
                    pre_start_error_response=lambda exc: self._pre_start_error_response(
                        exc, request_id=request_id
                    ),
                )
            return await self._collect_non_stream_response(
                streamed, request_data, request_id=request_id
            )
        except ChatCompletionsConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        except ApplicationError:
            raise
        except ExecutionFailure as exc:
            return self._execution_failure_response(exc, request_id=request_id)
        except Exception as exc:
            failure = find_execution_failure(exc)
            if failure is not None:
                return self._execution_failure_response(failure, request_id=request_id)
            return self._unexpected_execution_error_response(
                exc, request_id=request_id, context="CREATE_CHAT_COMPLETION_ERROR"
            )

    async def _collect_non_stream_response(
        self,
        streamed: Any,
        request_data: OpenAIChatCompletionsRequest,
        *,
        request_id: str,
    ) -> object:
        try:
            message, error = await aggregate_anthropic_sse_to_message(streamed)
        except GeneratorExit, asyncio.CancelledError:
            raise
        except ExecutionFailure as exc:
            return self._execution_failure_response(exc, request_id=request_id)
        except BaseExceptionGroup as exc:
            failure = find_execution_failure(exc)
            if failure is not None:
                return self._execution_failure_response(failure, request_id=request_id)
            return self._unexpected_execution_error_response(
                exc,
                request_id=request_id,
                context="CREATE_CHAT_COMPLETION_NON_STREAM_ERROR",
            )
        except Exception as exc:
            return self._unexpected_execution_error_response(
                exc,
                request_id=request_id,
                context="CREATE_CHAT_COMPLETION_NON_STREAM_ERROR",
            )
        if error is not None:
            return self._stream_error_response(error, request_id=request_id)
        completion = self._chat_adapter.message_to_completion(message, request_data)
        return JSONResponse(content=completion)

    def _pre_start_error_response(
        self, exc: BaseException, *, request_id: str
    ) -> Response:
        failure = find_execution_failure(exc)
        if failure is not None:
            return self._execution_failure_response(failure, request_id=request_id)
        return self._unexpected_execution_error_response(
            exc,
            request_id=request_id,
            context="CREATE_CHAT_COMPLETION_STREAM_START_ERROR",
        )

    def _execution_failure_response(
        self, failure: ExecutionFailure, *, request_id: str
    ) -> JSONResponse:
        error_type = openai_error_type_for_failure(failure)
        trace_terminal_execution_error(
            wire_api="chat",
            request_id=request_id,
            status_code=failure.status_code,
            error_type=error_type,
            error=failure,
        )
        return terminal_execution_error_response(
            status_code=failure.status_code,
            content=self._chat_adapter.error_payload(
                message=failure.message, error_type=error_type
            ),
        )

    def _stream_error_response(
        self, error: dict[str, Any], *, request_id: str
    ) -> JSONResponse:
        error_type, message_text = _stream_error_fields(error)
        status_code = anthropic_status_for_error_type(error_type)
        trace_terminal_execution_error(
            wire_api="chat",
            request_id=request_id,
            status_code=status_code,
            error_type=error_type,
        )
        return terminal_execution_error_response(
            status_code=status_code,
            content=self._chat_adapter.error_payload(
                message=message_text, error_type=error_type
            ),
        )

    def _unexpected_execution_error_response(
        self, exc: BaseException, *, request_id: str, context: str
    ) -> JSONResponse:
        log_unexpected_api_exception(
            self._settings, exc, context=context, request_id=request_id
        )
        status_code = http_status_for_unexpected_api_exception(exc)
        trace_terminal_execution_error(
            wire_api="chat",
            request_id=request_id,
            status_code=status_code,
            error_type="api_error",
            error=exc,
        )
        return terminal_execution_error_response(
            status_code=status_code,
            content=self._chat_adapter.error_payload(
                message=safe_exception_message(exc), error_type="api_error"
            ),
        )

    def _trace_post_start_terminal_failure(
        self, exc: BaseException, *, request_id: str
    ) -> None:
        failure = find_execution_failure(exc)
        trace_terminal_execution_error(
            wire_api="chat",
            request_id=request_id,
            status_code=failure.status_code if failure is not None else 500,
            error_type=(
                openai_error_type_for_failure(failure)
                if failure is not None
                else "api_error"
            ),
            error=exc,
        )


def _stream_error_fields(error: dict[str, Any]) -> tuple[str, str]:
    raw_type = error.get("type")
    error_type = (
        raw_type.strip()
        if isinstance(raw_type, str) and raw_type.strip()
        else "api_error"
    )
    raw_message = error.get("message")
    message = (
        raw_message.strip()
        if isinstance(raw_message, str) and raw_message.strip()
        else "Provider request failed unexpectedly."
    )
    return error_type, message

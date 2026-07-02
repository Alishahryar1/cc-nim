"""OpenAI Responses API product flow for Codex clients."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.responses import JSONResponse

from api.model_router import ModelRouter
from api.models.anthropic import MessagesRequest
from api.models.openai_responses import OpenAIResponsesRequest
from api.provider_execution import ProviderExecutionService
from api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
)
from api.response_streams import openai_responses_sse_streaming_response
from config.settings import Settings
from core.anthropic import get_user_facing_error_message
from core.openai_responses import OpenAIResponsesAdapter
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

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
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._responses_adapter = responses_adapter or OpenAIResponsesAdapter()
        self._provider_execution = provider_execution or ProviderExecutionService(
            settings,
            provider_getter,
        )

    async def create(self, request_data: OpenAIResponsesRequest) -> object:
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
            require_non_empty_messages(response_request.messages)
            routed = self._model_router.resolve_messages_request(response_request)

            streamed = self._provider_execution.stream(
                routed,
                wire_api="responses",
                raw_log_label="FULL_RESPONSES_PAYLOAD",
                raw_log_payload=request_payload,
            )
            return openai_responses_sse_streaming_response(
                self._responses_adapter.iter_sse_from_anthropic(
                    streamed,
                    request_payload,
                ),
                headers=self._responses_adapter.sse_headers,
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

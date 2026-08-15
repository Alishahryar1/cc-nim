"""OpenAI Chat Completions API product flow.

This handler is a thin edge adapter over :class:`MessagesHandler`. It performs
no routing, provider selection or streaming logic of its own -- it lowers the
inbound request into Anthropic Messages, delegates, and lifts the result back
into Chat Completions shape. Keeping it parasitic on the Messages flow means
every provider, optimization and recovery path FCC already has applies here
for free.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import openai_responses_sse_streaming_response
from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import TokenCounter
from free_claude_code.application.ports import ProviderResolver
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import get_token_count
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.openai_chat_api import (
    ChatCompletionsRequest,
    anthropic_message_to_chat_completion,
    anthropic_sse_to_chat_sse,
    chat_request_to_messages_request,
    new_completion_id,
)
from free_claude_code.core.trace import trace_event

from .messages import MessagesHandler

_ANTHROPIC_ERROR_TO_OPENAI = {
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "invalid_request_error",
    "permission_error": "invalid_request_error",
    "not_found_error": "invalid_request_error",
    "rate_limit_error": "rate_limit_error",
    "overloaded_error": "server_error",
    "api_error": "server_error",
}

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _openai_error_payload(anthropic_error: dict[str, Any]) -> dict[str, Any]:
    error = anthropic_error.get("error")
    if not isinstance(error, dict):
        error = {}
    anthropic_type = str(error.get("type") or "api_error")
    return {
        "error": {
            "message": error.get("message") or "Request failed.",
            "type": _ANTHROPIC_ERROR_TO_OPENAI.get(anthropic_type, "server_error"),
            "code": anthropic_type,
            "param": None,
        }
    }


async def _decoded_chunks(source: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Normalise a body iterator that may yield ``bytes`` or ``str``."""
    async for chunk in source:
        if isinstance(chunk, bytes | bytearray):
            yield bytes(chunk).decode("utf-8", errors="replace")
        else:
            yield str(chunk)


class ChatCompletionsHandler:
    """Handle inbound OpenAI-compatible Chat Completions requests."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        token_counter: TokenCounter = get_token_count,
        generation_id: int | None = None,
    ) -> None:
        self._settings = settings
        self._messages = MessagesHandler(
            settings,
            provider_resolver=provider_resolver,
            token_counter=token_counter,
            generation_id=generation_id,
        )

    async def create(
        self, request_data: ChatCompletionsRequest, *, request_id: str | None = None
    ) -> object:
        """Create a Chat Completions response (JSON, or SSE when stream=true)."""
        request_id = request_id or new_request_id()
        completion_id = new_completion_id()

        messages_request = chat_request_to_messages_request(request_data)
        trace_event(
            stage="routing",
            event="free_claude_code.api.chat_completions.translated",
            source="api",
            model=request_data.model,
            stream=request_data.stream,
            tools=len(messages_request.tools or []),
        )
        logger.debug(
            "chat/completions -> messages: model={} stream={} tools={}",
            request_data.model,
            request_data.stream,
            len(messages_request.tools or []),
        )

        try:
            response = await self._messages.create(
                messages_request, request_id=request_id
            )
        except ApplicationError:
            raise

        if request_data.stream:
            return await self._stream_response(
                response,
                request_data=request_data,
                completion_id=completion_id,
            )
        return self._json_response(
            response,
            request_data=request_data,
            completion_id=completion_id,
        )

    async def _stream_response(
        self,
        response: object,
        *,
        request_data: ChatCompletionsRequest,
        completion_id: str,
    ) -> object:
        if not isinstance(response, StreamingResponse):
            # An error surfaced before the stream opened; report it in
            # OpenAI's non-streaming error shape, which SDKs handle.
            return self._error_passthrough(response)

        return await openai_responses_sse_streaming_response(
            self._lift_anthropic_sse(
                response,
                model=request_data.model,
                completion_id=completion_id,
                include_usage=request_data.wants_usage_in_stream(),
            ),
            headers=_SSE_HEADERS,
            pre_start_error_response=self._pre_start_stream_error,
        )

    async def _lift_anthropic_sse(
        self,
        response: StreamingResponse,
        *,
        model: str,
        completion_id: str,
        include_usage: bool,
    ) -> AsyncIterator[str]:
        """Translate Anthropic SSE and close the inner Messages stream."""
        try:
            async for chunk in anthropic_sse_to_chat_sse(
                _decoded_chunks(response.body_iterator),
                model=model,
                completion_id=completion_id,
                include_usage=include_usage,
            ):
                yield chunk
        finally:
            aclose = getattr(response, "aclose", None)
            if callable(aclose):
                await aclose()

    def _pre_start_stream_error(self, exc: BaseException) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": safe_exception_message(exc) or "Request failed.",
                    "type": "server_error",
                    "code": "api_error",
                    "param": None,
                }
            },
        )

    def _json_response(
        self,
        response: object,
        *,
        request_data: ChatCompletionsRequest,
        completion_id: str,
    ) -> object:
        if not isinstance(response, JSONResponse):
            return response

        try:
            payload = json.loads(bytes(response.body).decode("utf-8"))
        except (TypeError, ValueError):
            return response

        if not isinstance(payload, dict) or payload.get("type") == "error":
            return self._error_passthrough(response, payload=payload)
        if "error" in payload:
            return self._error_passthrough(response, payload=payload)

        return JSONResponse(
            content=anthropic_message_to_chat_completion(
                payload,
                model=request_data.model,
                completion_id=completion_id,
            )
        )

    def _error_passthrough(
        self, response: object, *, payload: dict[str, Any] | None = None
    ) -> object:
        if not isinstance(response, JSONResponse):
            return response
        if payload is None:
            try:
                payload = json.loads(bytes(response.body).decode("utf-8"))
            except (TypeError, ValueError):
                return response
        if not isinstance(payload, dict):
            return response
        return JSONResponse(
            status_code=response.status_code,
            content=_openai_error_payload(payload),
        )

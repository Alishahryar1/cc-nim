"""Facade for OpenAI Responses protocol adaptation."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, ClassVar

from .errors import ResponsesConversionError, openai_error_payload
from .events import OPENAI_RESPONSES_SSE_HEADERS
from .ids import new_response_id
from .input import convert_request_to_anthropic_payload
from .output import convert_message_to_response
from .stream import (
    collect_response_from_anthropic_sse,
    iter_responses_sse_from_anthropic,
    iter_responses_sse_from_message,
)


class OpenAIResponsesAdapter:
    """Convert between OpenAI Responses and the proxy's Anthropic core path."""

    ConversionError: ClassVar[type[ResponsesConversionError]] = ResponsesConversionError
    sse_headers: ClassVar[dict[str, str]] = OPENAI_RESPONSES_SSE_HEADERS

    def to_anthropic_payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return convert_request_to_anthropic_payload(request)

    def from_anthropic_message(
        self,
        message: Mapping[str, Any],
        request: Mapping[str, Any],
        *,
        response_id: str | None = None,
    ) -> dict[str, Any]:
        return convert_message_to_response(
            message,
            request,
            response_id=response_id or new_response_id(),
        )

    def iter_sse_from_anthropic(
        self,
        chunks: AsyncIterable[Any],
        request: Mapping[str, Any],
    ) -> AsyncIterator[str]:
        return iter_responses_sse_from_anthropic(chunks, request)

    async def collect_from_anthropic_sse(
        self,
        chunks: AsyncIterable[Any],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await collect_response_from_anthropic_sse(chunks, request)

    def iter_sse_from_anthropic_message(
        self,
        message: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> list[str]:
        return iter_responses_sse_from_message(message, request)

    def error_payload(self, *, message: str, error_type: str) -> dict[str, Any]:
        return openai_error_payload(message=message, error_type=error_type)

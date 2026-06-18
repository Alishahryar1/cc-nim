"""Facade for OpenAI Responses protocol adaptation."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, ClassVar

from .errors import ResponsesConversionError, openai_error_payload
from .events import OPENAI_RESPONSES_SSE_HEADERS
from .input import convert_request_to_anthropic_payload
from .stream import iter_responses_sse_from_anthropic


class OpenAIResponsesAdapter:
    """Convert between OpenAI Responses and the proxy's Anthropic core path."""

    ConversionError: ClassVar[type[ResponsesConversionError]] = ResponsesConversionError
    sse_headers: ClassVar[dict[str, str]] = OPENAI_RESPONSES_SSE_HEADERS

    def to_anthropic_payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return convert_request_to_anthropic_payload(request)

    def iter_sse_from_anthropic(
        self,
        chunks: AsyncIterable[Any],
        request: Mapping[str, Any],
    ) -> AsyncIterator[str]:
        return iter_responses_sse_from_anthropic(chunks, request)

    def error_payload(self, *, message: str, error_type: str) -> dict[str, Any]:
        return openai_error_payload(message=message, error_type=error_type)

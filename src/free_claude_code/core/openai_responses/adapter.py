"""Facade for OpenAI Responses protocol adaptation."""

from typing import Any, ClassVar

from .errors import ResponsesConversionError, openai_error_payload
from .events import OPENAI_RESPONSES_SSE_HEADERS
from .input import convert_request_to_anthropic_payload
from .models import OpenAIResponsesRequest


class OpenAIResponsesAdapter:
    """Convert the current Responses request bridge and public error envelopes."""

    ConversionError: ClassVar[type[ResponsesConversionError]] = ResponsesConversionError
    sse_headers: ClassVar[dict[str, str]] = OPENAI_RESPONSES_SSE_HEADERS

    def to_anthropic_payload(self, request: OpenAIResponsesRequest) -> dict[str, Any]:
        return convert_request_to_anthropic_payload(request)

    def error_payload(self, *, message: str, error_type: str) -> dict[str, Any]:
        return openai_error_payload(message=message, error_type=error_type)

"""OpenAI Responses compatibility helpers."""

from .conversion import (
    ResponsesConversionError,
    anthropic_message_response_to_openai_response,
    openai_error_payload,
    responses_request_to_anthropic_payload,
)
from .sse import (
    OPENAI_RESPONSES_SSE_HEADERS,
    collect_openai_response_from_anthropic_sse,
    format_response_sse_event,
    iter_anthropic_sse_as_openai_responses,
    iter_message_response_as_openai_responses,
)

__all__ = [
    "OPENAI_RESPONSES_SSE_HEADERS",
    "ResponsesConversionError",
    "anthropic_message_response_to_openai_response",
    "collect_openai_response_from_anthropic_sse",
    "format_response_sse_event",
    "iter_anthropic_sse_as_openai_responses",
    "iter_message_response_as_openai_responses",
    "openai_error_payload",
    "responses_request_to_anthropic_payload",
]

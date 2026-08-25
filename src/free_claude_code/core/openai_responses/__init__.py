"""OpenAI Responses protocol adapter."""

from .adapter import OpenAIResponsesAdapter
from .errors import (
    ResponsesConversionError,
    openai_error_payload,
    openai_error_type_for_failure,
    openai_failure_payload,
)
from .models import OpenAIResponsesRequest
from .presenter import (
    ResponsesEventPresenter,
    iter_responses_sse_from_events,
)
from .provider_events import ResponsesEventDecoder, ResponsesStreamFailure
from .provider_input import build_responses_provider_request

__all__ = [
    "OpenAIResponsesAdapter",
    "OpenAIResponsesRequest",
    "ResponsesConversionError",
    "ResponsesEventDecoder",
    "ResponsesEventPresenter",
    "ResponsesStreamFailure",
    "build_responses_provider_request",
    "iter_responses_sse_from_events",
    "openai_error_payload",
    "openai_error_type_for_failure",
    "openai_failure_payload",
]

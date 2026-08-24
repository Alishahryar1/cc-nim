"""OpenAI Responses protocol adapter."""

from .errors import (
    ResponsesConversionError,
    openai_error_payload,
    openai_error_type_for_failure,
    openai_failure_payload,
)
from .events import OPENAI_RESPONSES_SSE_HEADERS
from .ingress import (
    ResponsesIngressResult,
    responses_to_inference_request,
    validate_responses_field_policy,
)
from .models import OpenAIResponsesRequest, ResponsesPresentationSnapshot
from .presenter import (
    ResponsesEventPresenter,
    iter_responses_sse_from_events,
)

__all__ = [
    "OPENAI_RESPONSES_SSE_HEADERS",
    "OpenAIResponsesRequest",
    "ResponsesConversionError",
    "ResponsesEventPresenter",
    "ResponsesIngressResult",
    "ResponsesPresentationSnapshot",
    "iter_responses_sse_from_events",
    "openai_error_payload",
    "openai_error_type_for_failure",
    "openai_failure_payload",
    "responses_to_inference_request",
    "validate_responses_field_policy",
]

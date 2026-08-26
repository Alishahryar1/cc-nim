"""Best-effort token estimates for native Responses requests."""

import json

from free_claude_code.core.token_estimation import estimate_text_tokens

from .models import OpenAIResponsesRequest

_MAX_ESTIMATE_CHARS_PER_FIELD = 1_000_000


def estimate_responses_input_tokens(request: OpenAIResponsesRequest) -> int:
    """Estimate only request fields that contribute model input tokens."""

    values: tuple[object, ...] = (
        request.instructions,
        request.input,
        request.tools,
    )
    total = 0
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        total += estimate_text_tokens(text[:_MAX_ESTIMATE_CHARS_PER_FIELD])
    return total

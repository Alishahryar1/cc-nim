"""OpenAI Responses SSE event formatting."""

import json
from collections.abc import Mapping
from typing import Any

OPENAI_RESPONSES_SSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def format_response_sse_event(event_type: str, data: Mapping[str, Any]) -> str:
    """Format one OpenAI Responses SSE event."""

    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

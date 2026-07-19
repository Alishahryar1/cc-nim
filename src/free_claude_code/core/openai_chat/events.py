"""OpenAI Chat Completions SSE formatting."""

import json
from collections.abc import Mapping
from typing import Any

OPENAI_CHAT_SSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

CHAT_COMPLETION_SSE_DONE = "data: [DONE]\n\n"


def format_chat_sse_chunk(data: Mapping[str, Any]) -> str:
    """Format one OpenAI Chat Completions ``chat.completion.chunk`` SSE frame."""
    return f"data: {json.dumps(data)}\n\n"

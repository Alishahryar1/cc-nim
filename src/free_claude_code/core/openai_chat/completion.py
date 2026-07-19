"""Build non-streaming OpenAI Chat Completions responses.

The internal provider pipeline is always SSE, so a non-streaming client response
is produced by aggregating the Anthropic stream into a single Message (see
``core.anthropic.aggregate_anthropic_sse_to_message``) and converting that here.
"""

import json
import time
from typing import Any

from .ids import new_chat_completion_id
from .models import OpenAIChatCompletionsRequest
from .stop_reason import finish_reason_from_stop_reason


def anthropic_message_to_chat_completion(
    message: dict[str, Any],
    request: OpenAIChatCompletionsRequest,
    *,
    completion_id: str | None = None,
) -> dict[str, Any]:
    """Convert an aggregated Anthropic Message into a ``chat.completion`` object."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    content = "".join(text_parts)
    chat_message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else None,
    }
    if tool_calls:
        chat_message["tool_calls"] = tool_calls

    finish_reason = finish_reason_from_stop_reason(
        _optional_str(message.get("stop_reason")), has_tool_calls=bool(tool_calls)
    )
    raw_usage = message.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = _int(usage.get("input_tokens"))
    completion_tokens = _int(usage.get("output_tokens"))

    return {
        "id": completion_id or new_chat_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": chat_message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0

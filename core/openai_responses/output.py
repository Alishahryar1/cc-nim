"""Convert complete Anthropic messages into OpenAI Responses objects."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from .ids import new_call_id, new_message_item_id, new_reasoning_item_id
from .items import (
    custom_tool_call_item,
    encrypted_reasoning_item,
    function_call_item,
    message_item,
    openai_usage,
    reasoning_item,
)
from .tools import (
    custom_tool_input_text_from_anthropic,
    responses_tool_identity_from_anthropic_name,
)


def convert_message_to_response(
    message: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    response_id: str,
    status: str = "completed",
) -> dict[str, Any]:
    """Convert a complete Anthropic message response into a Responses object."""

    output: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        text = "".join(text_parts)
        text_parts.clear()
        if text:
            output.append(message_item(new_message_item_id(), text, "completed"))

    for block in _message_content_blocks(message):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "thinking":
            flush_text()
            output.append(
                reasoning_item(
                    new_reasoning_item_id(),
                    str(block.get("thinking", "")),
                    "completed",
                )
            )
        elif block_type == "redacted_thinking":
            flush_text()
            output.append(
                encrypted_reasoning_item(
                    new_reasoning_item_id(),
                    str(block.get("data", "")),
                    "completed",
                )
            )
        elif block_type == "tool_use":
            flush_text()
            identity = responses_tool_identity_from_anthropic_name(
                request, str(block.get("name", ""))
            )
            block_id = str(block.get("id", "") or new_call_id())
            if identity.kind == "custom":
                output.append(
                    custom_tool_call_item(
                        block_id=block_id,
                        name=identity.name,
                        namespace=identity.namespace,
                        input_text=custom_tool_input_text_from_anthropic(
                            block.get("input")
                        ),
                        status="completed",
                    )
                )
            else:
                output.append(
                    function_call_item(
                        block_id=block_id,
                        name=identity.name,
                        namespace=identity.namespace,
                        arguments=json.dumps(block.get("input") or {}),
                        status="completed",
                    )
                )
    flush_text()

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": str(request.get("model", message.get("model", ""))),
        "output": output,
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "tool_choice": request.get("tool_choice", "auto"),
        "temperature": request.get("temperature"),
        "top_p": request.get("top_p"),
        "max_output_tokens": request.get("max_output_tokens"),
        "usage": openai_usage(message.get("usage")),
        "error": None if status == "completed" else {},
    }


def _message_content_blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]

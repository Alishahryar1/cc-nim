"""Responses object and output item builders."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any


def base_response(
    request: Mapping[str, Any], *, response_id: str, status: str
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": str(request.get("model", "")),
        "output": [],
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "tool_choice": request.get("tool_choice", "auto"),
        "temperature": request.get("temperature"),
        "top_p": request.get("top_p"),
        "max_output_tokens": request.get("max_output_tokens"),
        "usage": None,
        "error": None,
    }


def message_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def reasoning_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def encrypted_reasoning_item(
    item_id: str, encrypted_content: str, status: str
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "encrypted_content": encrypted_content,
    }


def function_call_item(
    *,
    block_id: str,
    name: str,
    namespace: str | None,
    arguments: str,
    status: str,
) -> dict[str, Any]:
    item = {
        "id": block_id if block_id.startswith("fc_") else f"fc_{uuid.uuid4().hex[:24]}",
        "type": "function_call",
        "status": status,
        "call_id": block_id,
        "name": name,
        "arguments": arguments,
    }
    if namespace:
        item["namespace"] = namespace
    return item


def custom_tool_call_item(
    *,
    block_id: str,
    name: str,
    namespace: str | None,
    input_text: str,
    status: str,
) -> dict[str, Any]:
    item = {
        "id": (
            block_id if block_id.startswith("ctc_") else f"ctc_{uuid.uuid4().hex[:24]}"
        ),
        "type": "custom_tool_call",
        "status": status,
        "call_id": block_id,
        "name": name,
        "input": input_text,
    }
    if namespace:
        item["namespace"] = namespace
    return item


def openai_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    return {
        "input_tokens": input_tokens if isinstance(input_tokens, int) else 0,
        "output_tokens": output_tokens if isinstance(output_tokens, int) else 0,
        "total_tokens": (
            (input_tokens if isinstance(input_tokens, int) else 0)
            + (output_tokens if isinstance(output_tokens, int) else 0)
        ),
    }


def message_item_text(item: Mapping[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    return "".join(parts)


def reasoning_item_text(item: Mapping[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "reasoning_text"
    ]
    return "".join(parts)


def in_progress_item(item: Mapping[str, Any]) -> dict[str, Any]:
    clone = dict(item)
    clone["status"] = "in_progress"
    if clone.get("type") == "message":
        clone["content"] = []
    if clone.get("type") == "reasoning" and "content" in clone:
        clone["content"] = []
    if clone.get("type") == "function_call":
        clone["arguments"] = ""
    if clone.get("type") == "custom_tool_call":
        clone["input"] = ""
    return clone

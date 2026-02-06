"""Response conversion for NVIDIA NIM provider."""

import json
import uuid
from typing import Any

from .utils import map_stop_reason, extract_think_content


def convert_response(response_json: dict, original_request: Any) -> dict:
    """Convert OpenAI response to Anthropic format."""
    choice = response_json["choices"][0]
    message = choice["message"]
    content = []

    # Extract reasoning from various sources
    reasoning = message.get("reasoning_content")
    if not reasoning:
        reasoning_details = message.get("reasoning_details")
        if reasoning_details and isinstance(reasoning_details, list):
            reasoning = "\n".join(
                item.get("text", "")
                for item in reasoning_details
                if isinstance(item, dict)
            )

    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    # Extract text content (with think tag handling)
    if message.get("content"):
        raw_content = message["content"]
        if isinstance(raw_content, str):
            if not reasoning:
                think_content, raw_content = extract_think_content(raw_content)
                if think_content:
                    content.append({"type": "thinking", "thinking": think_content})
            if raw_content:
                content.append({"type": "text", "text": raw_content})
        elif isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content.append(item)

    # Extract tool calls
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = tc["function"].get("arguments", {})
            content.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": args,
                }
            )

    if not content:
        # NIM models (especially Mistral-based) often require non-empty content.
        # Adding a single space satisfies this requirement while avoiding
        # the "(no content)" display issue in Claude Code.
        content.append({"type": "text", "text": " "})

    usage = response_json.get("usage", {})

    return {
        "id": response_json.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": "assistant",
        "model": original_request.model,
        "content": content,
        "stop_reason": map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }

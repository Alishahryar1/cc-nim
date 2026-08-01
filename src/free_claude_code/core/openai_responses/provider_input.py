"""Convert Anthropic Messages into an upstream OpenAI Responses request."""

import json
from typing import Any

from free_claude_code.core.anthropic.content import get_block_attr, get_block_type
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.anthropic.request_serialization import (
    serialize_tool_result_content,
)
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy

from .errors import ResponsesConversionError


def build_responses_provider_request(
    request: MessagesRequest,
    *,
    reasoning: ReasoningPolicy,
) -> dict[str, Any]:
    """Build a stateless Responses request without silently dropping fields."""

    _validate_supported_request(request)
    tool_names = OpenAIToolNameCodec.from_request(request)
    instructions = _system_text(request)
    input_items: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            text = _message_text(message.content, context="system message")
            if text:
                instructions.append(text)
        elif message.role == "assistant":
            input_items.extend(
                _assistant_items(
                    message.content,
                    reasoning_content=message.reasoning_content,
                    tool_names=tool_names,
                )
            )
        else:
            input_items.extend(_user_items(message.content))

    if not input_items:
        raise ResponsesConversionError(
            "OpenAI Responses conversion requires at least one user or assistant item."
        )

    body: dict[str, Any] = {
        "model": request.model,
        "input": input_items,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if request.max_tokens is not None:
        body["max_output_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.metadata is not None:
        body["metadata"] = request.metadata
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool_names.encode(tool.name),
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
                "strict": False,
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        body["tool_choice"] = _tool_choice(request.tool_choice, tool_names=tool_names)
    if reasoning_config := _reasoning_config(reasoning):
        body["reasoning"] = reasoning_config
    return body


def _validate_supported_request(request: MessagesRequest) -> None:
    if request.model_extra:
        raise ResponsesConversionError(
            "OpenAI Responses does not support these request fields: "
            f"{sorted(str(key) for key in request.model_extra)}."
        )
    provider_tool_types = sorted(
        {tool.type for tool in request.tools or () if tool.type is not None}
    )
    if provider_tool_types:
        raise ResponsesConversionError(
            "OpenAI Responses cannot represent provider-managed tool types: "
            f"{provider_tool_types}."
        )
    unsupported: list[str] = []
    if request.stop_sequences:
        unsupported.append("stop_sequences")
    if request.top_k is not None:
        unsupported.append("top_k")
    if request.context_management:
        unsupported.append("context_management")
    if request.output_config:
        unsupported.append("output_config")
    if request.mcp_servers:
        unsupported.append("mcp_servers")
    if request.extra_body:
        unsupported.append("extra_body")
    if unsupported:
        raise ResponsesConversionError(
            "OpenAI Responses cannot represent these fields without data loss: "
            f"{unsupported}."
        )


def _system_text(request: MessagesRequest) -> list[str]:
    if request.system is None:
        return []
    if isinstance(request.system, str):
        return [request.system] if request.system else []
    return [part.text for part in request.system if part.text]


def _assistant_items(
    content: Any,
    *,
    reasoning_content: str | None,
    tool_names: OpenAIToolNameCodec,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        items = [_assistant_message([{"type": "output_text", "text": content}])]
        if reasoning_content is not None:
            items.insert(
                0,
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning_content}],
                },
            )
        return items
    if not isinstance(content, list):
        raise ResponsesConversionError(
            "Assistant content must be text or content blocks."
        )

    items: list[dict[str, Any]] = []
    text_parts: list[dict[str, Any]] = []
    thinking_parts: list[str] = (
        [reasoning_content] if reasoning_content is not None else []
    )
    encrypted_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            items.append(_assistant_message(list(text_parts)))
            text_parts.clear()

    for block in content:
        block_type = get_block_type(block)
        if block_type == "text":
            text_parts.append(
                {
                    "type": "output_text",
                    "text": str(get_block_attr(block, "text", "")),
                }
            )
        elif block_type == "thinking":
            thinking_parts.append(str(get_block_attr(block, "thinking", "")))
        elif block_type == "redacted_thinking":
            encrypted_parts.append(str(get_block_attr(block, "data", "")))
        elif block_type == "tool_use":
            flush_text()
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(get_block_attr(block, "id", "")),
                    "name": tool_names.encode(str(get_block_attr(block, "name", ""))),
                    "arguments": json.dumps(
                        get_block_attr(block, "input", {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        else:
            raise ResponsesConversionError(
                "OpenAI Responses cannot represent assistant content block "
                f"{block_type!r}."
            )
    if thinking_parts or encrypted_parts:
        summary = [
            {"type": "summary_text", "text": text} for text in thinking_parts if text
        ]
        if encrypted_parts:
            for index, encrypted in enumerate(encrypted_parts):
                item: dict[str, Any] = {
                    "type": "reasoning",
                    "summary": summary if index == 0 else [],
                    "encrypted_content": encrypted,
                }
                items.insert(index, item)
        else:
            items.insert(0, {"type": "reasoning", "summary": summary})
    flush_text()
    return items


def _user_items(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [_user_message([{"type": "input_text", "text": content}])]
    if not isinstance(content, list):
        raise ResponsesConversionError("User content must be text or content blocks.")

    items: list[dict[str, Any]] = []
    message_parts: list[dict[str, Any]] = []

    def flush_message() -> None:
        if message_parts:
            items.append(_user_message(list(message_parts)))
            message_parts.clear()

    for block in content:
        block_type = get_block_type(block)
        if block_type == "text":
            message_parts.append(
                {
                    "type": "input_text",
                    "text": str(get_block_attr(block, "text", "")),
                }
            )
        elif block_type == "image":
            message_parts.append(_image_part(block))
        elif block_type == "tool_result":
            flush_message()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(get_block_attr(block, "tool_use_id", "")),
                    "output": serialize_tool_result_content(
                        get_block_attr(block, "content")
                    ),
                }
            )
        elif block_type == "document":
            raise ResponsesConversionError(
                "OpenAI Responses provider does not support Anthropic document blocks."
            )
        else:
            raise ResponsesConversionError(
                f"OpenAI Responses cannot represent user content block {block_type!r}."
            )
    flush_message()
    return items


def _image_part(block: Any) -> dict[str, Any]:
    source = get_block_attr(block, "source", {})
    source_type = get_block_attr(source, "type")
    if source_type == "url":
        url = get_block_attr(source, "url")
    elif source_type == "base64":
        media_type = get_block_attr(source, "media_type")
        data = get_block_attr(source, "data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise ResponsesConversionError("Base64 images require media_type and data.")
        url = f"data:{media_type};base64,{data}"
    else:
        raise ResponsesConversionError(
            f"Unsupported image source type {source_type!r}."
        )
    if not isinstance(url, str) or not url:
        raise ResponsesConversionError("Image source requires a non-empty URL.")
    return {"type": "input_image", "image_url": url}


def _message_text(content: Any, *, context: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesConversionError(f"{context} must contain only text.")
    parts: list[str] = []
    for block in content:
        if get_block_type(block) != "text":
            raise ResponsesConversionError(f"{context} must contain only text.")
        parts.append(str(get_block_attr(block, "text", "")))
    return "\n\n".join(parts)


def _assistant_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": content}


def _user_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": content}


def _tool_choice(
    choice: dict[str, Any],
    *,
    tool_names: OpenAIToolNameCodec,
) -> str | dict[str, str]:
    choice_type = choice.get("type")
    if choice_type in {"auto", "none"}:
        return str(choice_type)
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesConversionError("Forced tool choice requires a tool name.")
        return {"type": "function", "name": tool_names.encode(name)}
    raise ResponsesConversionError(f"Unsupported tool_choice type {choice_type!r}.")


def _reasoning_config(reasoning: ReasoningPolicy) -> dict[str, str]:
    if reasoning.control is ReasoningControl.OFF:
        return {"effort": "none"}
    if reasoning.effort is not None:
        return {"effort": reasoning.effort.value, "summary": "auto"}
    if reasoning.requests_reasoning:
        return {"summary": "auto"}
    return {}

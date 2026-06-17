"""Convert between OpenAI Responses payloads and Anthropic-style payloads."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

_MAX_ANTHROPIC_TOOL_NAME_LEN = 64
_NAMESPACE_TOOL_SEPARATOR = "__"
_UNSUPPORTED_PASSIVE_TOOL_TYPES = frozenset({"web_search", "image_generation"})
_INVALID_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


class ResponsesConversionError(ValueError):
    """Raised when a Responses request cannot be converted deterministically."""


def responses_request_to_anthropic_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert an OpenAI Responses request into an Anthropic Messages payload."""

    system_parts: list[str] = []
    if instructions := _optional_str(request.get("instructions")):
        system_parts.append(instructions)

    messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None
    for item in _iter_input_items(request.get("input")):
        pending_reasoning = _append_input_item(
            item,
            messages=messages,
            system_parts=system_parts,
            pending_reasoning=pending_reasoning,
        )
    _append_pending_reasoning(messages, pending_reasoning)

    if not messages:
        raise ResponsesConversionError("Responses request input must contain a message")

    payload: dict[str, Any] = {
        "model": _required_str(request.get("model"), "model"),
        "messages": messages,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    _copy_if_present(request, payload, "temperature")
    _copy_if_present(request, payload, "top_p")
    if request.get("max_output_tokens") is not None:
        payload["max_tokens"] = request["max_output_tokens"]
    if isinstance(request.get("metadata"), dict):
        payload["metadata"] = request["metadata"]

    if thinking := _responses_reasoning_to_thinking(request.get("reasoning")):
        payload["thinking"] = thinking

    raw_tool_choice = request.get("tool_choice")
    tools = _convert_tools(request.get("tools"))
    if tools and raw_tool_choice != "none":
        payload["tools"] = tools
    tool_choice = _convert_tool_choice(raw_tool_choice)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return payload


def anthropic_message_response_to_openai_response(
    message: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    response_id: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """Convert a complete Anthropic message response into a Responses object."""

    response_id = response_id or _new_response_id()
    output: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        text = "".join(text_parts)
        text_parts.clear()
        if text:
            output.append(_message_item(_new_message_item_id(), text, "completed"))

    for block in _message_content_blocks(message):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "thinking":
            flush_text()
            output.append(
                _reasoning_item(
                    _new_reasoning_item_id(),
                    str(block.get("thinking", "")),
                    "completed",
                )
            )
        elif block_type == "redacted_thinking":
            flush_text()
            output.append(
                _encrypted_reasoning_item(
                    _new_reasoning_item_id(),
                    str(block.get("data", "")),
                    "completed",
                )
            )
        elif block_type == "tool_use":
            flush_text()
            namespace, name = responses_tool_identity_from_anthropic_name(
                request, str(block.get("name", ""))
            )
            output.append(
                _function_call_item(
                    block_id=str(block.get("id", "") or _new_call_id()),
                    name=name,
                    namespace=namespace,
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
        "usage": _openai_usage(message.get("usage")),
        "error": None if status == "completed" else {},
    }


def openai_error_payload(*, message: str, error_type: str) -> dict[str, Any]:
    """Return an OpenAI-compatible error envelope."""

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def _append_input_item(
    item: Any,
    *,
    messages: list[dict[str, Any]],
    system_parts: list[str],
    pending_reasoning: str | None,
) -> str | None:
    if isinstance(item, str):
        _append_pending_reasoning(messages, pending_reasoning)
        messages.append({"role": "user", "content": item})
        return None
    if not isinstance(item, dict):
        raise ResponsesConversionError(
            f"Unsupported Responses input item: {type(item).__name__}"
        )

    item_type = item.get("type")
    if item_type in (None, "message") or "role" in item:
        role = _required_str(item.get("role", "user"), "input.role")
        if role == "assistant":
            _append_message_item(
                role,
                item.get("content", ""),
                messages,
                system_parts,
                reasoning_content=pending_reasoning,
            )
            return None
        _append_pending_reasoning(messages, pending_reasoning)
        _append_message_item(role, item.get("content", ""), messages, system_parts)
        return None
    if item_type == "function_call":
        namespace = _optional_str(item.get("namespace"))
        name = _required_str(item.get("name"), "function_call.name")
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": _call_id_from_item(item),
                    "name": responses_tool_name_to_anthropic_name(
                        name, namespace=namespace
                    ),
                    "input": _parse_arguments(item.get("arguments")),
                }
            ],
        }
        if pending_reasoning:
            message["reasoning_content"] = pending_reasoning
        messages.append(message)
        return None
    if item_type == "function_call_output":
        _append_pending_reasoning(messages, pending_reasoning)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": _call_id_from_item(item),
                        "content": item.get("output", ""),
                    }
                ],
            }
        )
        return None
    if item_type == "reasoning":
        return _combine_reasoning(pending_reasoning, _reasoning_text_from_item(item))
    if item_type in {"input_text", "output_text", "text"}:
        _append_pending_reasoning(messages, pending_reasoning)
        messages.append({"role": "user", "content": _text_from_part(item)})
        return None

    raise ResponsesConversionError(
        f"Unsupported Responses input item type: {item_type!r}"
    )


def _append_message_item(
    role: str,
    content: Any,
    messages: list[dict[str, Any]],
    system_parts: list[str],
    *,
    reasoning_content: str | None = None,
) -> None:
    normalized_role = "system" if role == "developer" else role
    if normalized_role == "system":
        text = _content_as_text(content)
        if text:
            system_parts.append(text)
        return
    if normalized_role not in {"user", "assistant"}:
        raise ResponsesConversionError(f"Unsupported Responses message role: {role!r}")
    message = {
        "role": normalized_role,
        "content": _convert_message_content(content),
    }
    if normalized_role == "assistant" and reasoning_content:
        message["reasoning_content"] = reasoning_content
    messages.append(message)


def _append_pending_reasoning(
    messages: list[dict[str, Any]], pending_reasoning: str | None
) -> None:
    if pending_reasoning:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": pending_reasoning,
            }
        )


def _iter_input_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _convert_message_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                raise ResponsesConversionError(
                    f"Unsupported Responses content part: {type(part).__name__}"
                )
            part_type = part.get("type")
            if part_type in {"input_text", "output_text", "text"} or "text" in part:
                blocks.append({"type": "text", "text": _text_from_part(part)})
                continue
            if part_type == "refusal":
                blocks.append({"type": "text", "text": str(part.get("refusal", ""))})
                continue
            raise ResponsesConversionError(
                f"Unsupported Responses content part type: {part_type!r}"
            )
        return blocks
    if isinstance(content, dict):
        return [{"type": "text", "text": _text_from_part(content)}]
    raise ResponsesConversionError(
        f"Unsupported Responses message content: {type(content).__name__}"
    )


def _content_as_text(content: Any) -> str:
    converted = _convert_message_content(content)
    if isinstance(converted, str):
        return converted
    return "\n".join(str(block.get("text", "")) for block in converted)


def _text_from_part(part: Mapping[str, Any]) -> str:
    if text := _optional_str(part.get("text")):
        return text
    if text := _optional_str(part.get("input_text")):
        return text
    if text := _optional_str(part.get("output_text")):
        return text
    return ""


def _reasoning_text_from_item(item: Mapping[str, Any]) -> str | None:
    content_parts = _text_parts_from_items(
        item.get("content"), item_type="reasoning_text"
    )
    if content_parts:
        return "\n".join(content_parts)
    summary_parts = _text_parts_from_items(
        item.get("summary"), item_type="summary_text"
    )
    if summary_parts:
        return "\n".join(summary_parts)
    return None


def _text_parts_from_items(value: Any, *, item_type: str) -> list[str]:
    if not isinstance(value, list):
        return []
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == item_type:
            text = _optional_str(item.get("text"))
            if text:
                parts.append(text)
    return parts


def _combine_reasoning(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing}\n{addition}"


def _responses_reasoning_to_thinking(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("effort") == "none":
        return {"type": "disabled", "enabled": False}
    if any(item is not None for item in value.values()):
        return {"type": "enabled", "enabled": True}
    return None


def _convert_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ResponsesConversionError("Responses tools must be a list")

    tools: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict):
            raise ResponsesConversionError(
                f"Unsupported Responses tool: {type(tool).__name__}"
            )
        tool_type = tool.get("type")
        if tool_type == "function":
            tools.append(_convert_function_tool(tool, namespace=None))
            continue
        if tool_type == "namespace":
            tools.extend(_convert_namespace_tool(tool))
            continue
        if tool_type in _UNSUPPORTED_PASSIVE_TOOL_TYPES:
            continue
        if tool_type != "function":
            raise ResponsesConversionError(
                f"Unsupported Responses tool type: {tool_type!r}"
            )
    return tools


def _convert_namespace_tool(tool: Mapping[str, Any]) -> list[dict[str, Any]]:
    namespace = _required_str(tool.get("name"), "tool.namespace.name")
    nested_tools = tool.get("tools")
    if not isinstance(nested_tools, list):
        raise ResponsesConversionError(
            f"Responses namespace tool {namespace!r} tools must be a list"
        )

    converted_tools: list[dict[str, Any]] = []
    for nested_tool in nested_tools:
        if not isinstance(nested_tool, dict):
            raise ResponsesConversionError(
                f"Unsupported Responses namespace tool: {type(nested_tool).__name__}"
            )
        nested_tool_type = nested_tool.get("type")
        if nested_tool_type != "function":
            raise ResponsesConversionError(
                f"Unsupported Responses namespace tool type: {nested_tool_type!r}"
            )
        converted_tools.append(_convert_function_tool(nested_tool, namespace=namespace))
    return converted_tools


def _convert_function_tool(
    tool: Mapping[str, Any], *, namespace: str | None
) -> dict[str, Any]:
    function = tool.get("function")
    source = function if isinstance(function, dict) else tool
    name = _required_str(source.get("name"), "tool.name")
    schema = source.get("parameters")
    if schema is None:
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        raise ResponsesConversionError(
            f"Responses tool {name!r} parameters must be an object"
        )
    converted: dict[str, Any] = {
        "name": responses_tool_name_to_anthropic_name(name, namespace=namespace),
        "input_schema": schema,
    }
    if description := _optional_str(source.get("description")):
        converted["description"] = description
    return converted


def _convert_tool_choice(value: Any) -> dict[str, Any] | None:
    if value is None or value == "auto":
        return None
    if value == "none":
        return None
    if value == "required":
        return {"type": "any"}
    if isinstance(value, dict):
        choice_type = value.get("type")
        if choice_type == "function":
            namespace = _optional_str(value.get("namespace"))
            name = _required_str(value.get("name"), "tool_choice.name")
            return {
                "type": "tool",
                "name": responses_tool_name_to_anthropic_name(
                    name, namespace=namespace
                ),
            }
        if choice_type in {"auto", "any", "tool"}:
            return dict(value)
    raise ResponsesConversionError(f"Unsupported Responses tool_choice: {value!r}")


def responses_tool_name_to_anthropic_name(
    name: str, *, namespace: str | None = None
) -> str:
    """Return a deterministic Anthropic tool name for a Responses tool identity."""

    if not namespace:
        return name
    combined = (
        f"{_tool_name_part(namespace)}"
        f"{_NAMESPACE_TOOL_SEPARATOR}"
        f"{_tool_name_part(name)}"
    )
    if len(combined) <= _MAX_ANTHROPIC_TOOL_NAME_LEN:
        return combined
    digest = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:8]
    prefix_len = _MAX_ANTHROPIC_TOOL_NAME_LEN - len(digest) - 1
    return f"{combined[:prefix_len]}_{digest}"


def responses_tool_identity_from_anthropic_name(
    request: Mapping[str, Any], anthropic_name: str
) -> tuple[str | None, str]:
    """Return the Responses namespace/name represented by an Anthropic tool name."""

    tools = request.get("tools")
    if not isinstance(tools, list):
        return None, anthropic_name
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            source = tool.get("function")
            function = source if isinstance(source, dict) else tool
            if (name := _optional_str(function.get("name"))) and (
                responses_tool_name_to_anthropic_name(name) == anthropic_name
            ):
                return None, name
            continue
        if tool_type != "namespace":
            continue
        namespace = _optional_str(tool.get("name"))
        nested_tools = tool.get("tools")
        if not namespace or not isinstance(nested_tools, list):
            continue
        for nested_tool in nested_tools:
            if not isinstance(nested_tool, dict):
                continue
            if nested_tool.get("type") != "function":
                continue
            source = nested_tool.get("function")
            function = source if isinstance(source, dict) else nested_tool
            if (name := _optional_str(function.get("name"))) and (
                responses_tool_name_to_anthropic_name(name, namespace=namespace)
                == anthropic_name
            ):
                return namespace, name
    return None, anthropic_name


def _tool_name_part(value: str) -> str:
    normalized = _INVALID_TOOL_NAME_CHARS.sub("_", value).strip("_")
    return normalized or "tool"


def _parse_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ResponsesConversionError("Responses function_call arguments must be JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ResponsesConversionError(
            f"Responses function_call arguments are invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ResponsesConversionError(
            "Responses function_call arguments must decode to an object"
        )
    return parsed


def _call_id_from_item(item: Mapping[str, Any]) -> str:
    for key in ("call_id", "id"):
        if value := _optional_str(item.get(key)):
            return value
    return _new_call_id()


def _required_str(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ResponsesConversionError(
        f"Responses field {field_name} must be a non-empty string"
    )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _copy_if_present(
    source: Mapping[str, Any], target: dict[str, Any], field_name: str
) -> None:
    if source.get(field_name) is not None:
        target[field_name] = source[field_name]


def _message_content_blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _message_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _encrypted_reasoning_item(
    item_id: str, encrypted_content: str, status: str
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "encrypted_content": encrypted_content,
    }


def _function_call_item(
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


def _openai_usage(value: Any) -> dict[str, int] | None:
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


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _new_message_item_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _new_reasoning_item_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"

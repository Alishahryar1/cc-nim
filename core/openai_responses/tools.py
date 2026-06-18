"""Tool conversion helpers for the OpenAI Responses adapter."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ResponsesConversionError
from .ids import new_call_id

_MAX_ANTHROPIC_TOOL_NAME_LEN = 64
_NAMESPACE_TOOL_SEPARATOR = "__"
_UNSUPPORTED_PASSIVE_TOOL_TYPES = frozenset(
    {"web_search", "image_generation", "tool_search"}
)
_INVALID_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class ResponsesToolIdentity:
    kind: Literal["function", "custom"]
    name: str
    namespace: str | None = None


def convert_tools(value: Any) -> list[dict[str, Any]] | None:
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
        if tool_type == "custom":
            tools.append(_convert_custom_tool(tool, namespace=None))
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


def convert_tool_choice(value: Any) -> dict[str, Any] | None:
    if value is None or value == "auto":
        return None
    if value == "none":
        return None
    if value == "required":
        return {"type": "any"}
    if isinstance(value, dict):
        choice_type = value.get("type")
        if choice_type == "function":
            namespace = optional_str(value.get("namespace"))
            name = required_str(value.get("name"), "tool_choice.name")
            return {
                "type": "tool",
                "name": responses_tool_name_to_anthropic_name(
                    name, namespace=namespace
                ),
            }
        if choice_type == "custom":
            source = _custom_source(value)
            namespace = optional_str(source.get("namespace")) or optional_str(
                value.get("namespace")
            )
            name = required_str(source.get("name"), "tool_choice.name")
            return {
                "type": "tool",
                "name": responses_tool_name_to_anthropic_name(
                    name, namespace=namespace
                ),
            }
        if choice_type == "tool":
            namespace = optional_str(value.get("namespace"))
            name = optional_str(value.get("name"))
            if name:
                return {
                    "type": "tool",
                    "name": responses_tool_name_to_anthropic_name(
                        name, namespace=namespace
                    ),
                }
            return dict(value)
        if choice_type in {"auto", "any"}:
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
) -> ResponsesToolIdentity:
    """Return the Responses namespace/name represented by an Anthropic tool name."""

    tools = request.get("tools")
    if not isinstance(tools, list):
        return ResponsesToolIdentity(kind="function", name=anthropic_name)
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            source = tool.get("function")
            function = source if isinstance(source, dict) else tool
            if (name := optional_str(function.get("name"))) and (
                responses_tool_name_to_anthropic_name(name) == anthropic_name
            ):
                return ResponsesToolIdentity(kind="function", name=name)
            continue
        if tool_type == "custom":
            source = _custom_source(tool)
            if (name := optional_str(source.get("name"))) and (
                responses_tool_name_to_anthropic_name(name) == anthropic_name
            ):
                return ResponsesToolIdentity(kind="custom", name=name)
            continue
        if tool_type != "namespace":
            continue
        namespace = optional_str(tool.get("name"))
        nested_tools = tool.get("tools")
        if not namespace or not isinstance(nested_tools, list):
            continue
        for nested_tool in nested_tools:
            if not isinstance(nested_tool, dict):
                continue
            nested_tool_type = nested_tool.get("type")
            if nested_tool_type == "function":
                source = nested_tool.get("function")
                function = source if isinstance(source, dict) else nested_tool
                if (name := optional_str(function.get("name"))) and (
                    responses_tool_name_to_anthropic_name(name, namespace=namespace)
                    == anthropic_name
                ):
                    return ResponsesToolIdentity(
                        kind="function", name=name, namespace=namespace
                    )
                continue
            if nested_tool_type == "custom":
                source = _custom_source(nested_tool)
                if (name := optional_str(source.get("name"))) and (
                    responses_tool_name_to_anthropic_name(name, namespace=namespace)
                    == anthropic_name
                ):
                    return ResponsesToolIdentity(
                        kind="custom", name=name, namespace=namespace
                    )
    return ResponsesToolIdentity(kind="function", name=anthropic_name)


def parse_arguments(value: Any) -> dict[str, Any]:
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


def custom_tool_input_to_anthropic(value: Any) -> dict[str, str]:
    return {"input": custom_tool_input_text(value)}


def custom_tool_input_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def custom_tool_input_text_from_anthropic(value: Any) -> str:
    if isinstance(value, Mapping):
        raw_input = value.get("input")
        if isinstance(raw_input, str):
            return raw_input
        if raw_input is not None:
            return custom_tool_input_text(raw_input)
        if not value:
            return ""
        return _json_dumps(value)
    return custom_tool_input_text(value)


def custom_tool_input_text_from_arguments(arguments: str) -> str:
    if not arguments:
        return ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return custom_tool_input_text_from_anthropic(parsed)


def call_id_from_item(item: Mapping[str, Any]) -> str:
    for key in ("call_id", "id"):
        if value := optional_str(item.get(key)):
            return value
    return new_call_id()


def required_str(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ResponsesConversionError(
        f"Responses field {field_name} must be a non-empty string"
    )


def optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _convert_namespace_tool(tool: Mapping[str, Any]) -> list[dict[str, Any]]:
    namespace = required_str(tool.get("name"), "tool.namespace.name")
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
        if nested_tool_type == "function":
            converted_tools.append(
                _convert_function_tool(nested_tool, namespace=namespace)
            )
            continue
        if nested_tool_type == "custom":
            converted_tools.append(
                _convert_custom_tool(nested_tool, namespace=namespace)
            )
            continue
        raise ResponsesConversionError(
            f"Unsupported Responses namespace tool type: {nested_tool_type!r}"
        )
    return converted_tools


def _convert_function_tool(
    tool: Mapping[str, Any], *, namespace: str | None
) -> dict[str, Any]:
    function = tool.get("function")
    source = function if isinstance(function, dict) else tool
    name = required_str(source.get("name"), "tool.name")
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
    if description := optional_str(source.get("description")):
        converted["description"] = description
    return converted


def _convert_custom_tool(
    tool: Mapping[str, Any], *, namespace: str | None
) -> dict[str, Any]:
    source = _custom_source(tool)
    name = required_str(source.get("name"), "tool.name")
    converted: dict[str, Any] = {
        "name": responses_tool_name_to_anthropic_name(name, namespace=namespace),
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Free-form input for the custom tool.",
                }
            },
            "required": ["input"],
        },
    }
    if description := _custom_tool_description(source):
        converted["description"] = description
    return converted


def _custom_source(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    custom = tool.get("custom")
    return custom if isinstance(custom, Mapping) else tool


def _custom_tool_description(source: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    if description := optional_str(source.get("description")):
        parts.append(description)
    format_value = source.get("format")
    if isinstance(format_value, Mapping):
        format_type = optional_str(format_value.get("type"))
        if format_type == "text":
            parts.append("Custom tool input format: unconstrained text.")
        elif format_type == "grammar":
            syntax = optional_str(format_value.get("syntax"))
            definition = optional_str(format_value.get("definition"))
            guidance = "Custom tool input format: grammar"
            if syntax:
                guidance = f"{guidance} ({syntax})"
            guidance = f"{guidance}: {definition}" if definition else f"{guidance}."
            parts.append(guidance)
        elif format_type:
            parts.append(f"Custom tool input format: {format_type}.")
        else:
            parts.append(f"Custom tool input format: {_json_dumps(format_value)}")
    return "\n\n".join(parts) if parts else None


def _tool_name_part(value: str) -> str:
    normalized = _INVALID_TOOL_NAME_CHARS.sub("_", value).strip("_")
    return normalized or "tool"


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)

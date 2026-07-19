"""Tool and tool-choice conversion for OpenAI Chat Completions ingress."""

import json
from typing import Any

from .errors import ChatCompletionsConversionError


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert OpenAI Chat Completions ``tools`` into Anthropic tool definitions.

    Malformed entries are rejected rather than silently dropped, so a request
    never reaches the provider with fewer tools than the caller declared (which
    would silently change the model's capabilities). Mirrors the strict
    validation in ``core/openai_responses``.
    """
    if not tools:
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ChatCompletionsConversionError(
                f"Unsupported chat tool: {type(tool).__name__}"
            )
        tool_type = tool.get("type")
        if tool_type not in (None, "function"):
            raise ChatCompletionsConversionError(
                f"Unsupported chat tool type: {tool_type!r}"
            )
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ChatCompletionsConversionError(
                "chat tool requires a 'function' object"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ChatCompletionsConversionError(
                "chat tool function requires a non-empty name"
            )
        anthropic_tool: dict[str, Any] = {
            "name": name,
            "input_schema": _tool_input_schema(function.get("parameters")),
        }
        if description := function.get("description"):
            anthropic_tool["description"] = str(description)
        converted.append(anthropic_tool)
    return converted


def convert_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    """Convert an OpenAI ``tool_choice`` into an Anthropic ``tool_choice``.

    Unknown or malformed values raise instead of falling back to automatic tool
    selection, so a typo in a required named choice becomes a client error
    rather than silently letting the model pick a different tool or none.
    """
    if tool_choice in (None, "auto", "none"):
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise ChatCompletionsConversionError(
                "tool_choice.function.name is required"
            )
        return {"type": "tool", "name": name}
    raise ChatCompletionsConversionError(
        f"Unsupported chat tool_choice: {tool_choice!r}"
    )


def parse_arguments(arguments: Any) -> Any:
    """Parse an OpenAI tool-call ``arguments`` payload into Anthropic tool input."""
    if arguments in (None, ""):
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ChatCompletionsConversionError(
                f"Invalid tool call arguments JSON: {exc}"
            ) from exc
    raise ChatCompletionsConversionError(
        f"Unsupported tool call arguments type: {type(arguments).__name__}"
    )


def _tool_input_schema(parameters: Any) -> dict[str, Any]:
    if isinstance(parameters, dict):
        return parameters
    return {"type": "object", "properties": {}}

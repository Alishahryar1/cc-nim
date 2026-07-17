"""Convert Anthropic Messages API requests to ChatGPT Responses API format."""

from __future__ import annotations

import json
from typing import Any

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.conversion import (
    AnthropicToOpenAIConverter,
    OpenAIConversionError,
    ReasoningReplayMode,
)
from free_claude_code.core.anthropic.models import MessagesRequest

CHATGPT_DEFAULT_REASONING_EFFORT = "medium"
CHATGPT_DEFAULT_REASONING_SUMMARY = "auto"


def _strip_openai_system_message(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the leading system message as Responses API instructions."""
    if messages and messages[0].get("role") == "system":
        instructions = messages[0].get("content")
        return instructions, messages[1:]
    return None, messages


def _openai_message_to_chatgpt_input(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI-chat message to a ChatGPT Responses API input item."""
    role = message.get("role")
    content = message.get("content")

    if role == "tool":
        return {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Tool result ({message.get('tool_call_id')}): {content}",
                }
            ],
        }

    if role == "assistant":
        item: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
        }
        if isinstance(content, str):
            item["content"] = [{"type": "output_text", "text": content}]
        elif isinstance(content, list):
            item["content"] = content
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            item["tool_calls"] = tool_calls
        return item

    # user / system converted to user
    if role == "system":
        role = "user"
    if isinstance(content, str):
        return {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": content}],
        }
    return {
        "type": "message",
        "role": role,
        "content": content if isinstance(content, list) else [],
    }


def _openai_messages_to_chatgpt_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-chat message list to Responses API input list."""
    return [_openai_message_to_chatgpt_input(msg) for msg in messages]


def _convert_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tools to ChatGPT Responses API tool definitions.

    The ChatGPT/Codex backend historically exposes only a small set of built-in
    tools, but we forward tools in the standard function shape so the backend
    can reject or accept them with its own error message.
    """
    if not tools:
        return None
    result: list[dict[str, Any]] = []
    for tool in tools:
        schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
        result.append({
            "type": "function",
            "name": getattr(tool, "name", "unknown"),
            "description": getattr(tool, "description", None) or "",
            "parameters": schema,
        })
    return result


def _convert_tool_choice(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to ChatGPT Responses API tool_choice."""
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice_type = tool_choice.get("type")
    if choice_type == "tool":
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    if choice_type in {"auto", "none", "required"}:
        return choice_type
    if choice_type == "any":
        return "required"
    return tool_choice


def _supports_reasoning(model: str) -> bool:
    """Return True for models known to expose reasoning through the backend."""
    name = model.lower()
    return (
        name.startswith("gpt-5")
        or name.startswith("codex")
        or name.startswith("o")
    )


def _extract_system_instructions(request: MessagesRequest) -> str | None:
    """Return the top-level Anthropic system prompt as a single string."""
    system = request.system
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        text = "\n\n".join(parts)
        return text if text else None
    return None


def build_chatgpt_oauth_request_body(
    request: MessagesRequest,
    *,
    default_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a ChatGPT Responses API request body from an Anthropic request."""
    if request.extra_body:
        raise InvalidRequestError(
            "ChatGPT OAuth provider does not support caller extra_body on requests."
        )

    try:
        openai_messages = AnthropicToOpenAIConverter.convert_messages(
            request.messages,
            reasoning_replay=ReasoningReplayMode.THINK_TAGS,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    instructions = _extract_system_instructions(request)
    _, chat_messages = _strip_openai_system_message(openai_messages)

    body: dict[str, Any] = {
        "model": request.model,
        "input": _openai_messages_to_chatgpt_input(chat_messages),
        "store": False,
        "stream": True,
        "parallel_tool_calls": False,
    }

    if instructions:
        body["instructions"] = instructions

    # OpenCode's codex plugin clears maxOutputTokens to match the Codex CLI:
    # the ChatGPT/Codex Responses endpoint behaves best when the caller does
    # not impose an explicit output limit. ``default_max_tokens`` is kept in
    # the signature for backward compatibility with existing callers.
    _ = default_max_tokens

    tools = _convert_tools(request.tools)
    if tools:
        body["tools"] = tools
        tool_choice = _convert_tool_choice(request.tool_choice)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    if _supports_reasoning(request.model):
        body["reasoning"] = {
            "effort": CHATGPT_DEFAULT_REASONING_EFFORT,
            "summary": CHATGPT_DEFAULT_REASONING_SUMMARY,
        }
        body["include"] = ["reasoning.encrypted_content"]

    return body


def chatgpt_tool_call_to_anthropic(
    item: dict[str, Any],
    *,
    tool_name_override: str | None = None,
) -> dict[str, Any]:
    """Convert one ChatGPT function_call item to an Anthropic tool_use block."""
    name = item.get("name") or tool_name_override or "unknown"
    arguments = item.get("arguments") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    try:
        input_data = json.loads(arguments)
    except json.JSONDecodeError:
        input_data = {"raw": arguments}
    return {
        "type": "tool_use",
        "id": item.get("id", ""),
        "name": name,
        "input": input_data,
    }

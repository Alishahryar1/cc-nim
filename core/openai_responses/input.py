"""Convert OpenAI Responses requests into Anthropic Messages payloads."""

from collections.abc import Mapping
from typing import Any

from core.trace import trace_event

from .errors import ResponsesConversionError
from .reasoning import (
    combine_reasoning,
    reasoning_text_from_item,
    responses_reasoning_to_thinking,
)
from .tools import (
    call_id_from_item,
    convert_tool_choice,
    convert_tools,
    custom_tool_input_to_anthropic,
    optional_str,
    parse_arguments,
    required_str,
    responses_tool_name_to_anthropic_name,
)


def convert_request_to_anthropic_payload(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert an OpenAI Responses request into an Anthropic Messages payload."""

    system_parts: list[str] = []
    if instructions := optional_str(request.get("instructions")):
        system_parts.append(instructions)

    messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None
    quarantined_function_call_ids: set[str] = set()
    for item in _iter_input_items(request.get("input")):
        pending_reasoning = _append_input_item(
            item,
            messages=messages,
            system_parts=system_parts,
            pending_reasoning=pending_reasoning,
            quarantined_function_call_ids=quarantined_function_call_ids,
        )
    _append_pending_reasoning(messages, pending_reasoning)

    if not messages:
        raise ResponsesConversionError("Responses request input must contain a message")

    payload: dict[str, Any] = {
        "model": required_str(request.get("model"), "model"),
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

    if thinking := responses_reasoning_to_thinking(request.get("reasoning")):
        payload["thinking"] = thinking

    raw_tool_choice = request.get("tool_choice")
    tools = convert_tools(request.get("tools"))
    if tools and raw_tool_choice != "none":
        payload["tools"] = tools
    tool_choice = convert_tool_choice(raw_tool_choice)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return payload


def _append_input_item(
    item: Any,
    *,
    messages: list[dict[str, Any]],
    system_parts: list[str],
    pending_reasoning: str | None,
    quarantined_function_call_ids: set[str],
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
        role = required_str(item.get("role", "user"), "input.role")
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
    if item_type in {"function_call", "custom_tool_call"}:
        namespace = optional_str(item.get("namespace"))
        field_name = f"{item_type}.name"
        name = required_str(item.get("name"), field_name)
        call_id = call_id_from_item(item)
        if item_type == "custom_tool_call":
            tool_input = custom_tool_input_to_anthropic(item.get("input"))
        else:
            try:
                tool_input = parse_arguments(item.get("arguments"))
            except ResponsesConversionError as exc:
                quarantined_function_call_ids.add(call_id)
                _trace_quarantined_function_call(call_id, exc)
                return pending_reasoning
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": responses_tool_name_to_anthropic_name(
                        name, namespace=namespace
                    ),
                    "input": tool_input,
                }
            ],
        }
        if pending_reasoning:
            message["reasoning_content"] = pending_reasoning
        messages.append(message)
        return None
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        call_id = call_id_from_item(item)
        if (
            item_type == "function_call_output"
            and call_id in quarantined_function_call_ids
        ):
            return pending_reasoning
        _append_pending_reasoning(messages, pending_reasoning)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": item.get("output", ""),
                    }
                ],
            }
        )
        return None
    if item_type == "reasoning":
        return combine_reasoning(pending_reasoning, reasoning_text_from_item(item))
    if item_type in {"input_text", "output_text", "text"}:
        _append_pending_reasoning(messages, pending_reasoning)
        messages.append({"role": "user", "content": _text_from_part(item)})
        return None

    raise ResponsesConversionError(
        f"Unsupported Responses input item type: {item_type!r}"
    )


def _trace_quarantined_function_call(
    call_id: str, exc: ResponsesConversionError
) -> None:
    trace_event(
        stage="responses",
        event="responses.input.function_call_quarantined",
        source="openai_responses",
        call_id=call_id,
        error_type=type(exc).__name__,
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
    if text := optional_str(part.get("text")):
        return text
    if text := optional_str(part.get("input_text")):
        return text
    if text := optional_str(part.get("output_text")):
        return text
    return ""


def _copy_if_present(
    source: Mapping[str, Any], target: dict[str, Any], field_name: str
) -> None:
    if source.get(field_name) is not None:
        target[field_name] = source[field_name]

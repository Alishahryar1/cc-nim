"""Convert OpenAI Chat Completions requests into Anthropic Messages payloads."""

from collections.abc import Mapping
from typing import Any

from .errors import ChatCompletionsConversionError
from .ids import new_tool_call_id
from .models import OpenAIChatCompletionsRequest
from .tools import convert_tool_choice, convert_tools, parse_arguments


def convert_request_to_anthropic_payload(
    request: OpenAIChatCompletionsRequest,
) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions request into an Anthropic Messages payload."""
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    for raw in request.messages:
        if not isinstance(raw, dict):
            raise ChatCompletionsConversionError(
                f"Unsupported chat message: {type(raw).__name__}"
            )
        role = raw.get("role")
        if role in {"system", "developer"}:
            if text := _content_as_text(raw.get("content")):
                system_parts.append(text)
        elif role == "user":
            messages.append(
                {"role": "user", "content": _convert_user_content(raw.get("content"))}
            )
        elif role == "assistant":
            _append_assistant_message(messages, raw)
        elif role == "tool":
            _append_tool_message(messages, raw)
        else:
            raise ChatCompletionsConversionError(
                f"Unsupported chat message role: {role!r}"
            )

    if not messages:
        raise ChatCompletionsConversionError(
            "Chat request must include at least one non-system message"
        )

    payload: dict[str, Any] = {
        "model": _required_model(request.model),
        "messages": messages,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    max_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if request.metadata is not None:
        payload["metadata"] = request.metadata
    if stop_sequences := _normalize_stop(request.stop):
        payload["stop_sequences"] = stop_sequences

    tools = convert_tools(request.tools)
    # Validate tool_choice unconditionally so an invalid value is a client error
    # even when no tools are present, matching the Responses dialect.
    tool_choice = convert_tool_choice(request.tool_choice)
    if tools and request.tool_choice != "none":
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    return payload


def _append_assistant_message(
    messages: list[dict[str, Any]], raw: Mapping[str, Any]
) -> None:
    blocks: list[dict[str, Any]] = list(_convert_assistant_content(raw.get("content")))
    for tool_call in raw.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        function = function if isinstance(function, dict) else {}
        blocks.append(
            {
                "type": "tool_use",
                "id": _optional_str(tool_call.get("id")) or new_tool_call_id(),
                "name": _optional_str(function.get("name")),
                "input": parse_arguments(function.get("arguments")),
            }
        )
    if not blocks:
        # An assistant turn with neither text nor tool calls still needs content.
        messages.append({"role": "assistant", "content": ""})
        return
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        messages.append({"role": "assistant", "content": blocks[0]["text"]})
        return
    messages.append({"role": "assistant", "content": blocks})


def _append_tool_message(
    messages: list[dict[str, Any]], raw: Mapping[str, Any]
) -> None:
    tool_call_id = raw.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ChatCompletionsConversionError(
            "tool message requires a string tool_call_id"
        )
    block = {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": _content_as_text(raw.get("content")),
    }
    pending = _last_user_tool_result_message(messages)
    if pending is not None:
        pending["content"].append(block)
        return
    messages.append({"role": "user", "content": [block]})


def _last_user_tool_result_message(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not messages:
        return None
    message = messages[-1]
    if message.get("role") != "user":
        return None
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return None
    if all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return message
    return None


def _convert_user_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                raise ChatCompletionsConversionError(
                    f"Unsupported content part: {type(part).__name__}"
                )
            part_type = part.get("type")
            if part_type == "image_url":
                blocks.append(_convert_image_part(part.get("image_url")))
                continue
            if part_type in {"text", "input_text", "output_text"} or "text" in part:
                blocks.append({"type": "text", "text": _text_from_part(part)})
                continue
            raise ChatCompletionsConversionError(
                f"Unsupported content part type: {part_type!r}"
            )
        return blocks
    raise ChatCompletionsConversionError(
        f"Unsupported message content: {type(content).__name__}"
    )


def _convert_assistant_content(content: Any) -> list[dict[str, Any]]:
    if content in (None, ""):
        return []
    converted = _convert_user_content(content)
    if isinstance(converted, str):
        return [{"type": "text", "text": converted}] if converted else []
    return [block for block in converted if block.get("type") == "text"]


def _content_as_text(content: Any) -> str:
    converted = _convert_user_content(content)
    if isinstance(converted, str):
        return converted
    return "\n".join(
        str(block.get("text", "")) for block in converted if block.get("type") == "text"
    )


def _convert_image_part(image_url: Any) -> dict[str, Any]:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        raise ChatCompletionsConversionError("image_url content part requires a url")
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[len("data:") :].split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _normalize_stop(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [item for item in stop if isinstance(item, str)]
    return []


def _text_from_part(part: Mapping[str, Any]) -> str:
    for key in ("text", "input_text", "output_text"):
        value = part.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _optional_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _required_model(model: Any) -> str:
    if not isinstance(model, str) or not model:
        raise ChatCompletionsConversionError("model is required")
    return model

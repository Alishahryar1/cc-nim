"""Translation between OpenAI Chat Completions and Anthropic Messages.

FCC's internal pipeline speaks Anthropic Messages end to end. Serving the
Chat Completions protocol is therefore a pure edge translation: inbound
requests are lowered into ``MessagesRequest`` and the resulting Anthropic
response (JSON or SSE) is lifted back into Chat Completions shape. Nothing in
routing, provider selection, reasoning policy or streaming recovery changes.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any

from free_claude_code.core.anthropic import MessagesRequest

from .models import ChatCompletionsRequest, ChatMessage

_DEFAULT_MAX_TOKENS = 4096

_STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def new_completion_id() -> str:
    """Return an OpenAI-shaped completion id."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# --------------------------------------------------------------------------
# Request: Chat Completions -> Anthropic Messages
# --------------------------------------------------------------------------


def _image_source_from_url(url: str) -> dict[str, Any]:
    """Return an Anthropic image source for a data: or http(s): URL."""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        media_type = header[5:].split(";")[0] or "image/png"
        return {"type": "base64", "media_type": media_type, "data": payload}
    return {"type": "url", "url": url}


def _content_parts_to_blocks(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lower OpenAI multimodal content parts into Anthropic content blocks."""
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("text", "input_text"):
            text = part.get("text") or ""
            if text:
                blocks.append({"type": "text", "text": text})
        elif part_type in ("image_url", "input_image"):
            raw = part.get("image_url") or part.get("image")
            url = raw.get("url") if isinstance(raw, dict) else raw
            if isinstance(url, str) and url:
                blocks.append(
                    {"type": "image", "source": _image_source_from_url(url)}
                )
    return blocks


def _text_of(content: str | list[dict[str, Any]] | None) -> str:
    """Flatten message content down to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return ""


def _assistant_blocks(message: ChatMessage) -> list[dict[str, Any]]:
    """Build Anthropic assistant content, including tool_use blocks."""
    blocks: list[dict[str, Any]] = []
    text = _text_of(message.content)
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.tool_calls or []:
        raw_args = call.function.arguments or "{}"
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except (TypeError, ValueError):
            # A malformed argument string must not abort the whole request;
            # forward it verbatim so the model can still see what it emitted.
            parsed = {"_raw_arguments": raw_args}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id or f"call_{uuid.uuid4().hex[:16]}",
                "name": call.function.name or "unknown_tool",
                "input": parsed,
            }
        )
    return blocks


def _tools_to_anthropic(request: ChatCompletionsRequest) -> list[dict[str, Any]] | None:
    tools = [
        {
            "name": tool.function.name,
            "description": tool.function.description or "",
            "input_schema": tool.function.parameters
            or {"type": "object", "properties": {}},
        }
        for tool in (request.tools or [])
        if tool.function and tool.function.name
    ]
    return tools or None


def _tool_choice_to_anthropic(choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        return {
            "auto": {"type": "auto"},
            "required": {"type": "any"},
            "any": {"type": "any"},
            "none": {"type": "none"},
        }.get(choice)
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            name = (choice.get("function") or {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        # already Anthropic-shaped
        if choice.get("type") in ("auto", "any", "tool", "none"):
            return choice
    return None


def _structured_output_tool(
    response_format: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Map ``response_format={"type":"json_schema"}`` onto a forced tool call.

    Instructor -- and therefore Atomic Agents -- uses this to get structured
    output. Anthropic has no ``response_format``, so the equivalent is a single
    tool the model is forced to call.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    schema_block = response_format.get("json_schema")
    if not isinstance(schema_block, dict):
        return None
    schema = schema_block.get("schema")
    if not isinstance(schema, dict):
        return None
    name = schema_block.get("name") or "structured_output"
    tool = {
        "name": name,
        "description": schema_block.get("description")
        or "Return the result using this schema.",
        "input_schema": schema,
    }
    return tool, {"type": "tool", "name": name}


def chat_request_to_messages_request(
    request: ChatCompletionsRequest,
) -> MessagesRequest:
    """Lower an inbound Chat Completions request into a ``MessagesRequest``."""
    system_chunks: list[str] = []
    messages: list[dict[str, Any]] = []

    def flush_tool_results(buffer: list[dict[str, Any]]) -> None:
        if buffer:
            messages.append({"role": "user", "content": list(buffer)})
            buffer.clear()

    pending_tool_results: list[dict[str, Any]] = []

    for message in request.messages:
        if message.role in ("system", "developer"):
            flush_tool_results(pending_tool_results)
            text = _text_of(message.content)
            if text:
                system_chunks.append(text)
            continue

        if message.role in ("tool", "function"):
            # Parallel tool results must be batched into one user turn.
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id
                    or f"call_{uuid.uuid4().hex[:16]}",
                    "content": _text_of(message.content) or "",
                }
            )
            continue

        flush_tool_results(pending_tool_results)

        if message.role == "assistant":
            blocks = _assistant_blocks(message)
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue

        # user
        if isinstance(message.content, list):
            blocks = _content_parts_to_blocks(message.content)
            messages.append(
                {"role": "user", "content": blocks or [{"type": "text", "text": ""}]}
            )
        else:
            messages.append({"role": "user", "content": message.content or ""})

    flush_tool_results(pending_tool_results)

    if not messages:
        messages.append({"role": "user", "content": ""})

    tools = _tools_to_anthropic(request)
    tool_choice = _tool_choice_to_anthropic(request.tool_choice)

    structured = _structured_output_tool(request.response_format)
    if structured is not None:
        forced_tool, forced_choice = structured
        tools = [*(tools or []), forced_tool]
        tool_choice = forced_choice

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.effective_max_tokens(_DEFAULT_MAX_TOKENS),
        "stream": request.stream,
    }
    if system_chunks:
        payload["system"] = "\n\n".join(system_chunks)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if (stops := request.stop_sequences()) is not None:
        payload["stop_sequences"] = stops
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if request.metadata:
        payload["metadata"] = request.metadata

    return MessagesRequest.model_validate(payload)


# --------------------------------------------------------------------------
# Response: Anthropic Messages -> Chat Completions
# --------------------------------------------------------------------------


def _usage_to_openai(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    payload: dict[str, Any] = {
        "prompt_tokens": prompt + cache_read,
        "completion_tokens": completion,
        "total_tokens": prompt + cache_read + completion,
    }
    if cache_read:
        payload["prompt_tokens_details"] = {"cached_tokens": cache_read}
    return payload


def anthropic_message_to_chat_completion(
    message: dict[str, Any],
    *,
    model: str,
    completion_id: str | None = None,
) -> dict[str, Any]:
    """Lift a complete Anthropic Message into a Chat Completions response."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text") or "")
        elif block_type == "thinking":
            reasoning_parts.append(block.get("thinking") or "")
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "unknown_tool",
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    chat_message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if reasoning_parts:
        chat_message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        chat_message["tool_calls"] = tool_calls

    stop_reason = message.get("stop_reason")
    finish_reason = _STOP_REASON_TO_FINISH_REASON.get(str(stop_reason), "stop")
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": message.get("model") or model,
        "choices": [
            {"index": 0, "message": chat_message, "finish_reason": finish_reason}
        ],
        "usage": _usage_to_openai(message.get("usage")),
    }


# --------------------------------------------------------------------------
# Streaming: Anthropic SSE -> Chat Completions SSE
# --------------------------------------------------------------------------


def iter_sse_data(raw: str, buffer: list[str]) -> Iterable[dict[str, Any]]:
    """Yield decoded ``data:`` payloads from a raw SSE fragment.

    ``buffer`` is a single-element carry list holding the incomplete tail
    between fragments, since a provider chunk may split mid-line.
    """
    buffer[0] += raw
    while "\n" in buffer[0]:
        line, _, rest = buffer[0].partition("\n")
        buffer[0] = rest
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            yield decoded


def _chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def anthropic_sse_to_chat_sse(
    source: AsyncIterator[str],
    *,
    model: str,
    completion_id: str,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Transform an Anthropic SSE stream into a Chat Completions SSE stream."""
    buffer = [""]
    # Anthropic indexes every content block; OpenAI indexes only tool calls.
    tool_index_by_block: dict[int, int] = {}
    next_tool_index = 0
    finish_reason = "stop"
    usage: dict[str, Any] = {}
    role_sent = False
    response_model = model
    saw_tool_call = False

    async for raw in source:
        for event in iter_sse_data(raw, buffer):
            event_type = event.get("type")

            if event_type == "message_start":
                message = event.get("message") or {}
                response_model = message.get("model") or response_model
                usage.update(message.get("usage") or {})
                if not role_sent:
                    role_sent = True
                    yield _chunk(
                        completion_id, response_model, {"role": "assistant", "content": ""}
                    )

            elif event_type == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    saw_tool_call = True
                    block_index = int(event.get("index") or 0)
                    tool_index_by_block[block_index] = next_tool_index
                    yield _chunk(
                        completion_id,
                        response_model,
                        {
                            "tool_calls": [
                                {
                                    "index": next_tool_index,
                                    "id": block.get("id")
                                    or f"call_{uuid.uuid4().hex[:16]}",
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name") or "unknown_tool",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                    )
                    next_tool_index += 1

            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text") or ""
                    if text:
                        yield _chunk(completion_id, response_model, {"content": text})
                elif delta_type == "thinking_delta":
                    thinking = delta.get("thinking") or ""
                    if thinking:
                        yield _chunk(
                            completion_id,
                            response_model,
                            {"reasoning_content": thinking},
                        )
                elif delta_type == "input_json_delta":
                    partial = delta.get("partial_json") or ""
                    block_index = int(event.get("index") or 0)
                    tool_index = tool_index_by_block.get(block_index, 0)
                    if partial:
                        yield _chunk(
                            completion_id,
                            response_model,
                            {
                                "tool_calls": [
                                    {
                                        "index": tool_index,
                                        "function": {"arguments": partial},
                                    }
                                ]
                            },
                        )

            elif event_type == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason")
                if stop_reason:
                    finish_reason = _STOP_REASON_TO_FINISH_REASON.get(
                        str(stop_reason), "stop"
                    )
                usage.update(event.get("usage") or {})

            elif event_type == "error":
                # Surface a terminal frame the OpenAI SDK can classify.
                error = event.get("error") or {}
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "message": error.get("message")
                                or "Provider request failed.",
                                "type": error.get("type") or "api_error",
                            }
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
                return

    if saw_tool_call and finish_reason == "stop":
        finish_reason = "tool_calls"

    yield _chunk(
        completion_id,
        response_model,
        {},
        finish_reason=finish_reason,
        usage=_usage_to_openai(usage) if include_usage else None,
    )
    yield "data: [DONE]\n\n"

"""Translate Anthropic-style SSE streams into OpenAI Responses SSE streams."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .conversion import (
    anthropic_message_response_to_openai_response,
    responses_tool_identity_from_anthropic_name,
)

OPENAI_RESPONSES_SSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def format_response_sse_event(event_type: str, data: Mapping[str, Any]) -> str:
    """Format one OpenAI Responses SSE event."""

    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def iter_anthropic_sse_as_openai_responses(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> AsyncIterator[str]:
    """Yield Responses SSE events translated from an Anthropic SSE stream."""

    transformer = _ResponsesStreamTransformer(request)
    async for event in _iter_sse_events(chunks):
        for chunk in transformer.process_anthropic_event(event):
            yield chunk
        if transformer.terminal:
            return
    for chunk in transformer.finish_if_needed():
        yield chunk


async def collect_openai_response_from_anthropic_sse(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect a translated Anthropic SSE stream into one Responses object."""

    transformer = _ResponsesStreamTransformer(request)
    async for event in _iter_sse_events(chunks):
        transformer.process_anthropic_event(event)
        if transformer.terminal:
            break
    if transformer.final_response is not None:
        return transformer.final_response
    transformer.finish_if_needed()
    return transformer.final_response or transformer.response_payload(
        status="completed"
    )


def iter_message_response_as_openai_responses(
    message: Mapping[str, Any],
    request: Mapping[str, Any],
) -> list[str]:
    """Return Responses SSE chunks for a non-stream Anthropic message response."""

    response_id = _new_response_id()
    response = _base_response(request, response_id=response_id, status="in_progress")
    chunks = [
        format_response_sse_event(
            "response.created",
            {"type": "response.created", "response": response},
        )
    ]
    completed = anthropic_message_response_to_openai_response(
        message, request, response_id=response_id
    )
    for output_index, item in enumerate(completed["output"]):
        chunks.append(
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": _in_progress_item(item),
                },
            )
        )
        if item.get("type") == "message":
            text = _message_item_text(item)
            chunks.extend(_message_text_events(item, output_index, text))
        elif item.get("type") == "reasoning":
            text = _reasoning_item_text(item)
            if text:
                chunks.extend(_reasoning_text_events(item, output_index, text))
        elif item.get("type") == "function_call":
            arguments = str(item.get("arguments", ""))
            if arguments:
                chunks.append(
                    format_response_sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "delta": arguments,
                        },
                    )
                )
            chunks.append(
                format_response_sse_event(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "arguments": arguments,
                    },
                )
            )
        chunks.append(
            format_response_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
        )
    chunks.append(
        format_response_sse_event(
            "response.completed",
            {"type": "response.completed", "response": completed},
        )
    )
    return chunks


@dataclass(slots=True)
class _AnthropicSseEvent:
    event: str
    data: dict[str, Any]


@dataclass(slots=True)
class _ToolState:
    output_index: int
    item_id: str
    call_id: str
    name: str
    namespace: str | None = None
    argument_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ReasoningState:
    output_index: int
    item_id: str
    text_parts: list[str] = field(default_factory=list)
    encrypted_content: str | None = None


class _ResponsesStreamTransformer:
    def __init__(self, request: Mapping[str, Any]) -> None:
        self._request = request
        self._response_id = _new_response_id()
        self._created_at = int(time.time())
        self._output: list[dict[str, Any]] = []
        self._text_item_id = _new_message_item_id()
        self._text_started = False
        self._text_done = False
        self._text_output_index: int | None = None
        self._text_block_indexes: set[int] = set()
        self._text_parts: list[str] = []
        self._tools_by_block_index: dict[int, _ToolState] = {}
        self._reasoning_by_block_index: dict[int, _ReasoningState] = {}
        self._stop_reason: str | None = None
        self._usage: dict[str, int] | None = None
        self._started = False
        self.terminal = False
        self.final_response: dict[str, Any] | None = None

    def process_anthropic_event(self, event: _AnthropicSseEvent) -> list[str]:
        if self.terminal:
            return []

        chunks = self._ensure_started()
        if event.event == "content_block_start":
            chunks.extend(self._handle_content_block_start(event.data))
        elif event.event == "content_block_delta":
            chunks.extend(self._handle_content_block_delta(event.data))
        elif event.event == "content_block_stop":
            chunks.extend(self._handle_content_block_stop(event.data))
        elif event.event == "message_delta":
            self._handle_message_delta(event.data)
        elif event.event == "message_stop":
            chunks.extend(self._complete_response())
        elif event.event == "error":
            chunks.extend(self._error_event(event.data))
        return chunks

    def finish_if_needed(self) -> list[str]:
        if self.terminal:
            return []
        chunks = self._ensure_started()
        chunks.extend(self._complete_response())
        return chunks

    def response_payload(self, *, status: str) -> dict[str, Any]:
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": str(self._request.get("model", "")),
            "output": list(self._output),
            "parallel_tool_calls": bool(self._request.get("parallel_tool_calls", True)),
            "tool_choice": self._request.get("tool_choice", "auto"),
            "temperature": self._request.get("temperature"),
            "top_p": self._request.get("top_p"),
            "max_output_tokens": self._request.get("max_output_tokens"),
            "usage": self._usage,
            "error": None if status == "completed" else {},
        }

    def _ensure_started(self) -> list[str]:
        if self._started:
            return []
        self._started = True
        return [
            format_response_sse_event(
                "response.created",
                {
                    "type": "response.created",
                    "response": self.response_payload(status="in_progress"),
                },
            )
        ]

    def _handle_content_block_start(self, data: Mapping[str, Any]) -> list[str]:
        block = data.get("content_block")
        if not isinstance(block, dict):
            return []
        block_type = block.get("type")
        if block_type == "text":
            index = _event_index(data)
            if index is not None:
                self._text_block_indexes.add(index)
            chunks = self._ensure_text_started()
            if text := str(block.get("text", "")):
                chunks.extend(self._emit_text_delta(text))
            return chunks
        if block_type == "thinking":
            index = _event_index(data)
            if index is None:
                return []
            state = self._start_reasoning_block(index)
            chunks = self._reasoning_item_added(state)
            if text := str(block.get("thinking", "")):
                chunks.extend(self._emit_reasoning_delta(state, text))
            return chunks
        if block_type == "redacted_thinking":
            index = _event_index(data)
            if index is None:
                return []
            state = self._start_reasoning_block(
                index, encrypted_content=str(block.get("data", ""))
            )
            return self._reasoning_item_added(state)
        if block_type == "tool_use":
            index = _event_index(data)
            if index is None:
                return []
            call_id = str(block.get("id", "") or _new_call_id())
            item_id = f"fc_{uuid.uuid4().hex[:24]}"
            output_index = len(self._output)
            namespace, name = responses_tool_identity_from_anthropic_name(
                self._request, str(block.get("name", ""))
            )
            state = _ToolState(
                output_index=output_index,
                item_id=item_id,
                call_id=call_id,
                name=name,
                namespace=namespace,
            )
            initial_input = block.get("input")
            if isinstance(initial_input, dict) and initial_input:
                state.argument_parts.append(json.dumps(initial_input))
            self._tools_by_block_index[index] = state
        return []

    def _handle_content_block_delta(self, data: Mapping[str, Any]) -> list[str]:
        delta = data.get("delta")
        if not isinstance(delta, dict):
            return []
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            return self._emit_text_delta(str(delta.get("text", "")))
        if delta_type == "thinking_delta":
            index = _event_index(data)
            if index is None:
                return []
            state = self._reasoning_by_block_index.get(index)
            if state is None:
                state = self._start_reasoning_block(index)
                return [
                    *self._reasoning_item_added(state),
                    *self._emit_reasoning_delta(state, str(delta.get("thinking", ""))),
                ]
            return self._emit_reasoning_delta(state, str(delta.get("thinking", "")))
        if delta_type == "input_json_delta":
            index = _event_index(data)
            if index is not None and index in self._tools_by_block_index:
                self._tools_by_block_index[index].argument_parts.append(
                    str(delta.get("partial_json", ""))
                )
        return []

    def _handle_content_block_stop(self, data: Mapping[str, Any]) -> list[str]:
        index = _event_index(data)
        if index is None:
            return []
        if index in self._text_block_indexes:
            self._text_block_indexes.remove(index)
            return self._complete_text_if_needed()
        reasoning_state = self._reasoning_by_block_index.pop(index, None)
        if reasoning_state is not None:
            return self._complete_reasoning(reasoning_state)
        return self._handle_tool_block_stop(index)

    def _start_reasoning_block(
        self, index: int, *, encrypted_content: str | None = None
    ) -> _ReasoningState:
        state = _ReasoningState(
            output_index=len(self._output),
            item_id=_new_reasoning_item_id(),
            encrypted_content=encrypted_content,
        )
        self._reasoning_by_block_index[index] = state
        return state

    def _reasoning_item_added(self, state: _ReasoningState) -> list[str]:
        item = _reasoning_output_item(state, status="in_progress")
        return [
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": state.output_index,
                    "item": item,
                },
            )
        ]

    def _emit_reasoning_delta(self, state: _ReasoningState, text: str) -> list[str]:
        if not text:
            return []
        state.text_parts.append(text)
        return [
            format_response_sse_event(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        ]

    def _complete_reasoning(self, state: _ReasoningState) -> list[str]:
        item = _reasoning_output_item(state, status="completed")
        self._output.append(item)
        chunks: list[str] = []
        text = "".join(state.text_parts)
        if text:
            chunks.append(
                format_response_sse_event(
                    "response.reasoning_text.done",
                    {
                        "type": "response.reasoning_text.done",
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
        chunks.append(
            format_response_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": state.output_index,
                    "item": item,
                },
            )
        )
        return chunks

    def _handle_tool_block_stop(self, index: int) -> list[str]:
        state = self._tools_by_block_index.pop(index, None)
        if state is None:
            return []
        return self._complete_tool_call(state)

    def _handle_message_delta(self, data: Mapping[str, Any]) -> None:
        delta = data.get("delta")
        if isinstance(delta, dict):
            self._stop_reason = str(delta.get("stop_reason") or "")
        usage = data.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            safe_in = input_tokens if isinstance(input_tokens, int) else 0
            safe_out = output_tokens if isinstance(output_tokens, int) else 0
            self._usage = {
                "input_tokens": safe_in,
                "output_tokens": safe_out,
                "total_tokens": safe_in + safe_out,
            }

    def _ensure_text_started(self) -> list[str]:
        if self._text_started:
            return []
        if self._text_done:
            self._text_item_id = _new_message_item_id()
            self._text_done = False
            self._text_parts = []
        self._text_started = True
        output_index = len(self._output)
        self._text_output_index = output_index
        item = {
            "id": self._text_item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return [
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                },
            ),
            format_response_sse_event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": self._text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            ),
        ]

    def _emit_text_delta(self, text: str) -> list[str]:
        if not text:
            return []
        chunks = self._ensure_text_started()
        self._text_parts.append(text)
        output_index = self._current_text_output_index()
        chunks.append(
            format_response_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": self._text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
        return chunks

    def _complete_text_if_needed(self) -> list[str]:
        if not self._text_started or self._text_done:
            return []
        self._text_done = True
        self._text_started = False
        text = "".join(self._text_parts)
        output_index = self._current_text_output_index()
        item = _message_item(self._text_item_id, text, "completed")
        if output_index >= len(self._output):
            self._output.append(item)
        else:
            self._output.insert(output_index, item)
        return [
            format_response_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": self._text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                },
            ),
            format_response_sse_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": self._text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                },
            ),
            format_response_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            ),
        ]

    def _complete_tool_call(self, state: _ToolState) -> list[str]:
        arguments = "".join(state.argument_parts) or "{}"
        item = {
            "id": state.item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": state.call_id,
            "name": state.name,
            "arguments": arguments,
        }
        if state.namespace:
            item["namespace"] = state.namespace
        self._output.append(item)
        chunks = [
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": state.output_index,
                    "item": {**item, "status": "in_progress", "arguments": ""},
                },
            )
        ]
        if arguments:
            chunks.append(
                format_response_sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "delta": arguments,
                    },
                )
            )
        chunks.extend(
            [
                format_response_sse_event(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "arguments": arguments,
                    },
                ),
                format_response_sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": state.output_index,
                        "item": item,
                    },
                ),
            ]
        )
        return chunks

    def _complete_response(self) -> list[str]:
        chunks = self._complete_text_if_needed()
        for index in list(self._reasoning_by_block_index):
            state = self._reasoning_by_block_index.pop(index)
            chunks.extend(self._complete_reasoning(state))
        self.final_response = self.response_payload(status="completed")
        if self._stop_reason:
            self.final_response["stop_reason"] = self._stop_reason
        chunks.append(
            format_response_sse_event(
                "response.completed",
                {"type": "response.completed", "response": self.final_response},
            )
        )
        self.terminal = True
        return chunks

    def _error_event(self, data: Mapping[str, Any]) -> list[str]:
        error = data.get("error")
        if not isinstance(error, dict):
            error = {"type": "api_error", "message": str(data)}
        self.final_response = self.response_payload(status="failed")
        self.final_response["error"] = {
            "message": str(error.get("message", "")),
            "type": str(error.get("type", "api_error")),
            "param": None,
            "code": None,
        }
        self.terminal = True
        return [
            format_response_sse_event(
                "error",
                {"type": "error", "error": self.final_response["error"]},
            )
        ]

    def _current_text_output_index(self) -> int:
        if self._text_output_index is None:
            self._text_output_index = len(self._output)
        return self._text_output_index


async def _iter_sse_events(
    chunks: AsyncIterable[Any],
) -> AsyncIterator[_AnthropicSseEvent]:
    buffer = ""
    async for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        else:
            buffer += str(chunk)

        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            event = _parse_sse_event(raw)
            if event is not None:
                yield event

    if buffer.strip():
        event = _parse_sse_event(buffer)
        if event is not None:
            yield event


def _parse_sse_event(raw: str) -> _AnthropicSseEvent | None:
    event_type = ""
    data_parts: list[str] = []
    for line in raw.splitlines():
        stripped = line.rstrip("\r")
        if stripped.startswith("event:"):
            event_type = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("data:"):
            data_parts.append(stripped.split(":", 1)[1].strip())
    if not event_type and not data_parts:
        return None
    data_text = "\n".join(data_parts)
    if data_text == "[DONE]":
        return None
    try:
        parsed = json.loads(data_text) if data_text else {}
    except json.JSONDecodeError:
        parsed = {"raw": data_text}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return _AnthropicSseEvent(event=event_type, data=parsed)


def _event_index(data: Mapping[str, Any]) -> int | None:
    value = data.get("index")
    return value if isinstance(value, int) else None


def _base_response(
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


def _message_text_events(
    item: Mapping[str, Any], output_index: int, text: str
) -> list[str]:
    item_id = str(item.get("id", ""))
    return [
        format_response_sse_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
        format_response_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        format_response_sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        ),
        format_response_sse_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        ),
    ]


def _reasoning_text_events(
    item: Mapping[str, Any], output_index: int, text: str
) -> list[str]:
    item_id = str(item.get("id", ""))
    return [
        format_response_sse_event(
            "response.reasoning_text.delta",
            {
                "type": "response.reasoning_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        format_response_sse_event(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        ),
    ]


def _message_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning_output_item(state: _ReasoningState, *, status: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": state.item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
    }
    if state.encrypted_content is not None:
        item["encrypted_content"] = state.encrypted_content
        return item
    item["content"] = [{"type": "reasoning_text", "text": "".join(state.text_parts)}]
    return item


def _message_item_text(item: Mapping[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    return "".join(parts)


def _reasoning_item_text(item: Mapping[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "reasoning_text"
    ]
    return "".join(parts)


def _in_progress_item(item: Mapping[str, Any]) -> dict[str, Any]:
    clone = dict(item)
    clone["status"] = "in_progress"
    if clone.get("type") == "message":
        clone["content"] = []
    if clone.get("type") == "reasoning" and "content" in clone:
        clone["content"] = []
    if clone.get("type") == "function_call":
        clone["arguments"] = ""
    return clone


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _new_message_item_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _new_reasoning_item_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"

"""Translate Anthropic-style SSE streams into OpenAI Responses SSE streams."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .anthropic_sse import AnthropicSseEvent, iter_sse_events
from .events import format_response_sse_event
from .ids import (
    new_call_id,
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
)
from .items import (
    base_response,
    custom_tool_call_item,
    encrypted_reasoning_item,
    in_progress_item,
    message_item,
    message_item_text,
    reasoning_item,
    reasoning_item_text,
)
from .output import convert_message_to_response
from .tools import (
    custom_tool_input_text_from_arguments,
    responses_tool_identity_from_anthropic_name,
)


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> AsyncIterator[str]:
    """Yield Responses SSE events translated from an Anthropic SSE stream."""

    transformer = _ResponsesStreamTransformer(request)
    async for event in iter_sse_events(chunks):
        for chunk in transformer.process_anthropic_event(event):
            yield chunk
        if transformer.terminal:
            return
    for chunk in transformer.finish_if_needed():
        yield chunk


async def collect_response_from_anthropic_sse(
    chunks: AsyncIterable[Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect a translated Anthropic SSE stream into one Responses object."""

    transformer = _ResponsesStreamTransformer(request)
    async for event in iter_sse_events(chunks):
        transformer.process_anthropic_event(event)
        if transformer.terminal:
            break
    if transformer.final_response is not None:
        return transformer.final_response
    transformer.finish_if_needed()
    return transformer.final_response or transformer.response_payload(
        status="completed"
    )


def iter_responses_sse_from_message(
    message: Mapping[str, Any],
    request: Mapping[str, Any],
) -> list[str]:
    """Return Responses SSE chunks for a non-stream Anthropic message response."""

    response_id = new_response_id()
    response = base_response(request, response_id=response_id, status="in_progress")
    chunks = [
        format_response_sse_event(
            "response.created",
            {"type": "response.created", "response": response},
        )
    ]
    completed = convert_message_to_response(message, request, response_id=response_id)
    for output_index, item in enumerate(completed["output"]):
        chunks.append(
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": in_progress_item(item),
                },
            )
        )
        if item.get("type") == "message":
            text = message_item_text(item)
            chunks.extend(_message_text_events(item, output_index, text))
        elif item.get("type") == "reasoning":
            text = reasoning_item_text(item)
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
        elif item.get("type") == "custom_tool_call":
            input_text = str(item.get("input", ""))
            if input_text:
                chunks.append(
                    format_response_sse_event(
                        "response.custom_tool_call_input.delta",
                        {
                            "type": "response.custom_tool_call_input.delta",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "delta": input_text,
                        },
                    )
                )
            chunks.append(
                format_response_sse_event(
                    "response.custom_tool_call_input.done",
                    {
                        "type": "response.custom_tool_call_input.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "input": input_text,
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
class _ToolState:
    output_index: int
    item_id: str
    call_id: str
    kind: str
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
        self._response_id = new_response_id()
        self._created_at = int(time.time())
        self._output: list[dict[str, Any]] = []
        self._text_item_id = new_message_item_id()
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

    def process_anthropic_event(self, event: AnthropicSseEvent) -> list[str]:
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
            call_id = str(block.get("id", "") or new_call_id())
            identity = responses_tool_identity_from_anthropic_name(
                self._request, str(block.get("name", ""))
            )
            item_prefix = "ctc" if identity.kind == "custom" else "fc"
            item_id = f"{item_prefix}_{uuid.uuid4().hex[:24]}"
            output_index = len(self._output)
            state = _ToolState(
                output_index=output_index,
                item_id=item_id,
                call_id=call_id,
                kind=identity.kind,
                name=identity.name,
                namespace=identity.namespace,
            )
            initial_input = block.get("input")
            if (identity.kind == "custom" and initial_input not in (None, {}, "")) or (
                isinstance(initial_input, dict) and initial_input
            ):
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
            item_id=new_reasoning_item_id(),
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
            self._text_item_id = new_message_item_id()
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
        item = message_item(self._text_item_id, text, "completed")
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
        if state.kind == "custom":
            return self._complete_custom_tool_call(state)
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

    def _complete_custom_tool_call(self, state: _ToolState) -> list[str]:
        input_text = custom_tool_input_text_from_arguments(
            "".join(state.argument_parts)
        )
        item = custom_tool_call_item(
            block_id=state.call_id,
            name=state.name,
            namespace=state.namespace,
            input_text=input_text,
            status="completed",
        )
        item["id"] = state.item_id
        self._output.append(item)
        chunks = [
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": state.output_index,
                    "item": {**item, "status": "in_progress", "input": ""},
                },
            )
        ]
        if input_text:
            chunks.append(
                format_response_sse_event(
                    "response.custom_tool_call_input.delta",
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "delta": input_text,
                    },
                )
            )
        chunks.extend(
            [
                format_response_sse_event(
                    "response.custom_tool_call_input.done",
                    {
                        "type": "response.custom_tool_call_input.done",
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "input": input_text,
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


def _event_index(data: Mapping[str, Any]) -> int | None:
    value = data.get("index")
    return value if isinstance(value, int) else None


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


def _reasoning_output_item(state: _ReasoningState, *, status: str) -> dict[str, Any]:
    if state.encrypted_content is not None:
        return encrypted_reasoning_item(state.item_id, state.encrypted_content, status)
    return reasoning_item(state.item_id, "".join(state.text_parts), status)

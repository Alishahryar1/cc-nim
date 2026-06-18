"""Block-indexed OpenAI Responses stream assembly."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .anthropic_sse import AnthropicSseEvent
from .events import format_response_sse_event
from .ids import (
    new_call_id,
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
)
from .items import (
    encrypted_reasoning_item,
    message_item,
    reasoning_item,
)
from .tools import (
    custom_tool_input_text_from_arguments,
    responses_tool_identity_from_anthropic_name,
)


@dataclass(slots=True)
class _TextBlockState:
    index: int
    output_index: int
    item_id: str
    text_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ReasoningBlockState:
    index: int
    output_index: int
    item_id: str
    text_parts: list[str] = field(default_factory=list)
    encrypted_content: str | None = None


@dataclass(slots=True)
class _ToolBlockState:
    index: int
    output_index: int
    item_id: str
    call_id: str
    kind: Literal["function", "custom"]
    name: str
    namespace: str | None = None
    argument_parts: list[str] = field(default_factory=list)


_BlockState = _TextBlockState | _ReasoningBlockState | _ToolBlockState


class ResponsesStreamAssembler:
    """Assemble Responses SSE events from indexed Anthropic content blocks."""

    def __init__(self, request: Mapping[str, Any]) -> None:
        self._request = request
        self._response_id = new_response_id()
        self._created_at = int(time.time())
        self._output_slots: list[dict[str, Any] | None] = []
        self._active_blocks: dict[int, _BlockState] = {}
        self._fallback_text_index = -1
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
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
            chunks.extend(self.complete_response())
        elif event.event == "error":
            chunks.extend(self.fail_response(event.data))
        return chunks

    def finish_if_needed(self) -> list[str]:
        if self.terminal:
            return []
        chunks = self._ensure_started()
        chunks.extend(self.complete_response())
        return chunks

    def response_payload(
        self, *, status: str, error: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": str(self._request.get("model", "")),
            "output": self._output(),
            "parallel_tool_calls": bool(self._request.get("parallel_tool_calls", True)),
            "tool_choice": self._request.get("tool_choice", "auto"),
            "temperature": self._request.get("temperature"),
            "top_p": self._request.get("top_p"),
            "max_output_tokens": self._request.get("max_output_tokens"),
            "usage": self._usage(),
            "error": error,
        }

    def complete_response(self) -> list[str]:
        chunks = self._flush_active_blocks()
        self.final_response = self.response_payload(status="completed")
        chunks.append(
            format_response_sse_event(
                "response.completed",
                {"type": "response.completed", "response": self.final_response},
            )
        )
        self.terminal = True
        return chunks

    def fail_response(self, data: Mapping[str, Any]) -> list[str]:
        chunks = self._flush_active_blocks()
        error = _openai_error_from_anthropic_error(data)
        self.final_response = self.response_payload(status="failed", error=error)
        chunks.append(
            format_response_sse_event(
                "response.failed",
                {"type": "response.failed", "response": self.final_response},
            )
        )
        self.terminal = True
        return chunks

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
        index = _event_index(data)
        if block_type == "text":
            index = self._safe_index(index)
            chunks, state = self._start_text_block(index)
            if text := _string_value(block.get("text")):
                chunks.extend(self._emit_text_delta(state, text))
            return chunks
        if block_type == "thinking":
            if index is None:
                return []
            chunks, state = self._start_reasoning_block(index)
            if text := _string_value(block.get("thinking")):
                chunks.extend(self._emit_reasoning_delta(state, text))
            return chunks
        if block_type == "redacted_thinking":
            if index is None:
                return []
            chunks, _state = self._start_reasoning_block(
                index, encrypted_content=_string_value(block.get("data"))
            )
            return chunks
        if block_type == "tool_use":
            if index is None:
                return []
            return self._start_tool_block(index, block)
        return []

    def _handle_content_block_delta(self, data: Mapping[str, Any]) -> list[str]:
        delta = data.get("delta")
        if not isinstance(delta, dict):
            return []
        delta_type = delta.get("type")
        index = _event_index(data)
        if delta_type == "text_delta":
            index = self._safe_index(index)
            state = self._active_blocks.get(index)
            chunks: list[str] = []
            if not isinstance(state, _TextBlockState):
                chunks, state = self._start_text_block(index)
            chunks.extend(
                self._emit_text_delta(state, _string_value(delta.get("text")))
            )
            return chunks
        if delta_type == "thinking_delta":
            if index is None:
                return []
            state = self._active_blocks.get(index)
            chunks = []
            if not isinstance(state, _ReasoningBlockState):
                chunks, state = self._start_reasoning_block(index)
            chunks.extend(
                self._emit_reasoning_delta(state, _string_value(delta.get("thinking")))
            )
            return chunks
        if delta_type == "input_json_delta":
            state = self._active_blocks.get(index) if index is not None else None
            if isinstance(state, _ToolBlockState):
                state.argument_parts.append(_string_value(delta.get("partial_json")))
        return []

    def _handle_content_block_stop(self, data: Mapping[str, Any]) -> list[str]:
        index = _event_index(data)
        if index is None:
            return []
        state = self._active_blocks.pop(index, None)
        if state is None:
            return []
        return self._complete_block(state)

    def _handle_message_delta(self, data: Mapping[str, Any]) -> None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        if isinstance(usage.get("input_tokens"), int):
            self._input_tokens = usage["input_tokens"]
        if isinstance(usage.get("output_tokens"), int):
            self._output_tokens = usage["output_tokens"]

    def _start_text_block(self, index: int) -> tuple[list[str], _TextBlockState]:
        chunks = self._complete_existing_block(index)
        output_index = self._reserve_output_slot()
        state = _TextBlockState(
            index=index,
            output_index=output_index,
            item_id=new_message_item_id(),
        )
        self._active_blocks[index] = state
        item = {
            "id": state.item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        chunks.extend(
            [
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
                        "item_id": state.item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                ),
            ]
        )
        return chunks, state

    def _start_reasoning_block(
        self, index: int, *, encrypted_content: str | None = None
    ) -> tuple[list[str], _ReasoningBlockState]:
        chunks = self._complete_existing_block(index)
        output_index = self._reserve_output_slot()
        state = _ReasoningBlockState(
            index=index,
            output_index=output_index,
            item_id=new_reasoning_item_id(),
            encrypted_content=encrypted_content,
        )
        self._active_blocks[index] = state
        chunks.append(
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": _reasoning_output_item(state, status="in_progress"),
                },
            )
        )
        return chunks, state

    def _start_tool_block(self, index: int, block: Mapping[str, Any]) -> list[str]:
        chunks = self._complete_existing_block(index)
        identity = responses_tool_identity_from_anthropic_name(
            self._request, _string_value(block.get("name"))
        )
        state = _ToolBlockState(
            index=index,
            output_index=self._reserve_output_slot(),
            item_id=f"{'ctc' if identity.kind == 'custom' else 'fc'}_"
            f"{uuid.uuid4().hex[:24]}",
            call_id=_string_value(block.get("id")) or new_call_id(),
            kind=identity.kind,
            name=identity.name,
            namespace=identity.namespace,
        )
        initial_input = block.get("input")
        if (identity.kind == "custom" and initial_input not in (None, {}, "")) or (
            isinstance(initial_input, dict) and initial_input
        ):
            state.argument_parts.append(json.dumps(initial_input))
        self._active_blocks[index] = state
        chunks.append(
            format_response_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": state.output_index,
                    "item": self._tool_item(state, status="in_progress"),
                },
            )
        )
        return chunks

    def _emit_text_delta(self, state: _TextBlockState, text: str) -> list[str]:
        if not text:
            return []
        state.text_parts.append(text)
        return [
            format_response_sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        ]

    def _emit_reasoning_delta(
        self, state: _ReasoningBlockState, text: str
    ) -> list[str]:
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

    def _complete_existing_block(self, index: int) -> list[str]:
        existing = self._active_blocks.pop(index, None)
        if existing is None:
            return []
        return self._complete_block(existing)

    def _complete_block(self, state: _BlockState) -> list[str]:
        if isinstance(state, _TextBlockState):
            return self._complete_text_block(state)
        if isinstance(state, _ReasoningBlockState):
            return self._complete_reasoning_block(state)
        return self._complete_tool_block(state)

    def _complete_text_block(self, state: _TextBlockState) -> list[str]:
        text = "".join(state.text_parts)
        item = message_item(state.item_id, text, "completed")
        self._commit_output(state.output_index, item)
        return [
            format_response_sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "text": text,
                },
            ),
            format_response_sse_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
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

    def _complete_reasoning_block(self, state: _ReasoningBlockState) -> list[str]:
        item = _reasoning_output_item(state, status="completed")
        self._commit_output(state.output_index, item)
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

    def _complete_tool_block(self, state: _ToolBlockState) -> list[str]:
        if state.kind == "custom":
            return self._complete_custom_tool_block(state)
        arguments = "".join(state.argument_parts) or "{}"
        item = self._tool_item(state, status="completed", arguments=arguments)
        self._commit_output(state.output_index, item)
        chunks: list[str] = []
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

    def _complete_custom_tool_block(self, state: _ToolBlockState) -> list[str]:
        input_text = custom_tool_input_text_from_arguments(
            "".join(state.argument_parts)
        )
        item = self._tool_item(state, status="completed", input_text=input_text)
        self._commit_output(state.output_index, item)
        chunks: list[str] = []
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

    def _tool_item(
        self,
        state: _ToolBlockState,
        *,
        status: str,
        arguments: str = "",
        input_text: str = "",
    ) -> dict[str, Any]:
        if state.kind == "custom":
            item = {
                "id": state.item_id,
                "type": "custom_tool_call",
                "status": status,
                "call_id": state.call_id,
                "name": state.name,
                "input": input_text,
            }
        else:
            item = {
                "id": state.item_id,
                "type": "function_call",
                "status": status,
                "call_id": state.call_id,
                "name": state.name,
                "arguments": arguments,
            }
        if state.namespace:
            item["namespace"] = state.namespace
        return item

    def _flush_active_blocks(self) -> list[str]:
        chunks: list[str] = []
        states = sorted(
            self._active_blocks.values(), key=lambda state: state.output_index
        )
        self._active_blocks.clear()
        for state in states:
            chunks.extend(self._complete_block(state))
        return chunks

    def _reserve_output_slot(self) -> int:
        output_index = len(self._output_slots)
        self._output_slots.append(None)
        return output_index

    def _commit_output(self, output_index: int, item: dict[str, Any]) -> None:
        while output_index >= len(self._output_slots):
            self._output_slots.append(None)
        self._output_slots[output_index] = item

    def _output(self) -> list[dict[str, Any]]:
        return [item for item in self._output_slots if item is not None]

    def _usage(self) -> dict[str, int] | None:
        if self._input_tokens is None and self._output_tokens is None:
            return None
        input_tokens = self._input_tokens or 0
        output_tokens = self._output_tokens or 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _safe_index(self, index: int | None) -> int:
        if index is not None:
            return index
        value = self._fallback_text_index
        self._fallback_text_index -= 1
        return value


def _event_index(data: Mapping[str, Any]) -> int | None:
    value = data.get("index")
    return value if isinstance(value, int) else None


def _reasoning_output_item(
    state: _ReasoningBlockState, *, status: str
) -> dict[str, Any]:
    if state.encrypted_content is not None:
        return encrypted_reasoning_item(state.item_id, state.encrypted_content, status)
    return reasoning_item(state.item_id, "".join(state.text_parts), status)


def _openai_error_from_anthropic_error(data: Mapping[str, Any]) -> dict[str, Any]:
    error = data.get("error")
    if not isinstance(error, dict):
        error = {"type": "api_error", "message": str(data)}
    return {
        "message": str(error.get("message", "")),
        "type": str(error.get("type", "api_error")),
        "param": None,
        "code": None,
    }


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)

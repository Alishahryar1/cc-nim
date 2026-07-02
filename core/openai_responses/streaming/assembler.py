"""Anthropic SSE to OpenAI Responses stream assembly."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from core.trace import trace_event

from ..anthropic_sse import AnthropicSseEvent
from ..errors import ResponsesConversionError
from ..ids import (
    new_call_id,
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
)
from ..tools import responses_tool_identity_from_anthropic_name
from . import event_builders as events
from .blocks import ReasoningBlockState, TextBlockState, ToolBlockState
from .completion import ResponseBlockCompleter, reasoning_output_item, tool_item
from .error_mapping import (
    openai_error_from_anthropic_error,
    replay_unsafe_function_call_error,
)
from .ledger import ResponsesOutputLedger


class ResponsesStreamAssembler:
    """Assemble Responses SSE events from indexed Anthropic content blocks."""

    def __init__(self, request: Mapping[str, Any]) -> None:
        self._request = request
        self._response_id = new_response_id()
        self._created_at = int(time.time())
        self._ledger = ResponsesOutputLedger()
        self._completer = ResponseBlockCompleter(
            self._ledger,
            on_invalid_function_call=self._fail_invalid_function_call,
        )
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
            self._ledger.record_usage_delta(event.data)
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
            "output": self._ledger.output(),
            "parallel_tool_calls": bool(self._request.get("parallel_tool_calls", True)),
            "tool_choice": self._request.get("tool_choice", "auto"),
            "temperature": self._request.get("temperature"),
            "top_p": self._request.get("top_p"),
            "max_output_tokens": self._request.get("max_output_tokens"),
            "usage": self._ledger.usage(),
            "error": error,
        }

    def complete_response(self) -> list[str]:
        chunks = self._flush_active_blocks()
        if self.terminal:
            return chunks
        self.final_response = self.response_payload(status="completed")
        chunks.append(events.response_completed(self.final_response))
        self.terminal = True
        return chunks

    def fail_response(self, data: Mapping[str, Any]) -> list[str]:
        chunks = self._flush_active_blocks()
        if self.terminal:
            return chunks
        error = openai_error_from_anthropic_error(data)
        self.final_response = self.response_payload(status="failed", error=error)
        chunks.append(events.response_failed(self.final_response))
        self.terminal = True
        return chunks

    def _ensure_started(self) -> list[str]:
        if self._started:
            return []
        self._started = True
        return [events.response_created(self.response_payload(status="in_progress"))]

    def _handle_content_block_start(self, data: Mapping[str, Any]) -> list[str]:
        block = data.get("content_block")
        if not isinstance(block, dict):
            return []
        block_type = block.get("type")
        index = _event_index(data)
        if block_type == "text":
            index = self._ledger.safe_text_index(index)
            chunks, state = self._start_text_block(index)
            if state is None:
                return chunks
            if text := _string_value(block.get("text")):
                chunks.extend(self._emit_text_delta(state, text))
            return chunks
        if block_type == "thinking":
            if index is None:
                return []
            chunks, state = self._start_reasoning_block(index)
            if state is None:
                return chunks
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
            index = self._ledger.safe_text_index(index)
            state = self._ledger.active_block(index)
            chunks: list[str] = []
            if not isinstance(state, TextBlockState):
                chunks, state = self._start_text_block(index)
                if state is None:
                    return chunks
            chunks.extend(
                self._emit_text_delta(state, _string_value(delta.get("text")))
            )
            return chunks
        if delta_type == "thinking_delta":
            if index is None:
                return []
            state = self._ledger.active_block(index)
            chunks = []
            if not isinstance(state, ReasoningBlockState):
                chunks, state = self._start_reasoning_block(index)
                if state is None:
                    return chunks
            chunks.extend(
                self._emit_reasoning_delta(state, _string_value(delta.get("thinking")))
            )
            return chunks
        if delta_type == "input_json_delta":
            state = self._ledger.active_block(index) if index is not None else None
            if isinstance(state, ToolBlockState):
                state.argument_parts.append(_string_value(delta.get("partial_json")))
        return []

    def _handle_content_block_stop(self, data: Mapping[str, Any]) -> list[str]:
        index = _event_index(data)
        if index is None:
            return []
        state = self._ledger.pop_active_block(index)
        if state is None:
            return []
        return self._completer.complete_block(state)

    def _start_text_block(self, index: int) -> tuple[list[str], TextBlockState | None]:
        chunks = self._complete_existing_block(index)
        if self.terminal:
            return chunks, None
        output_index = self._ledger.reserve_output_slot()
        state = TextBlockState(
            index=index,
            output_index=output_index,
            item_id=new_message_item_id(),
        )
        self._ledger.set_active_block(state)
        item = {
            "id": state.item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        chunks.extend(
            [
                events.output_item_added(output_index, item),
                events.content_part_added(state.item_id, output_index),
            ]
        )
        return chunks, state

    def _start_reasoning_block(
        self, index: int, *, encrypted_content: str | None = None
    ) -> tuple[list[str], ReasoningBlockState | None]:
        chunks = self._complete_existing_block(index)
        if self.terminal:
            return chunks, None
        output_index = self._ledger.reserve_output_slot()
        state = ReasoningBlockState(
            index=index,
            output_index=output_index,
            item_id=new_reasoning_item_id(),
            encrypted_content=encrypted_content,
        )
        self._ledger.set_active_block(state)
        chunks.append(
            events.output_item_added(
                output_index,
                reasoning_output_item(state, status="in_progress"),
            )
        )
        return chunks, state

    def _start_tool_block(self, index: int, block: Mapping[str, Any]) -> list[str]:
        chunks = self._complete_existing_block(index)
        if self.terminal:
            return chunks
        identity = responses_tool_identity_from_anthropic_name(
            self._request, _string_value(block.get("name"))
        )
        state = ToolBlockState(
            index=index,
            output_index=self._ledger.reserve_output_slot(),
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
        self._ledger.set_active_block(state)
        chunks.append(
            events.output_item_added(
                state.output_index,
                tool_item(state, status="in_progress"),
            )
        )
        return chunks

    def _emit_text_delta(self, state: TextBlockState, text: str) -> list[str]:
        if not text:
            return []
        state.text_parts.append(text)
        return [events.output_text_delta(state.item_id, state.output_index, text)]

    def _emit_reasoning_delta(self, state: ReasoningBlockState, text: str) -> list[str]:
        if not text:
            return []
        state.text_parts.append(text)
        return [events.reasoning_text_delta(state.item_id, state.output_index, text)]

    def _complete_existing_block(self, index: int) -> list[str]:
        existing = self._ledger.pop_active_block(index)
        if existing is None:
            return []
        return self._completer.complete_block(existing)

    def _flush_active_blocks(self) -> list[str]:
        chunks: list[str] = []
        for state in self._ledger.pop_active_blocks_by_output_order():
            if self.terminal:
                break
            chunks.extend(self._completer.complete_block(state))
        return chunks

    def _fail_invalid_function_call(
        self, state: ToolBlockState, exc: ResponsesConversionError
    ) -> list[str]:
        trace_event(
            stage="responses",
            event="responses.output.function_call_invalid_arguments",
            source="openai_responses",
            call_id=state.call_id,
            tool_name=state.name,
            error_type=type(exc).__name__,
        )
        error = replay_unsafe_function_call_error()
        self.final_response = self.response_payload(status="failed", error=error)
        self.terminal = True
        return [events.response_failed(self.final_response)]


def _event_index(data: Mapping[str, Any]) -> int | None:
    value = data.get("index")
    return value if isinstance(value, int) else None


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)

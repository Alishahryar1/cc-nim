"""Present canonical inference events through the Anthropic Messages protocol."""

import asyncio
import json
import sys
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceUsage,
    ReasoningBlockCompleted,
    ReasoningBlockStarted,
    ReasoningDelta,
    ReplayArtifact,
    ReplayArtifactKind,
    ResponseCompleted,
    ResponseStarted,
    TextBlockCompleted,
    TextBlockStarted,
    TextDelta,
    TokenMeasurement,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.replay_envelope import encode_replay_envelope
from free_claude_code.core.trace import close_stream_input

from .streaming.emitter import AnthropicSseEmitter


@dataclass(slots=True)
class _WireBlock:
    index: int | None
    block_type: str
    open: bool


class AnthropicEventPresenter:
    """Serialize one canonical stream while owning all Anthropic wire state."""

    def __init__(self, *, log_raw_events: bool = False) -> None:
        self._emitter = AnthropicSseEmitter(log_raw_events=log_raw_events)
        self._message_id = f"msg_{uuid.uuid4()}"
        self._next_index = 0
        self._blocks: dict[str, _WireBlock] = {}
        self._started = False
        self._completed = False
        self._usage = InferenceUsage()

    def present(self, event: InferenceEvent) -> list[str]:
        if self._completed:
            raise RuntimeError("canonical event arrived after response completion")
        if isinstance(event, ResponseStarted):
            return self._response_started(event)
        if not self._started:
            raise RuntimeError("canonical response must start before output")
        if isinstance(event, TextBlockStarted):
            return self._text_started(event)
        if isinstance(event, TextDelta):
            return [self._text_delta(event)]
        if isinstance(event, TextBlockCompleted):
            return [self._block_completed(event.block_id)]
        if isinstance(event, ReasoningBlockStarted):
            return self._reasoning_started(event)
        if isinstance(event, ReasoningDelta):
            return self._reasoning_delta(event)
        if isinstance(event, ReasoningBlockCompleted):
            return self._reasoning_completed(event)
        if isinstance(event, ToolCallStarted):
            return self._tool_started(event)
        if isinstance(event, ToolCallArgumentsDelta):
            return [self._tool_delta(event)]
        if isinstance(event, ToolCallCompleted):
            return [self._block_completed(event.block_id)]
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            return []
        return self._response_completed(event)

    def _response_started(self, event: ResponseStarted) -> list[str]:
        if self._started:
            raise RuntimeError("canonical response started more than once")
        self._started = True
        self._usage = event.initial_usage
        usage: JsonObject = {
            "input_tokens": _usage_value(event.initial_usage.input_tokens),
            "output_tokens": 1,
        }
        return [
            self._emitter.event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self._message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": event.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage,
                    },
                },
            )
        ]

    def _text_started(self, event: TextBlockStarted) -> list[str]:
        index = self._allocate_index()
        self._blocks[event.block_id] = _WireBlock(index, "text", True)
        return [
            self._emitter.event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]

    def _text_delta(self, event: TextDelta) -> str:
        block = self._require_open_block(event.block_id, "text")
        return self._emitter.event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": _required_index(block),
                "delta": {"type": "text_delta", "text": event.delta},
            },
        )

    def _reasoning_started(self, event: ReasoningBlockStarted) -> list[str]:
        block = _WireBlock(None, "reasoning", False)
        self._blocks[event.block_id] = block
        chunks: list[str] = []
        if not event.artifacts or any(
            artifact.kind is ReplayArtifactKind.THINKING_SIGNATURE
            for artifact in event.artifacts
        ):
            chunks.append(self._start_reasoning_wire_block(block))
        return chunks

    def _reasoning_delta(self, event: ReasoningDelta) -> list[str]:
        block = self._require_block(event.block_id, "reasoning")
        chunks: list[str] = []
        if not block.open:
            chunks.append(self._start_reasoning_wire_block(block))
        if event.delta:
            chunks.append(
                self._emitter.event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": _required_index(block),
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": event.delta,
                        },
                    },
                )
            )
        return chunks

    def _reasoning_completed(self, event: ReasoningBlockCompleted) -> list[str]:
        block = self._require_block(event.block_id, "reasoning")
        chunks: list[str] = []
        if event.artifacts:
            if block.open or event.reasoning:
                if not block.open:
                    chunks.append(self._start_reasoning_wire_block(block))
                chunks.append(self._signature_delta(block, event.artifacts))
                chunks.append(self._stop_wire_block(block))
            else:
                chunks.extend(self._redacted_reasoning_block(event.artifacts))
        elif block.open:
            chunks.append(self._stop_wire_block(block))
        block.open = False
        return chunks

    def _tool_started(self, event: ToolCallStarted) -> list[str]:
        index = self._allocate_index()
        self._blocks[event.block_id] = _WireBlock(index, "tool_call", True)
        content_block: JsonObject = {
            "type": "tool_use",
            "id": event.call_id,
            "name": event.name,
            "input": {},
        }
        extra_content = _tool_extra_content(event.artifacts)
        if extra_content:
            content_block["extra_content"] = extra_content
        return [
            self._emitter.event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": content_block,
                },
            )
        ]

    def _tool_delta(self, event: ToolCallArgumentsDelta) -> str:
        block = self._require_open_block(event.block_id, "tool_call")
        return self._emitter.event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": _required_index(block),
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": event.delta,
                },
            },
        )

    def _block_completed(self, block_id: str) -> str:
        block = self._require_open_block(block_id, None)
        return self._stop_wire_block(block)

    def _response_completed(self, event: ResponseCompleted) -> list[str]:
        if any(block.open for block in self._blocks.values()):
            raise RuntimeError("canonical response completed with open output blocks")
        self._usage = event.final_usage
        self._completed = True
        return [
            self._emitter.event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _anthropic_stop_reason(event.finish_reason),
                        "stop_sequence": event.stop_sequence,
                    },
                    "usage": _anthropic_usage(event.final_usage),
                },
            ),
            self._emitter.event("message_stop", {"type": "message_stop"}),
        ]

    def _signature_delta(
        self,
        block: _WireBlock,
        artifacts: tuple[ReplayArtifact, ...],
    ) -> str:
        return self._emitter.event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": _required_index(block),
                "delta": {
                    "type": "signature_delta",
                    "signature": encode_replay_envelope(artifacts),
                },
            },
        )

    def _redacted_reasoning_block(
        self, artifacts: tuple[ReplayArtifact, ...]
    ) -> list[str]:
        index = self._allocate_index()
        return [
            self._emitter.event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "redacted_thinking",
                        "data": encode_replay_envelope(artifacts),
                    },
                },
            ),
            self._emitter.event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            ),
        ]

    def _start_reasoning_wire_block(self, block: _WireBlock) -> str:
        block.index = self._allocate_index()
        block.open = True
        return self._emitter.event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block.index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )

    def _stop_wire_block(self, block: _WireBlock) -> str:
        index = _required_index(block)
        block.open = False
        return self._emitter.event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )

    def _allocate_index(self) -> int:
        index = self._next_index
        self._next_index += 1
        return index

    def _require_block(self, block_id: str, expected_type: str) -> _WireBlock:
        block = self._blocks.get(block_id)
        if block is None or block.block_type != expected_type:
            raise RuntimeError(f"unknown canonical {expected_type} block")
        return block

    def _require_open_block(
        self, block_id: str, expected_type: str | None
    ) -> _WireBlock:
        block = self._blocks.get(block_id)
        if (
            block is None
            or not block.open
            or (expected_type is not None and block.block_type != expected_type)
        ):
            raise RuntimeError("canonical delta/completion targeted a closed block")
        return block


async def iter_anthropic_sse(
    events: AsyncIterator[InferenceEvent],
    *,
    log_raw_events: bool = False,
) -> AsyncIterator[str]:
    """Serialize canonical events and retain ownership of the event stream."""

    presenter = AnthropicEventPresenter(log_raw_events=log_raw_events)
    try:
        async for event in events:
            for chunk in presenter.present(event):
                yield chunk
    finally:
        await close_stream_input(
            events,
            owner="anthropic_event_presenter",
            source="core",
            preserved_error=sys.exception(),
        )


async def aggregate_inference_events_to_message(
    events: AsyncIterator[InferenceEvent],
) -> JsonObject:
    """Fold canonical events directly into one non-stream Messages response."""

    message_id = f"msg_{uuid.uuid4()}"
    model = "unknown"
    content: list[JsonValue] = []
    final_usage = InferenceUsage()
    finish_reason = FinishReason.END_TURN
    stop_sequence: str | None = None
    started = False
    completed = False
    try:
        async for event in events:
            if isinstance(event, ResponseStarted):
                if started:
                    raise RuntimeError("canonical response started more than once")
                started = True
                model = event.model
                final_usage = event.initial_usage
            elif isinstance(event, TextBlockCompleted):
                content.append({"type": "text", "text": event.text})
            elif isinstance(event, ReasoningBlockCompleted):
                if event.reasoning:
                    block: JsonObject = {
                        "type": "thinking",
                        "thinking": event.reasoning,
                        "signature": _reasoning_signature(event.artifacts),
                    }
                    content.append(block)
                elif event.artifacts:
                    content.extend(_redacted_reasoning_blocks(event.artifacts))
            elif isinstance(event, ToolCallCompleted):
                tool_input: JsonValue = {}
                if event.arguments.strip():
                    try:
                        parsed: JsonValue = json.loads(event.arguments)
                    except json.JSONDecodeError:
                        parsed = {}
                    if isinstance(parsed, Mapping):
                        tool_input = {
                            str(key): _json_value(value)
                            for key, value in parsed.items()
                        }
                tool_block: JsonObject = {
                    "type": "tool_use",
                    "id": event.call_id,
                    "name": event.name,
                    "input": tool_input,
                }
                extra_content = _tool_extra_content(event.artifacts)
                if extra_content:
                    tool_block["extra_content"] = extra_content
                content.append(tool_block)
            elif isinstance(event, UsageUpdated):
                final_usage = event.usage
            elif isinstance(event, ResponseCompleted):
                final_usage = event.final_usage
                finish_reason = event.finish_reason
                stop_sequence = event.stop_sequence
                completed = True
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    finally:
        await close_stream_input(
            events,
            owner="anthropic_event_aggregator",
            source="core",
            preserved_error=sys.exception(),
        )
    if not started or not completed:
        raise RuntimeError("canonical response ended without its lifecycle markers")
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": _anthropic_stop_reason(finish_reason),
        "stop_sequence": stop_sequence,
        "usage": _anthropic_usage(final_usage),
    }


def _anthropic_stop_reason(reason: FinishReason) -> str:
    return {
        FinishReason.TOOL_CALLS: "tool_use",
        FinishReason.OUTPUT_LIMIT: "max_tokens",
        FinishReason.STOP_SEQUENCE: "stop_sequence",
    }.get(reason, "end_turn")


def _anthropic_usage(usage: InferenceUsage) -> JsonObject:
    result: JsonObject = {
        "input_tokens": _usage_value(usage.input_tokens),
        "output_tokens": _usage_value(usage.output_tokens),
    }
    if usage.cache_read_input_tokens is not None:
        result["cache_read_input_tokens"] = usage.cache_read_input_tokens.value
    if usage.cache_creation_input_tokens is not None:
        result["cache_creation_input_tokens"] = usage.cache_creation_input_tokens.value
    return result


def _usage_value(measurement: TokenMeasurement | None) -> int:
    return measurement.value if measurement is not None else 0


def _tool_extra_content(
    artifacts: tuple[ReplayArtifact, ...],
) -> JsonObject | None:
    return {"fcc_replay": encode_replay_envelope(artifacts)} if artifacts else None


def _reasoning_signature(artifacts: tuple[ReplayArtifact, ...]) -> str:
    return encode_replay_envelope(artifacts) if artifacts else ""


def _redacted_reasoning_blocks(
    artifacts: tuple[ReplayArtifact, ...],
) -> list[JsonValue]:
    if not artifacts:
        return []
    return [
        {
            "type": "redacted_thinking",
            "data": encode_replay_envelope(artifacts),
        }
    ]


def _required_index(block: _WireBlock) -> int:
    if block.index is None:
        raise RuntimeError("Anthropic wire block has no index")
    return block.index


def _json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)

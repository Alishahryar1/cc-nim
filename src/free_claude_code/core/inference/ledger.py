"""Semantic state for one provider-neutral inference stream."""

import hashlib
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from loguru import logger

from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.token_estimation import estimate_text_tokens

from .events import (
    FinishReason,
    InferenceEvent,
    InferenceUsage,
    ReasoningBlockCompleted,
    ReasoningBlockStarted,
    ReasoningDelta,
    ReplayArtifact,
    ResponseCompleted,
    ResponseStarted,
    TextBlockCompleted,
    TextBlockStarted,
    TextDelta,
    TokenMeasurement,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallKind,
    ToolCallStarted,
    UsageSource,
    UsageUpdated,
)


def _new_identity(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class ToolCallState:
    """Mutable assembly state for one upstream tool-call index."""

    block_id: str = ""
    item_id: str = ""
    call_id: str = ""
    name: str = ""
    kind: ToolCallKind = ToolCallKind.FUNCTION
    namespace: str | None = None
    artifacts: tuple[ReplayArtifact, ...] = ()
    started: bool = False
    task_arg_buffer: str = ""
    task_args_emitted: bool = False
    pre_start_args: str = ""


@dataclass(slots=True)
class InferenceBlockState:
    """Accumulated semantic state for one canonical output block."""

    block_id: str
    item_id: str
    block_type: str
    open: bool = True
    call_id: str = ""
    name: str = ""
    kind: ToolCallKind = ToolCallKind.FUNCTION
    namespace: str | None = None
    artifacts: tuple[ReplayArtifact, ...] = ()
    parts: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "".join(self.parts)


@dataclass(slots=True)
class InferenceBlockLedger:
    """Track active canonical blocks without assigning public wire indexes."""

    reasoning_block_id: str | None = None
    text_block_id: str | None = None
    reasoning_started: bool = False
    text_started: bool = False
    tool_states: dict[int, ToolCallState] = field(default_factory=dict)

    def ensure_tool_state(self, index: int) -> ToolCallState:
        return self.tool_states.setdefault(index, ToolCallState())

    def set_stream_tool_id(self, index: int, call_id: str | None) -> None:
        if call_id:
            self.ensure_tool_state(index).call_id = str(call_id)

    def set_tool_artifacts(
        self, index: int, artifacts: tuple[ReplayArtifact, ...]
    ) -> None:
        if artifacts:
            state = self.ensure_tool_state(index)
            state.artifacts = _merge_artifacts(state.artifacts, artifacts)

    def register_tool_name(self, index: int, name: str) -> None:
        state = self.ensure_tool_state(index)
        previous = state.name
        if not previous or name.startswith(previous):
            state.name = name
        elif not previous.startswith(name):
            state.name = "".join((previous, name))

    def buffer_task_args(self, index: int, args: str) -> dict[str, JsonValue] | None:
        state = self.tool_states.get(index)
        if state is None or state.task_args_emitted:
            return None

        state.task_arg_buffer += args
        try:
            parsed: JsonValue = json.loads(state.task_arg_buffer)
        except json.JSONDecodeError, TypeError, ValueError:
            return None
        if not isinstance(parsed, dict):
            return None

        normalized = {str(key): value for key, value in parsed.items()}
        _normalize_task_run_in_background(normalized)
        state.task_args_emitted = True
        state.task_arg_buffer = ""
        return normalized

    def flush_task_arg_buffers(self) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for tool_index, state in list(self.tool_states.items()):
            if not state.task_arg_buffer or state.task_args_emitted:
                continue

            output = "{}"
            try:
                parsed: JsonValue = json.loads(state.task_arg_buffer)
                if isinstance(parsed, dict):
                    normalized = {str(key): value for key, value in parsed.items()}
                    _normalize_task_run_in_background(normalized)
                    output = json.dumps(normalized)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                digest = hashlib.sha256(
                    state.task_arg_buffer.encode("utf-8", errors="replace")
                ).hexdigest()[:16]
                logger.warning(
                    "Task args invalid JSON (id={} len={} buffer_sha256_prefix={}): {}",
                    state.call_id or "unknown",
                    len(state.task_arg_buffer),
                    digest,
                    exc,
                )

            state.task_args_emitted = True
            state.task_arg_buffer = ""
            results.append((tool_index, output))
        return results


class InferenceStreamLedger:
    """Validate and accumulate one canonical provider output stream."""

    def __init__(
        self,
        response_id: str | None,
        model: str,
        input_tokens: int = 0,
    ) -> None:
        self.response_id = response_id or _new_identity("response")
        self.model = model
        self.input_tokens = max(input_tokens, 0)
        self.blocks = InferenceBlockLedger()
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._open_stack: list[str] = []
        self._content_blocks: dict[str, InferenceBlockState] = {}
        self.response_started = False
        self.response_completed = False
        self.finish_reason: FinishReason | None = None
        self.usage = InferenceUsage(
            input_tokens=TokenMeasurement(self.input_tokens, UsageSource.ESTIMATED)
        )

    def start_response(self) -> ResponseStarted:
        if self.response_started:
            raise RuntimeError("response already started")
        self.response_started = True
        return ResponseStarted(self.response_id, self.model, self.usage)

    def start_reasoning_block(
        self, *, artifacts: tuple[ReplayArtifact, ...] = ()
    ) -> ReasoningBlockStarted:
        self._require_response_open()
        block_id = _new_identity("block")
        item_id = _new_identity("item")
        self.blocks.reasoning_block_id = block_id
        self.blocks.reasoning_started = True
        self._record_block_start(
            InferenceBlockState(
                block_id=block_id,
                item_id=item_id,
                block_type="reasoning",
                artifacts=artifacts,
            )
        )
        return ReasoningBlockStarted(item_id, block_id, artifacts)

    def emit_reasoning_delta(self, content: str) -> ReasoningDelta:
        block = self._active_block(self.blocks.reasoning_block_id, "reasoning")
        block.parts.append(content)
        self._reasoning_parts.append(content)
        return ReasoningDelta(block.block_id, content)

    def stop_reasoning_block(self) -> ReasoningBlockCompleted:
        block = self._active_block(self.blocks.reasoning_block_id, "reasoning")
        self._record_block_stop(block.block_id)
        return ReasoningBlockCompleted(
            block.item_id,
            block.block_id,
            block.content,
            block.artifacts,
        )

    def emit_reasoning_artifact(
        self, artifact: ReplayArtifact
    ) -> Iterator[InferenceEvent]:
        yield from self.close_content_blocks()
        yield self.start_reasoning_block(artifacts=(artifact,))
        yield self.stop_reasoning_block()

    def start_text_block(self) -> TextBlockStarted:
        self._require_response_open()
        block_id = _new_identity("block")
        item_id = _new_identity("item")
        self.blocks.text_block_id = block_id
        self.blocks.text_started = True
        self._record_block_start(
            InferenceBlockState(
                block_id=block_id,
                item_id=item_id,
                block_type="text",
            )
        )
        return TextBlockStarted(item_id, block_id)

    def emit_text_delta(self, content: str) -> TextDelta:
        block = self._active_block(self.blocks.text_block_id, "text")
        block.parts.append(content)
        self._text_parts.append(content)
        return TextDelta(block.block_id, content)

    def stop_text_block(self) -> TextBlockCompleted:
        block = self._active_block(self.blocks.text_block_id, "text")
        self._record_block_stop(block.block_id)
        return TextBlockCompleted(block.item_id, block.block_id, block.content)

    def start_tool_block(
        self,
        tool_index: int,
        call_id: str,
        name: str,
        *,
        kind: ToolCallKind = ToolCallKind.FUNCTION,
        namespace: str | None = None,
        artifacts: tuple[ReplayArtifact, ...] = (),
    ) -> ToolCallStarted:
        self._require_response_open()
        block_id = _new_identity("block")
        item_id = _new_identity("item")
        state = self.blocks.ensure_tool_state(tool_index)
        state.block_id = block_id
        state.item_id = item_id
        state.call_id = call_id
        state.name = name
        state.kind = kind
        state.namespace = namespace
        state.artifacts = _merge_artifacts(state.artifacts, artifacts)
        state.started = True
        block = InferenceBlockState(
            block_id=block_id,
            item_id=item_id,
            block_type="tool_call",
            call_id=call_id,
            name=name,
            kind=kind,
            namespace=namespace,
            artifacts=state.artifacts,
        )
        self._record_block_start(block)
        return ToolCallStarted(
            item_id,
            block_id,
            call_id,
            kind,
            name,
            namespace,
            block.artifacts,
        )

    def set_tool_artifacts(
        self, tool_index: int, artifacts: tuple[ReplayArtifact, ...]
    ) -> None:
        self.blocks.set_tool_artifacts(tool_index, artifacts)
        state = self.blocks.tool_states[tool_index]
        if not state.block_id:
            return
        block = self._content_blocks.get(state.block_id)
        if block is not None:
            block.artifacts = state.artifacts

    def emit_tool_delta(
        self, tool_index: int, partial_arguments: str
    ) -> ToolCallArgumentsDelta:
        state = self.blocks.tool_states[tool_index]
        block = self._active_block(state.block_id, "tool_call")
        block.parts.append(partial_arguments)
        return ToolCallArgumentsDelta(block.block_id, partial_arguments)

    def stop_tool_block(self, tool_index: int) -> ToolCallCompleted:
        state = self.blocks.tool_states[tool_index]
        block = self._active_block(state.block_id, "tool_call")
        self._record_block_stop(block.block_id)
        state.started = False
        return ToolCallCompleted(
            block.item_id,
            block.block_id,
            block.call_id,
            block.kind,
            block.name,
            block.content,
            block.namespace,
            block.artifacts,
        )

    def ensure_reasoning_block(self) -> Iterator[InferenceEvent]:
        if self.blocks.text_started:
            yield self.stop_text_block()
        if not self.blocks.reasoning_started:
            yield self.start_reasoning_block()

    def ensure_text_block(self) -> Iterator[InferenceEvent]:
        if self.blocks.reasoning_started:
            yield self.stop_reasoning_block()
        if not self.blocks.text_started:
            yield self.start_text_block()

    def close_content_blocks(self) -> Iterator[InferenceEvent]:
        if self.blocks.reasoning_started:
            yield self.stop_reasoning_block()
        if self.blocks.text_started:
            yield self.stop_text_block()

    def close_all_blocks(self) -> Iterator[InferenceEvent]:
        yield from self.close_content_blocks()
        for tool_index, state in list(self.blocks.tool_states.items()):
            if state.started:
                yield self.stop_tool_block(tool_index)

    def close_unclosed_blocks(self) -> Iterator[InferenceEvent]:
        while self._open_stack:
            block_id = self._open_stack[-1]
            block = self._content_blocks[block_id]
            if block.block_type == "text":
                yield self.stop_text_block()
            elif block.block_type == "reasoning":
                yield self.stop_reasoning_block()
            else:
                tool_index = self._tool_index_for_block(block_id)
                if tool_index is None:
                    raise RuntimeError("tool block has no assembly state")
                yield self.stop_tool_block(tool_index)

    def finish_events(
        self,
        finish_reason: FinishReason,
        usage: InferenceUsage,
        *,
        stop_sequence: str | None = None,
    ) -> tuple[UsageUpdated, ResponseCompleted]:
        self._require_response_open()
        if self._open_stack:
            raise RuntimeError("response cannot complete with open output blocks")
        self.finish_reason = self.final_finish_reason(finish_reason)
        self.usage = usage
        self.response_completed = True
        return (
            UsageUpdated(usage),
            ResponseCompleted(self.finish_reason, usage, stop_sequence),
        )

    def tool_blocks(self) -> list[InferenceBlockState]:
        return [
            block
            for block in self._content_blocks.values()
            if block.block_type == "tool_call"
        ]

    def tool_block_for_tool_index(self, tool_index: int) -> InferenceBlockState | None:
        state = self.blocks.tool_states.get(tool_index)
        if state is None or not state.block_id:
            return None
        block = self._content_blocks.get(state.block_id)
        if block is None or block.block_type != "tool_call":
            return None
        return block

    def has_emitted_tool_block(self) -> bool:
        return bool(self.tool_blocks())

    def has_content_block(self) -> bool:
        return bool(self._content_blocks)

    def has_generated_output(self) -> bool:
        return self.has_content_block()

    def final_finish_reason(self, fallback: FinishReason) -> FinishReason:
        if self.has_emitted_tool_block():
            return FinishReason.TOOL_CALLS
        return fallback

    @property
    def accumulated_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def accumulated_reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    def estimate_output_tokens(self) -> int:
        text_tokens = estimate_text_tokens(self.accumulated_text)
        reasoning_tokens = estimate_text_tokens(self.accumulated_reasoning)
        tool_tokens = 0
        tool_count = 0
        for name, content in self._iter_tool_token_payloads():
            tool_tokens += estimate_text_tokens(name)
            tool_tokens += estimate_text_tokens(content)
            tool_tokens += 15
            tool_count += 1

        block_count = (
            (1 if self.accumulated_reasoning else 0)
            + (1 if self.accumulated_text else 0)
            + tool_count
        )
        return text_tokens + reasoning_tokens + tool_tokens + (block_count * 4)

    def estimate_reasoning_tokens(self) -> int:
        return estimate_text_tokens(self.accumulated_reasoning)

    def _iter_tool_token_payloads(self) -> Iterator[tuple[str, str]]:
        for block in self.tool_blocks():
            yield block.name, block.content

    def _record_block_start(self, block: InferenceBlockState) -> None:
        self._content_blocks[block.block_id] = block
        self._open_stack.append(block.block_id)

    def _record_block_stop(self, block_id: str) -> None:
        block = self._content_blocks[block_id]
        if not block.open:
            raise RuntimeError("canonical block already completed")
        try:
            self._open_stack.remove(block_id)
        except ValueError as exc:
            raise RuntimeError("canonical block is not open") from exc
        block.open = False
        if block.block_type == "text":
            self.blocks.text_started = False
        elif block.block_type == "reasoning":
            self.blocks.reasoning_started = False

    def _active_block(
        self, block_id: str | None, expected_type: str
    ) -> InferenceBlockState:
        if block_id is None:
            raise RuntimeError(f"no active {expected_type} block")
        block = self._content_blocks.get(block_id)
        if block is None or block.block_type != expected_type or not block.open:
            raise RuntimeError(f"no active {expected_type} block")
        return block

    def _tool_index_for_block(self, block_id: str) -> int | None:
        for tool_index, state in self.blocks.tool_states.items():
            if state.block_id == block_id:
                return tool_index
        return None

    def _require_response_open(self) -> None:
        if not self.response_started:
            raise RuntimeError("response has not started")
        if self.response_completed:
            raise RuntimeError("response already completed")


def _merge_artifacts(
    existing: tuple[ReplayArtifact, ...], incoming: tuple[ReplayArtifact, ...]
) -> tuple[ReplayArtifact, ...]:
    return (*existing, *(artifact for artifact in incoming if artifact not in existing))


def _normalize_task_run_in_background(args: dict[str, JsonValue]) -> None:
    if args.get("run_in_background") is not False:
        args["run_in_background"] = False

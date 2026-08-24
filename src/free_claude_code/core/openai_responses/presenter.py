"""Present canonical inference events through the OpenAI Responses protocol."""

import asyncio
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field

from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
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
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallKind,
    ToolCallStarted,
    UsageUpdated,
    replay_payload_text,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.token_estimation import estimate_text_tokens
from free_claude_code.core.trace import close_stream_input, trace_event

from .errors import (
    ResponsesConversionError,
    openai_error_from_failure,
    openai_error_payload,
)
from .ids import (
    new_call_id,
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
)
from .models import OpenAIResponsesRequest
from .streaming.error_mapping import replay_unsafe_function_call_error
from .streaming.event_builders import ResponseEventBuilder
from .tools import (
    custom_tool_input_text_from_arguments,
    normalized_function_call_arguments,
    responses_tool_identity_from_anthropic_name,
)

PostStartTerminalFailureObserver = Callable[[BaseException], None]


@dataclass(slots=True)
class _TextState:
    output_index: int
    item_id: str
    parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ReasoningState:
    output_index: int
    item_id: str
    artifacts: tuple[ReplayArtifact, ...]
    parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ToolState:
    output_index: int
    item_id: str
    call_id: str
    kind: ToolCallKind
    name: str
    namespace: str | None
    argument_parts: list[str] = field(default_factory=list)


type _OutputState = _TextState | _ReasoningState | _ToolState


class ResponsesEventPresenter:
    """Serialize one canonical stream while owning Responses envelope state."""

    def __init__(self, request: OpenAIResponsesRequest) -> None:
        self._request = request
        self._response_id = new_response_id()
        self._created_at = int(time.time())
        self._events = ResponseEventBuilder()
        self._blocks: dict[str, _OutputState] = {}
        self._output: list[JsonObject | None] = []
        self._usage = InferenceUsage()
        self._reasoning_tokens_estimate = 0
        self._provisional_error: JsonObject | None = None
        self.started = False
        self.terminal = False
        self.final_response: JsonObject | None = None

    def present(self, event: InferenceEvent) -> list[str]:
        if self.terminal:
            raise RuntimeError("canonical event arrived after response completion")
        if isinstance(event, ResponseStarted):
            return self._response_started(event)
        if not self.started:
            raise RuntimeError("canonical response must start before output")
        if isinstance(event, TextBlockStarted):
            return self._text_started(event)
        if isinstance(event, TextDelta):
            return self._text_delta(event)
        if isinstance(event, TextBlockCompleted):
            return self._text_completed(event)
        if isinstance(event, ReasoningBlockStarted):
            return self._reasoning_started(event)
        if isinstance(event, ReasoningDelta):
            return self._reasoning_delta(event)
        if isinstance(event, ReasoningBlockCompleted):
            return self._reasoning_completed(event)
        if isinstance(event, ToolCallStarted):
            return self._tool_started(event)
        if isinstance(event, ToolCallArgumentsDelta):
            return self._tool_delta(event)
        if isinstance(event, ToolCallCompleted):
            return self._tool_completed(event)
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            return []
        return self._response_completed(event)

    def response_payload(
        self,
        *,
        status: str,
        error: JsonObject | None = None,
        incomplete_details: JsonObject | None = None,
    ) -> JsonObject:
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": self._request.model,
            "output": [item for item in self._output if item is not None],
            "parallel_tool_calls": (
                True
                if self._request.parallel_tool_calls is None
                else self._request.parallel_tool_calls
            ),
            "tool_choice": _json_value(
                "auto"
                if self._request.tool_choice is None
                else self._request.tool_choice
            ),
            "temperature": self._request.temperature,
            "top_p": self._request.top_p,
            "max_output_tokens": self._request.max_output_tokens,
            "usage": _responses_usage(
                self._usage,
                reasoning_estimate=self._reasoning_tokens_estimate,
            ),
            "error": error,
            "incomplete_details": incomplete_details,
        }

    def fail_execution(self, failure: ExecutionFailure) -> list[str]:
        return self._finish_failed(_json_object(openai_error_from_failure(failure)))

    def fail_unexpected(self, exc: BaseException) -> list[str]:
        payload = openai_error_payload(
            message=safe_exception_message(exc),
            error_type="api_error",
        )["error"]
        return self._finish_failed(_json_object(payload))

    def _response_started(self, event: ResponseStarted) -> list[str]:
        if self.started:
            raise RuntimeError("canonical response started more than once")
        self.started = True
        self._usage = event.initial_usage
        return [
            self._events.response_created(self.response_payload(status="in_progress"))
        ]

    def _text_started(self, event: TextBlockStarted) -> list[str]:
        output_index = self._reserve_output()
        state = _TextState(output_index, new_message_item_id())
        self._set_block(event.block_id, state)
        item: JsonObject = {
            "id": state.item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return [
            self._events.output_item_added(output_index, item),
            self._events.content_part_added(state.item_id, output_index),
        ]

    def _text_delta(self, event: TextDelta) -> list[str]:
        state = self._require_block(event.block_id, _TextState)
        if not event.delta:
            return []
        state.parts.append(event.delta)
        return [
            self._events.output_text_delta(
                state.item_id, state.output_index, event.delta
            )
        ]

    def _text_completed(self, event: TextBlockCompleted) -> list[str]:
        state = self._pop_block(event.block_id, _TextState)
        text = event.text
        item = _message_item(state.item_id, text, "completed")
        self._commit_output(state.output_index, item)
        return [
            self._events.output_text_done(state.item_id, state.output_index, text),
            self._events.content_part_done(state.item_id, state.output_index, text),
            self._events.output_item_done(state.output_index, item),
        ]

    def _reasoning_started(self, event: ReasoningBlockStarted) -> list[str]:
        output_index = self._reserve_output()
        state = _ReasoningState(
            output_index,
            new_reasoning_item_id(),
            event.artifacts,
        )
        self._set_block(event.block_id, state)
        return [
            self._events.output_item_added(
                output_index,
                _reasoning_item(state, status="in_progress"),
            )
        ]

    def _reasoning_delta(self, event: ReasoningDelta) -> list[str]:
        state = self._require_block(event.block_id, _ReasoningState)
        if not event.delta:
            return []
        state.parts.append(event.delta)
        return [
            self._events.reasoning_text_delta(
                state.item_id, state.output_index, event.delta
            )
        ]

    def _reasoning_completed(self, event: ReasoningBlockCompleted) -> list[str]:
        state = self._pop_block(event.block_id, _ReasoningState)
        state.artifacts = event.artifacts
        text = event.reasoning
        item = _reasoning_item(state, status="completed", text=text)
        self._commit_output(state.output_index, item)
        chunks: list[str] = []
        if text:
            self._reasoning_tokens_estimate += estimate_text_tokens(text)
            chunks.append(
                self._events.reasoning_text_done(
                    state.item_id,
                    state.output_index,
                    text,
                )
            )
        chunks.append(self._events.output_item_done(state.output_index, item))
        return chunks

    def _tool_started(self, event: ToolCallStarted) -> list[str]:
        identity = responses_tool_identity_from_anthropic_name(
            self._request.tools,
            event.name,
        )
        kind = (
            ToolCallKind.CUSTOM
            if event.kind is ToolCallKind.CUSTOM or identity.kind == "custom"
            else ToolCallKind.FUNCTION
        )
        state = _ToolState(
            output_index=self._reserve_output(),
            item_id=f"{'ctc' if kind is ToolCallKind.CUSTOM else 'fc'}_"
            f"{uuid.uuid4().hex[:24]}",
            call_id=event.call_id or new_call_id(),
            kind=kind,
            name=identity.name if identity.name else event.name,
            namespace=identity.namespace or event.namespace,
        )
        self._set_block(event.block_id, state)
        return [
            self._events.output_item_added(
                state.output_index,
                _tool_item(state, status="in_progress"),
            )
        ]

    def _tool_delta(self, event: ToolCallArgumentsDelta) -> list[str]:
        state = self._require_block(event.block_id, _ToolState)
        state.argument_parts.append(event.delta)
        return []

    def _tool_completed(self, event: ToolCallCompleted) -> list[str]:
        state = self._pop_block(event.block_id, _ToolState)
        arguments = event.arguments
        if state.kind is ToolCallKind.CUSTOM:
            return self._complete_custom_tool(state, arguments)
        try:
            normalized = normalized_function_call_arguments(arguments or "{}")
        except ResponsesConversionError as exc:
            trace_event(
                stage="responses",
                event="responses.output.function_call_invalid_arguments",
                source="openai_responses",
                call_id=state.call_id,
                tool_name=state.name,
                error_type=type(exc).__name__,
            )
            self._provisional_error = _json_object(replay_unsafe_function_call_error())
            return []
        item = _tool_item(state, status="completed", arguments=normalized)
        self._commit_output(state.output_index, item)
        chunks: list[str] = []
        if normalized:
            chunks.append(
                self._events.function_call_arguments_delta(
                    state.item_id,
                    state.output_index,
                    normalized,
                )
            )
        chunks.extend(
            [
                self._events.function_call_arguments_done(
                    state.item_id,
                    state.output_index,
                    normalized,
                ),
                self._events.output_item_done(state.output_index, item),
            ]
        )
        return chunks

    def _complete_custom_tool(self, state: _ToolState, arguments: str) -> list[str]:
        input_text = custom_tool_input_text_from_arguments(arguments)
        item = _tool_item(state, status="completed", input_text=input_text)
        self._commit_output(state.output_index, item)
        chunks: list[str] = []
        if input_text:
            chunks.append(
                self._events.custom_tool_call_input_delta(
                    state.item_id,
                    state.output_index,
                    input_text,
                )
            )
        chunks.extend(
            [
                self._events.custom_tool_call_input_done(
                    state.item_id,
                    state.output_index,
                    input_text,
                ),
                self._events.output_item_done(state.output_index, item),
            ]
        )
        return chunks

    def _response_completed(self, event: ResponseCompleted) -> list[str]:
        if self._blocks:
            raise RuntimeError("canonical response completed with open output blocks")
        self._usage = event.final_usage
        if self._provisional_error is not None:
            return self._finish_failed(self._provisional_error)
        if event.finish_reason is FinishReason.OUTPUT_LIMIT:
            self.final_response = self.response_payload(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            )
            self.terminal = True
            return [self._events.response_incomplete(self.final_response)]
        self.final_response = self.response_payload(status="completed")
        self.terminal = True
        return [self._events.response_completed(self.final_response)]

    def _finish_failed(self, error: JsonObject) -> list[str]:
        if self.terminal:
            return []
        self._blocks.clear()
        self._provisional_error = None
        self.final_response = self.response_payload(status="failed", error=error)
        self.terminal = True
        return [self._events.response_failed(self.final_response)]

    def _set_block(self, block_id: str, state: _OutputState) -> None:
        if block_id in self._blocks:
            raise RuntimeError("canonical block started more than once")
        self._blocks[block_id] = state

    def _require_block[State: _OutputState](
        self, block_id: str, expected: type[State]
    ) -> State:
        state = self._blocks.get(block_id)
        if not isinstance(state, expected):
            raise RuntimeError("canonical delta targeted an unknown output block")
        return state

    def _pop_block[State: _OutputState](
        self, block_id: str, expected: type[State]
    ) -> State:
        state = self._require_block(block_id, expected)
        del self._blocks[block_id]
        return state

    def _reserve_output(self) -> int:
        index = len(self._output)
        self._output.append(None)
        return index

    def _commit_output(self, output_index: int, item: JsonObject) -> None:
        self._output[output_index] = item


async def iter_responses_sse_from_events(
    events: AsyncIterator[InferenceEvent],
    request: OpenAIResponsesRequest,
    *,
    on_post_start_terminal_failure: PostStartTerminalFailureObserver | None = None,
) -> AsyncIterator[str]:
    """Serialize canonical events and map only post-start failures to wire events."""

    presenter = ResponsesEventPresenter(request)
    emitted_any = False
    try:
        async for event in events:
            for chunk in presenter.present(event):
                emitted_any = True
                yield chunk
        if not presenter.terminal:
            raise RuntimeError(
                "canonical provider stream ended without response completion"
            )
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        if not emitted_any:
            raise
        failure = find_execution_failure(exc)
        observed = failure or exc
        _observe(on_post_start_terminal_failure, observed)
        chunks = (
            presenter.fail_execution(failure)
            if failure is not None
            else presenter.fail_unexpected(exc)
        )
        for chunk in chunks:
            yield chunk
    except Exception as exc:
        if not emitted_any:
            raise
        failure = find_execution_failure(exc)
        observed = failure or exc
        _observe(on_post_start_terminal_failure, observed)
        chunks = (
            presenter.fail_execution(failure)
            if failure is not None
            else presenter.fail_unexpected(exc)
        )
        for chunk in chunks:
            yield chunk
    finally:
        await close_stream_input(
            events,
            owner="responses_event_presenter",
            source="core",
            preserved_error=sys.exception(),
        )


def _observe(
    observer: PostStartTerminalFailureObserver | None,
    exc: BaseException,
) -> None:
    if observer is not None:
        observer(exc)


def _message_item(item_id: str, text: str, status: str) -> JsonObject:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning_item(
    state: _ReasoningState,
    *,
    status: str,
    text: str = "",
) -> JsonObject:
    encrypted = _encrypted_reasoning(state.artifacts)
    if encrypted is not None:
        return {
            "id": state.item_id,
            "type": "reasoning",
            "status": status,
            "summary": [],
            "encrypted_content": encrypted,
        }
    return {
        "id": state.item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _encrypted_reasoning(
    artifacts: tuple[ReplayArtifact, ...],
) -> str | None:
    for artifact in artifacts:
        if artifact.kind in {
            ReplayArtifactKind.REDACTED_THINKING,
            ReplayArtifactKind.ENCRYPTED_REASONING,
            ReplayArtifactKind.REASONING_DETAILS,
        }:
            return replay_payload_text(artifact)
    return None


def _tool_item(
    state: _ToolState,
    *,
    status: str,
    arguments: str = "",
    input_text: str = "",
) -> JsonObject:
    if state.kind is ToolCallKind.CUSTOM:
        item: JsonObject = {
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


def _responses_usage(
    usage: InferenceUsage,
    *,
    reasoning_estimate: int,
) -> JsonObject | None:
    values = (
        usage.input_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        usage.output_tokens,
    )
    if all(value is None for value in values):
        return None
    input_tokens = sum(
        value.value
        for value in (
            usage.input_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        if value is not None
    )
    output_tokens = usage.output_tokens.value if usage.output_tokens else 0
    reasoning_tokens = (
        usage.reasoning_output_tokens.value
        if usage.reasoning_output_tokens is not None
        else min(reasoning_estimate, output_tokens)
    )
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": (
                usage.cache_read_input_tokens.value
                if usage.cache_read_input_tokens is not None
                else 0
            )
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _json_object(value: Mapping[str, object]) -> JsonObject:
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)

"""Decode upstream OpenAI Responses events into canonical inference events."""

from collections.abc import Mapping
from dataclasses import dataclass

from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceStreamLedger,
    InferenceUsage,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
    TokenMeasurement,
    ToolCallKind,
    UsageSource,
)
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec


class ResponsesStreamFailure(RuntimeError):
    """An upstream Responses stream reported a terminal failure."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(slots=True)
class _ToolState:
    tool_index: int
    call_id: str
    name: str
    kind: ToolCallKind
    namespace: str | None
    started: bool = False
    received_delta: bool = False
    stopped: bool = False


class ResponsesEventDecoder:
    """Decode one upstream Responses stream without assigning public wire IDs."""

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        input_tokens: int,
        tool_names: OpenAIToolNameCodec | None = None,
        replay_scope: ReplayCompatibilityScope,
    ) -> None:
        self.ledger = InferenceStreamLedger(response_id, model, input_tokens)
        self.completed = False
        self.generated_output = False
        self._tool_names = tool_names or OpenAIToolNameCodec.from_names(())
        self._replay_scope = replay_scope
        self._tools: dict[str, _ToolState] = {}
        self._encrypted_reasoning: dict[str, str] = {}

    def start(self) -> list[InferenceEvent]:
        return [self.ledger.start_response()]

    def feed(self, event_type: str, data: Mapping[str, object]) -> list[InferenceEvent]:
        if self.completed:
            return []
        if event_type == "response.output_item.added":
            return self._item_added(data)
        if event_type in {
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            return self._reasoning_delta(data)
        if event_type == "response.output_text.delta":
            return self._text_delta(data)
        if event_type in {
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        }:
            return self._tool_delta(data)
        if event_type == "response.output_item.done":
            return self._item_done(data)
        if event_type in {"response.completed", "response.incomplete"}:
            return self._finish(data, incomplete=event_type == "response.incomplete")
        if event_type in {"response.failed", "error", "response.error"}:
            raise _stream_failure(data)
        return []

    def _item_added(self, data: Mapping[str, object]) -> list[InferenceEvent]:
        item = data.get("item")
        if not isinstance(item, dict):
            return []
        item_id = _string(item.get("id"))
        if item.get("type") in {"function_call", "custom_tool_call"} and item_id:
            identity = self._tool_names.decode_identity(_string(item.get("name")))
            self._tools[item_id] = _ToolState(
                tool_index=len(self._tools),
                call_id=_string(item.get("call_id")) or item_id,
                name=identity.name,
                kind=(
                    ToolCallKind.CUSTOM
                    if item.get("type") == "custom_tool_call"
                    else identity.kind
                ),
                namespace=identity.namespace,
            )
        if item.get("type") == "reasoning" and item_id:
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                self._encrypted_reasoning[item_id] = encrypted
        return []

    def _reasoning_delta(self, data: Mapping[str, object]) -> list[InferenceEvent]:
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        events = list(self.ledger.ensure_reasoning_block())
        events.append(self.ledger.emit_reasoning_delta(delta))
        self.generated_output = True
        return events

    def _text_delta(self, data: Mapping[str, object]) -> list[InferenceEvent]:
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        events = list(self.ledger.ensure_text_block())
        events.append(self.ledger.emit_text_delta(delta))
        self.generated_output = True
        return events

    def _tool_delta(self, data: Mapping[str, object]) -> list[InferenceEvent]:
        item_id = _string(data.get("item_id"))
        delta = data.get("delta")
        if not item_id or not isinstance(delta, str):
            return []
        state = self._tools.get(item_id)
        if state is None:
            state = _ToolState(
                tool_index=len(self._tools),
                call_id=item_id,
                name="",
                kind=ToolCallKind.FUNCTION,
                namespace=None,
            )
            self._tools[item_id] = state
        events = list(self.ledger.close_content_blocks())
        events.extend(self._ensure_tool_started(state))
        if delta:
            events.append(self.ledger.emit_tool_delta(state.tool_index, delta))
            state.received_delta = True
            self.generated_output = True
        return events

    def _item_done(self, data: Mapping[str, object]) -> list[InferenceEvent]:
        item = data.get("item")
        if not isinstance(item, dict):
            return []
        item_type = item.get("type")
        item_id = _string(item.get("id"))
        if item_type in {"function_call", "custom_tool_call"}:
            state = self._tools.get(item_id)
            if state is None:
                identity = self._tool_names.decode_identity(_string(item.get("name")))
                state = _ToolState(
                    tool_index=len(self._tools),
                    call_id=_string(item.get("call_id")) or item_id,
                    name=identity.name,
                    kind=(
                        ToolCallKind.CUSTOM
                        if item_type == "custom_tool_call"
                        else identity.kind
                    ),
                    namespace=identity.namespace,
                )
                self._tools[item_id] = state
            if not state.name:
                identity = self._tool_names.decode_identity(_string(item.get("name")))
                state.name = identity.name
                state.kind = identity.kind
                state.namespace = identity.namespace
            events = list(self.ledger.close_content_blocks())
            events.extend(self._ensure_tool_started(state))
            arguments = item.get(
                "input" if item_type == "custom_tool_call" else "arguments"
            )
            if not state.received_delta and isinstance(arguments, str) and arguments:
                events.append(self.ledger.emit_tool_delta(state.tool_index, arguments))
                self.generated_output = True
            if not state.stopped:
                events.append(self.ledger.stop_tool_block(state.tool_index))
                state.stopped = True
            return events
        if item_type == "reasoning":
            events = list(self.ledger.close_content_blocks())
            encrypted = item.get("encrypted_content")
            if not isinstance(encrypted, str) or not encrypted:
                encrypted = self._encrypted_reasoning.get(item_id)
            if isinstance(encrypted, str) and encrypted:
                events.extend(
                    self.ledger.emit_reasoning_artifact(
                        ReplayArtifact(
                            origin=ReplayArtifactOrigin.OPENAI,
                            kind=ReplayArtifactKind.ENCRYPTED_REASONING,
                            attachment=ReplayAttachment.REASONING,
                            payload=encrypted,
                            scope=self._replay_scope,
                        )
                    )
                )
                self.generated_output = True
            return events
        return []

    def _ensure_tool_started(self, state: _ToolState) -> list[InferenceEvent]:
        if state.started:
            return []
        state.started = True
        self.generated_output = True
        return [
            self.ledger.start_tool_block(
                state.tool_index,
                state.call_id,
                state.name,
                kind=state.kind,
                namespace=state.namespace,
            )
        ]

    def _finish(
        self, data: Mapping[str, object], *, incomplete: bool
    ) -> list[InferenceEvent]:
        response = data.get("response")
        response = response if isinstance(response, dict) else {}
        events = list(self.ledger.close_all_blocks())
        if not self.ledger.has_content_block():
            events.extend(self.ledger.ensure_text_block())
            events.append(self.ledger.emit_text_delta(" "))
            events.append(self.ledger.stop_text_block())

        usage = _response_usage(response, self.ledger)
        events.extend(
            self.ledger.finish_events(
                FinishReason.OUTPUT_LIMIT if incomplete else FinishReason.END_TURN,
                usage,
            )
        )
        self.completed = True
        return events


def _response_usage(
    response: Mapping[str, object], ledger: InferenceStreamLedger
) -> InferenceUsage:
    raw_usage = response.get("usage")
    raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = _integer(raw_usage.get("input_tokens"))
    output_tokens = _integer(raw_usage.get("output_tokens"))

    input_details = raw_usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    cached_tokens = _integer(input_details.get("cached_tokens"))
    if input_tokens is None:
        cached_tokens = None
    if (
        input_tokens is not None
        and cached_tokens is not None
        and (cached_tokens < 0 or cached_tokens > input_tokens)
    ):
        cached_tokens = None
    uncached_tokens = (
        input_tokens - cached_tokens
        if input_tokens is not None and cached_tokens is not None
        else input_tokens
    )

    output_details = raw_usage.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    reasoning_tokens = _integer(output_details.get("reasoning_tokens"))
    estimated_reasoning = ledger.estimate_reasoning_tokens()

    return InferenceUsage(
        input_tokens=_measurement(
            uncached_tokens,
            fallback=ledger.input_tokens,
        ),
        cache_read_input_tokens=(
            TokenMeasurement(cached_tokens, UsageSource.REPORTED)
            if cached_tokens is not None
            else None
        ),
        output_tokens=_measurement(
            output_tokens,
            fallback=ledger.estimate_output_tokens(),
        ),
        reasoning_output_tokens=(
            TokenMeasurement(reasoning_tokens, UsageSource.REPORTED)
            if reasoning_tokens is not None
            else TokenMeasurement(estimated_reasoning, UsageSource.ESTIMATED)
        ),
    )


def _measurement(value: int | None, *, fallback: int) -> TokenMeasurement:
    if value is not None and value >= 0:
        return TokenMeasurement(value, UsageSource.REPORTED)
    return TokenMeasurement(max(fallback, 0), UsageSource.ESTIMATED)


def _stream_failure(data: Mapping[str, object]) -> ResponsesStreamFailure:
    response = data.get("response")
    response = response if isinstance(response, dict) else {}
    error = response.get("error", data.get("error"))
    error = error if isinstance(error, dict) else {}
    message = error.get("message")
    code = error.get("code", error.get("type"))
    return ResponsesStreamFailure(
        message if isinstance(message, str) and message else "OpenAI response failed.",
        code=code if isinstance(code, str) else None,
    )


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None

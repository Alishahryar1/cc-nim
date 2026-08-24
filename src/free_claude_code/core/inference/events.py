"""Provider-neutral streamed inference output values."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from free_claude_code.core.json_types import JsonValue


class FinishReason(StrEnum):
    """Canonical reason a successful provider stream stopped."""

    END_TURN = "end_turn"
    TOOL_CALLS = "tool_calls"
    OUTPUT_LIMIT = "output_limit"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    PROVIDER_UNKNOWN = "provider_unknown"


class UsageSource(StrEnum):
    """Whether a token measurement came from upstream or FCC estimation."""

    REPORTED = "reported"
    ESTIMATED = "estimated"


@dataclass(frozen=True, slots=True)
class TokenMeasurement:
    """One non-negative token count with explicit provenance."""

    value: int
    source: UsageSource

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, int)
            or isinstance(self.value, bool)
            or self.value < 0
        ):
            raise ValueError("token measurements must be non-negative integers")


@dataclass(frozen=True, slots=True)
class InferenceUsage:
    """Cumulative provider-neutral token usage."""

    input_tokens: TokenMeasurement | None = None
    cache_read_input_tokens: TokenMeasurement | None = None
    cache_creation_input_tokens: TokenMeasurement | None = None
    output_tokens: TokenMeasurement | None = None
    reasoning_output_tokens: TokenMeasurement | None = None


class ReplayArtifactOrigin(StrEnum):
    """Protocol or provider family that owns an opaque replay value."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    OPENAI_COMPATIBLE = "openai_compatible"


class ReplayArtifactKind(StrEnum):
    """Known opaque replay value kinds preserved by FCC."""

    THINKING_SIGNATURE = "thinking_signature"
    REDACTED_THINKING = "redacted_thinking"
    ENCRYPTED_REASONING = "encrypted_reasoning"
    REASONING_DETAILS = "reasoning_details"
    THOUGHT_SIGNATURE = "thought_signature"
    TOOL_EXTRA_CONTENT = "tool_extra_content"


class ReplayAttachment(StrEnum):
    """Semantic item to which a replay artifact belongs."""

    REASONING = "reasoning"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True, slots=True)
class ReplayArtifact:
    """Opaque provider replay material with a typed owner and scope."""

    origin: ReplayArtifactOrigin
    kind: ReplayArtifactKind
    attachment: ReplayAttachment
    payload: JsonValue


class ToolCallKind(StrEnum):
    """Canonical tool invocation representation."""

    FUNCTION = "function"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ResponseStarted:
    response_id: str
    model: str
    initial_usage: InferenceUsage = InferenceUsage()


@dataclass(frozen=True, slots=True)
class TextBlockStarted:
    item_id: str
    block_id: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    block_id: str
    delta: str


@dataclass(frozen=True, slots=True)
class TextBlockCompleted:
    item_id: str
    block_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningBlockStarted:
    item_id: str
    block_id: str
    artifacts: tuple[ReplayArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    block_id: str
    delta: str


@dataclass(frozen=True, slots=True)
class ReasoningBlockCompleted:
    item_id: str
    block_id: str
    reasoning: str
    artifacts: tuple[ReplayArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    item_id: str
    block_id: str
    call_id: str
    kind: ToolCallKind
    name: str
    namespace: str | None = None
    artifacts: tuple[ReplayArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallArgumentsDelta:
    block_id: str
    delta: str


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    item_id: str
    block_id: str
    call_id: str
    kind: ToolCallKind
    name: str
    arguments: str
    namespace: str | None = None
    artifacts: tuple[ReplayArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class UsageUpdated:
    usage: InferenceUsage


@dataclass(frozen=True, slots=True)
class ResponseCompleted:
    finish_reason: FinishReason
    final_usage: InferenceUsage
    stop_sequence: str | None = None


type InferenceEvent = (
    ResponseStarted
    | TextBlockStarted
    | TextDelta
    | TextBlockCompleted
    | ReasoningBlockStarted
    | ReasoningDelta
    | ReasoningBlockCompleted
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallCompleted
    | UsageUpdated
    | ResponseCompleted
)


def inference_event_size(event: InferenceEvent) -> int:
    """Return deterministic compact UTF-8 size for private holdback accounting."""

    payload = _event_projection(event)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return len(encoded.encode("utf-8", errors="replace"))


def replay_payload_text(artifact: ReplayArtifact) -> str:
    """Render one opaque artifact payload for a wire format that requires text."""

    if isinstance(artifact.payload, str):
        return artifact.payload
    return json.dumps(
        _json_projection(artifact.payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _event_projection(event: InferenceEvent) -> JsonValue:
    if isinstance(event, ResponseStarted):
        return [
            "response_started",
            event.response_id,
            event.model,
            _usage(event.initial_usage),
        ]
    if isinstance(event, TextBlockStarted):
        return ["text_started", event.item_id, event.block_id]
    if isinstance(event, TextDelta):
        return ["text_delta", event.block_id, event.delta]
    if isinstance(event, TextBlockCompleted):
        return ["text_completed", event.item_id, event.block_id, event.text]
    if isinstance(event, ReasoningBlockStarted):
        return [
            "reasoning_started",
            event.item_id,
            event.block_id,
            _artifacts(event.artifacts),
        ]
    if isinstance(event, ReasoningDelta):
        return ["reasoning_delta", event.block_id, event.delta]
    if isinstance(event, ReasoningBlockCompleted):
        return [
            "reasoning_completed",
            event.item_id,
            event.block_id,
            event.reasoning,
            _artifacts(event.artifacts),
        ]
    if isinstance(event, ToolCallStarted):
        return [
            "tool_started",
            event.item_id,
            event.block_id,
            event.call_id,
            event.kind.value,
            event.namespace,
            event.name,
            _artifacts(event.artifacts),
        ]
    if isinstance(event, ToolCallArgumentsDelta):
        return ["tool_delta", event.block_id, event.delta]
    if isinstance(event, ToolCallCompleted):
        return [
            "tool_completed",
            event.item_id,
            event.block_id,
            event.call_id,
            event.kind.value,
            event.namespace,
            event.name,
            event.arguments,
            _artifacts(event.artifacts),
        ]
    if isinstance(event, UsageUpdated):
        return ["usage", _usage(event.usage)]
    return [
        "response_completed",
        event.finish_reason.value,
        event.stop_sequence,
        _usage(event.final_usage),
    ]


def _usage(usage: InferenceUsage) -> JsonValue:
    return {
        "input": _measurement(usage.input_tokens),
        "cache_read": _measurement(usage.cache_read_input_tokens),
        "cache_creation": _measurement(usage.cache_creation_input_tokens),
        "output": _measurement(usage.output_tokens),
        "reasoning_output": _measurement(usage.reasoning_output_tokens),
    }


def _measurement(measurement: TokenMeasurement | None) -> JsonValue:
    if measurement is None:
        return None
    return [measurement.value, measurement.source.value]


def _artifacts(artifacts: tuple[ReplayArtifact, ...]) -> JsonValue:
    return [
        [
            artifact.origin.value,
            artifact.kind.value,
            artifact.attachment.value,
            _json_projection(artifact.payload),
        ]
        for artifact in artifacts
    ]


def _json_projection(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_projection(item) for item in value]
    return value

"""Provider-neutral inference request values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort

from .events import ReplayArtifact, ToolCallKind


class CacheControlType(StrEnum):
    """Supported client prompt-cache breakpoint kind."""

    EPHEMERAL = "ephemeral"


class CacheTTL(StrEnum):
    """Supported Anthropic prompt-cache durations."""

    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"


@dataclass(frozen=True, slots=True)
class CacheControl:
    """One typed prompt-cache breakpoint."""

    type: CacheControlType = CacheControlType.EPHEMERAL
    ttl: CacheTTL | None = None


class InstructionOrigin(StrEnum):
    """Client role that supplied an instruction."""

    SYSTEM = "system"
    DEVELOPER = "developer"


class InstructionPlacement(StrEnum):
    """Whether an instruction was top-level or part of the transcript."""

    TOP_LEVEL = "top_level"
    TRANSCRIPT = "transcript"


class MessageRole(StrEnum):
    """Canonical conversational message roles."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class UrlMediaSource:
    """A media resource referenced by URL."""

    url: str

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("media URL must be non-empty")


@dataclass(frozen=True, slots=True)
class Base64MediaSource:
    """Inline base64 media with an explicit MIME type."""

    media_type: str
    data: str

    def __post_init__(self) -> None:
        if not self.media_type.strip():
            raise ValueError("base64 media_type must be non-empty")
        if not self.data.strip():
            raise ValueError("base64 media data must be non-empty")


@dataclass(frozen=True, slots=True)
class FileMediaSource:
    """A protocol-owned uploaded file reference."""

    file_id: str

    def __post_init__(self) -> None:
        if not self.file_id.strip():
            raise ValueError("file_id must be non-empty")


type MediaSource = UrlMediaSource | Base64MediaSource | FileMediaSource


@dataclass(frozen=True, slots=True)
class TextContent:
    """Visible text content."""

    text: str
    cache_control: CacheControl | None = None


@dataclass(frozen=True, slots=True)
class ImageContent:
    """User image content."""

    source: UrlMediaSource | Base64MediaSource
    cache_control: CacheControl | None = None


@dataclass(frozen=True, slots=True)
class DocumentContent:
    """User document content retained for transport preflight."""

    source: MediaSource
    cache_control: CacheControl | None = None


@dataclass(frozen=True, slots=True)
class RefusalContent:
    """A prior assistant refusal preserved as such."""

    refusal: str


type MessageContent = TextContent | ImageContent | DocumentContent | RefusalContent


@dataclass(frozen=True, slots=True)
class InstructionItem:
    """One ordered instruction."""

    text: str
    origin: InstructionOrigin
    placement: InstructionPlacement
    cache_control: CacheControl | None = None
    turn_id: str | None = None

    def __post_init__(self) -> None:
        if (self.placement is InstructionPlacement.TRANSCRIPT) != (
            self.turn_id is not None
        ):
            raise ValueError("only transcript instructions require a turn_id")


@dataclass(frozen=True, slots=True)
class MessageItem:
    """One conversational turn's visible content."""

    turn_id: str
    role: MessageRole
    content: tuple[MessageContent, ...]


@dataclass(frozen=True, slots=True)
class ReasoningItem:
    """Visible reasoning and its opaque continuation artifacts."""

    turn_id: str
    reasoning: str
    artifacts: tuple[ReplayArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallItem:
    """One prior assistant tool invocation."""

    turn_id: str
    call_id: str
    kind: ToolCallKind
    name: str
    input: JsonValue
    namespace: str | None = None
    artifacts: tuple[ReplayArtifact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", freeze_json(self.input))


@dataclass(frozen=True, slots=True)
class ToolResultItem:
    """One user-supplied result for a prior tool invocation."""

    turn_id: str
    call_id: str
    content: JsonValue
    is_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", freeze_json(self.content))


type InferenceItem = (
    InstructionItem | MessageItem | ReasoningItem | ToolCallItem | ToolResultItem
)


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """A JSON-schema function tool."""

    name: str
    description: str | None
    input_schema: Mapping[str, JsonValue]
    strict: bool = False
    namespace: str | None = None
    cache_control: CacheControl | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", freeze_json_object(self.input_schema))


class CustomToolFormatType(StrEnum):
    """Supported free-form tool input formats."""

    TEXT = "text"
    GRAMMAR = "grammar"


@dataclass(frozen=True, slots=True)
class CustomToolFormat:
    """Typed custom-tool input format."""

    type: CustomToolFormatType
    syntax: str | None = None
    definition: str | None = None


@dataclass(frozen=True, slots=True)
class CustomTool:
    """A free-form/custom tool."""

    name: str
    description: str | None
    format: CustomToolFormat
    namespace: str | None = None
    cache_control: CacheControl | None = None


type InferenceTool = FunctionTool | CustomTool


class ToolChoiceMode(StrEnum):
    """Canonical tool selection modes."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    SPECIFIC = "specific"


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """Tool-selection intent with an optional typed target."""

    mode: ToolChoiceMode
    kind: ToolCallKind | None = None
    name: str | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        specific = self.mode is ToolChoiceMode.SPECIFIC
        if specific != (self.kind is not None and self.name is not None):
            raise ValueError("specific tool choice requires kind and name")
        if not specific and self.namespace is not None:
            raise ValueError("only a specific tool choice may name a namespace")


@dataclass(frozen=True, slots=True)
class ClientReasoningIntent:
    """Unresolved reasoning intent supplied by the client."""

    control: ReasoningControl = ReasoningControl.DEFAULT
    effort: ReasoningEffort | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.budget_tokens is not None and (
            not isinstance(self.budget_tokens, int)
            or isinstance(self.budget_tokens, bool)
            or self.budget_tokens <= 0
        ):
            raise ValueError("reasoning budget must be a positive integer")
        if self.budget_tokens is not None and self.control is not ReasoningControl.ON:
            raise ValueError("a reasoning budget requires reasoning control to be on")


@dataclass(frozen=True, slots=True)
class OpenAIChatExtension:
    """Explicit caller extension for OpenAI-compatible Chat transports."""

    extra_body: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_body", freeze_json_object(self.extra_body))


type InferenceExtension = OpenAIChatExtension


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Immutable provider-neutral request passed below protocol ingress."""

    model: str
    items: tuple[InferenceItem, ...]
    tools: tuple[InferenceTool, ...] = ()
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] = ()
    reasoning: ClientReasoningIntent = ClientReasoningIntent()
    metadata: Mapping[str, JsonValue] | None = None
    extensions: tuple[InferenceExtension, ...] = ()

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.max_output_tokens is not None and (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if self.metadata is not None:
            object.__setattr__(self, "metadata", freeze_json_object(self.metadata))
        if sum(isinstance(value, OpenAIChatExtension) for value in self.extensions) > 1:
            raise ValueError("only one OpenAI Chat extension is allowed")

    @property
    def message_count(self) -> int:
        """Return the number of distinct transcript turns."""

        turn_ids: set[str] = set()
        for item in self.items:
            if isinstance(item, InstructionItem):
                if item.turn_id is not None:
                    turn_ids.add(item.turn_id)
            elif isinstance(
                item,
                MessageItem | ReasoningItem | ToolCallItem | ToolResultItem,
            ):
                turn_ids.add(item.turn_id)
        return len(turn_ids)

    @property
    def openai_chat_extension(self) -> OpenAIChatExtension | None:
        """Return the one explicit Chat extension, if supplied."""

        return next(
            (
                extension
                for extension in self.extensions
                if isinstance(extension, OpenAIChatExtension)
            ),
            None,
        )

    def with_stop_sequences(self, stop_sequences: tuple[str, ...]) -> InferenceRequest:
        """Return a request with an application-owned stop-policy adjustment."""

        return replace(self, stop_sequences=stop_sequences)


def freeze_json(value: JsonValue) -> JsonValue:
    """Return a recursively immutable, owned JSON value."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(freeze_json(item) for item in value)
    return value


def freeze_json_object(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """Return a recursively immutable, owned JSON object."""

    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("expected a JSON object")
    return frozen


def thaw_json(value: JsonValue) -> JsonValue:
    """Return a mutable JSON-ready copy of a canonical value."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [thaw_json(item) for item in value]
    return value


def thaw_json_object(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a mutable JSON-ready copy of a canonical object."""

    return {str(key): thaw_json(item) for key, item in value.items()}


def inference_request_snapshot(request: InferenceRequest) -> dict[str, object]:
    """Return trace-safe request structure without prompts or replay payloads."""

    return {
        "model": request.model,
        "item_count": len(request.items),
        "message_count": request.message_count,
        "tool_count": len(request.tools),
        "tool_choice": (
            request.tool_choice.mode.value if request.tool_choice is not None else None
        ),
        "parallel_tool_calls": request.parallel_tool_calls,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "stop_sequence_count": len(request.stop_sequences),
        "reasoning_control": request.reasoning.control.value,
        "reasoning_effort": (
            request.reasoning.effort.value
            if request.reasoning.effort is not None
            else None
        ),
        "reasoning_budget_tokens": request.reasoning.budget_tokens,
        "has_metadata": request.metadata is not None,
        "extension_count": len(request.extensions),
    }

"""Precise domain values for local Chat Sessions."""

from dataclasses import dataclass
from enum import StrEnum

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer accurately and clearly, using Markdown "
    "when it improves readability."
)


class ChatReasoning(StrEnum):
    """Reasoning controls exposed by the Chat UI."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"

    def policy(self) -> ReasoningPolicy:
        """Return the provider-neutral policy for one Chat selection."""

        if self is ChatReasoning.OFF:
            return ReasoningPolicy.off()
        return ReasoningPolicy.on(effort=ReasoningEffort(self.value))


class GenerationStatus(StrEnum):
    """Durable terminal and in-progress generation states."""

    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class SegmentKind(StrEnum):
    """Visible assistant segment kinds retained in exact stream order."""

    THINKING = "thinking"
    TEXT = "text"


class ChatOperationKind(StrEnum):
    """Long-running Chat commands owned by the application."""

    SEND = "send"
    RETRY = "retry"
    REGENERATE = "regenerate"
    COMPACT = "compact"


class ChatOperationPhase(StrEnum):
    """Current work phase exposed to Chat observers."""

    GENERATING = "generating"
    COMPACTING = "compacting"


@dataclass(frozen=True, slots=True)
class ChatPreferences:
    system_prompt: str
    last_model: str | None
    last_reasoning: ChatReasoning
    updated_at: int


@dataclass(frozen=True, slots=True)
class ChatSession:
    id: str
    title: str
    model: str
    reasoning: ChatReasoning
    revision: int
    created_at: int
    updated_at: int


@dataclass(frozen=True, slots=True)
class ChatSessionSummary:
    id: str
    title: str
    model: str
    reasoning: ChatReasoning
    revision: int
    preview: str
    created_at: int
    updated_at: int


@dataclass(frozen=True, slots=True)
class ChatSessionPage:
    sessions: tuple[ChatSessionSummary, ...]
    next_cursor: tuple[int, str] | None


@dataclass(frozen=True, slots=True)
class ChatSegment:
    ordinal: int
    kind: SegmentKind
    text: str


@dataclass(frozen=True, slots=True)
class ChatGeneration:
    id: str
    status: GenerationStatus
    requested_model: str
    actual_model: str | None
    reasoning: ChatReasoning
    effective_output_limit: int
    stop_reason: str | None
    error_code: str | None
    error_message: str | None
    started_at: int
    finished_at: int | None
    segments: tuple[ChatSegment, ...]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    id: str
    session_id: str
    operation_id: str
    sequence: int
    user_text: str
    created_at: int
    generation: ChatGeneration


@dataclass(frozen=True, slots=True)
class ChatCompaction:
    session_id: str
    covered_through_sequence: int
    summary: str
    estimated_tokens: int
    requested_model: str
    actual_model: str
    updated_at: int


@dataclass(frozen=True, slots=True)
class ChatTranscript:
    session: ChatSession
    turns: tuple[ChatTurn, ...]
    compaction: ChatCompaction | None


@dataclass(frozen=True, slots=True)
class ChatModelOption:
    model_ref: str
    provider_id: str
    model_id: str
    supports_reasoning: bool | None
    input_modalities: frozenset[ModelInputModality] | None
    context_window_tokens: int | None
    max_output_tokens: int | None


@dataclass(frozen=True, slots=True)
class ChatContextEstimate:
    estimated_input_tokens: int
    completion_tokens: int
    context_window_tokens: int | None
    usable_input_tokens: int | None
    usage_ratio: float | None
    should_auto_compact: bool
    can_compact: bool


@dataclass(frozen=True, slots=True)
class ChatOperationAcknowledgement:
    session_id: str
    operation_id: str
    kind: ChatOperationKind


@dataclass(frozen=True, slots=True)
class ChatActiveOperation:
    session_id: str
    operation_id: str
    kind: ChatOperationKind
    phase: ChatOperationPhase
    operation_sequence: int
    submitted_text: str | None
    turn_id: str | None
    generation_id: str | None
    regeneration: bool
    actual_model: str | None
    segments: tuple[ChatSegment, ...]


@dataclass(frozen=True, slots=True)
class ChatSessionDetail:
    session: ChatSession
    turns: tuple[ChatTurn, ...]
    next_before: int | None
    compaction: ChatCompaction | None
    context: ChatContextEstimate | None
    context_error: str | None


@dataclass(frozen=True, slots=True)
class ChatPublishedEvent:
    event: str
    id: int
    data: JsonObject


class ChatError(Exception):
    """Base class for application-owned Chat failures."""


class ChatUnavailableError(ChatError):
    """Chat state or lifecycle is unavailable while the proxy remains usable."""


class ChatNotFoundError(ChatError):
    """A requested Chat-owned resource does not exist."""


class ChatConflictError(ChatError):
    """The requested mutation conflicts with current session state."""


class ChatValidationError(ChatError):
    """The requested Chat operation cannot be executed as supplied."""


class ChatEventOverflowError(ChatError):
    """One slow Chat observer must reconnect from a fresh snapshot."""

    def __init__(self, cursor: int) -> None:
        super().__init__("Chat event subscription overflowed.")
        self.cursor = cursor

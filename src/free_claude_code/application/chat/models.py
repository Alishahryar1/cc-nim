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

MAX_CHAT_ATTACHMENTS_PER_TURN = 5
MAX_CHAT_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_CHAT_ATTACHMENTS_COMBINED_BYTES = 25 * 1024 * 1024
MAX_CHAT_ATTACHMENT_EXTRACTED_CHARACTERS = 1_000_000
MAX_CHAT_ATTACHMENTS_COMBINED_EXTRACTED_CHARACTERS = 2_000_000


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


class ChatAttachmentKind(StrEnum):
    """Portable attachment categories accepted by local Chat Sessions."""

    IMAGE = "image"
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"


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
class ChatAttachment:
    """Durable attachment metadata; file content remains outside SQLite."""

    id: str
    session_id: str
    turn_id: str | None
    position: int
    filename: str
    kind: ChatAttachmentKind
    media_type: str
    byte_size: int
    extracted_characters: int | None
    created_at: int
    available: bool = True


@dataclass(frozen=True, slots=True)
class ChatAttachmentFileInfo:
    """Verified file facts produced before an attachment row is committed."""

    kind: ChatAttachmentKind
    media_type: str
    byte_size: int
    extracted_characters: int | None


@dataclass(frozen=True, slots=True)
class ChatImageAttachment:
    """Materialized image input supplied to the pure context builder."""

    attachment: ChatAttachment
    data: bytes


@dataclass(frozen=True, slots=True)
class ChatDocumentAttachment:
    """Materialized portable text extracted from a user document."""

    attachment: ChatAttachment
    text: str


type ChatAttachmentMaterial = ChatImageAttachment | ChatDocumentAttachment


@dataclass(frozen=True, slots=True)
class ChatAttachmentContent:
    """Verified original bytes returned by the protected content route."""

    attachment: ChatAttachment
    data: bytes


@dataclass(frozen=True, slots=True)
class ChatTurn:
    id: str
    session_id: str
    operation_id: str
    sequence: int
    user_text: str
    created_at: int
    generation: ChatGeneration
    attachments: tuple[ChatAttachment, ...] = ()


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
class ChatSessionDetail:
    session: ChatSession
    turns: tuple[ChatTurn, ...]
    next_before: int | None
    compaction: ChatCompaction | None
    context: ChatContextEstimate | None
    context_error: str | None
    active_operation: bool
    staged_attachments: tuple[ChatAttachment, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    event: str
    sequence: int
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


class ChatPayloadTooLargeError(ChatValidationError):
    """A Chat attachment exceeds a server-owned byte or count limit."""


class ChatUnsupportedAttachmentError(ChatValidationError):
    """A Chat attachment cannot be represented by the portable input contract."""

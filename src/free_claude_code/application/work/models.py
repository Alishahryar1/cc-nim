"""Precise domain values for local Codex-backed Work Sessions."""

from dataclasses import dataclass
from enum import StrEnum

from free_claude_code.core.json_types import JsonObject


class WorkStatus(StrEnum):
    READY = "ready"
    WORKING = "working"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_INPUT = "waiting_for_input"
    STOPPING = "stopping"
    DELETING = "deleting"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


class WorkOperationKind(StrEnum):
    CREATE = "create"
    SEND = "send"
    STOP = "stop"
    DELETE = "delete"


class WorkOperationState(StrEnum):
    RESERVED = "reserved"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class WorkInteractionKind(StrEnum):
    COMMAND_APPROVAL = "command_approval"
    FILE_CHANGE_APPROVAL = "file_change_approval"
    PERMISSION_APPROVAL = "permission_approval"
    USER_INPUT = "user_input"


@dataclass(frozen=True, slots=True)
class WorkSessionSettings:
    model: str | None
    reasoning_effort: str | None
    collaboration_mode: str | None
    permission_profile: str | None


@dataclass(frozen=True, slots=True)
class WorkSessionRecord:
    thread_id: str
    cwd: str
    cwd_key: str
    settings: WorkSessionSettings
    revision: int
    registered_at_ms: int


@dataclass(frozen=True, slots=True)
class WorkSessionSummary:
    thread_id: str
    cwd: str
    title: str
    preview: str
    status: WorkStatus
    revision: int
    registered_at_ms: int
    updated_at_ms: int | None
    project_available: bool
    session_available: bool


@dataclass(frozen=True, slots=True)
class WorkSessionPage:
    sessions: tuple[WorkSessionSummary, ...]
    next_cursor: tuple[int, str] | None


@dataclass(frozen=True, slots=True)
class WorkTimelineItem:
    thread_id: str
    turn_id: str
    item_id: str
    kind: str
    status: str | None
    text: str | None
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class WorkTurnPage:
    items: tuple[WorkTimelineItem, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class WorkInteraction:
    interaction_id: str
    thread_id: str
    turn_id: str | None
    kind: WorkInteractionKind
    title: str
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class WorkSessionDetail:
    summary: WorkSessionSummary
    settings: WorkSessionSettings
    controls: JsonObject
    turns: WorkTurnPage
    live_items: tuple[WorkTimelineItem, ...]
    interactions: tuple[WorkInteraction, ...]
    event_cursor: int


@dataclass(frozen=True, slots=True)
class WorkOperation:
    operation_id: str
    kind: WorkOperationKind
    session_id: str | None
    intent_digest: str
    state: WorkOperationState
    result_thread_id: str | None
    result_turn_id: str | None
    error_code: str | None
    error_message: str | None
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True, slots=True)
class WorkOperationAcknowledgement:
    operation_id: str
    kind: WorkOperationKind
    state: WorkOperationState
    thread_id: str | None
    turn_id: str | None


@dataclass(frozen=True, slots=True)
class WorkBootstrap:
    available: bool
    reason: str | None
    codex_version: str | None
    recent_projects: tuple[str, ...]
    event_generation: str
    event_cursor: int


class WorkError(Exception):
    """Base class for application-owned Work failures."""


class WorkUnavailableError(WorkError):
    """Work state or Codex is unavailable while the proxy remains usable."""


class WorkCompatibilityError(WorkError):
    """The installed Codex is too old for the Work contract."""


class WorkNotFoundError(WorkError):
    """A requested Work-owned resource does not exist."""


class WorkConflictError(WorkError):
    """A mutation conflicts with current Work state."""


class WorkValidationError(WorkError):
    """A Work command cannot be executed as supplied."""

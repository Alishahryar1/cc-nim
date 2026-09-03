"""Domain values for process-lifetime Terminal Sessions."""

from dataclasses import dataclass
from enum import StrEnum


class TerminalStatus(StrEnum):
    """Lifecycle states exposed to attached terminal views."""

    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"


@dataclass(frozen=True, slots=True)
class TerminalSession:
    """Current observable state for one server-owned terminal."""

    id: str
    name: str
    status: TerminalStatus
    created_at: int
    rows: int
    columns: int
    exit_code: int | None
    error: str | None
    history_truncated: bool


@dataclass(frozen=True, slots=True)
class TerminalAttachmentSnapshot:
    """Atomic retained state delivered before one attachment follows live output."""

    session: TerminalSession
    output: bytes


@dataclass(frozen=True, slots=True)
class TerminalOutputEvent:
    data: bytes


@dataclass(frozen=True, slots=True)
class TerminalStateEvent:
    session: TerminalSession


@dataclass(frozen=True, slots=True)
class TerminalDeletedEvent:
    pass


type TerminalAttachmentEvent = (
    TerminalOutputEvent | TerminalStateEvent | TerminalDeletedEvent
)


class TerminalError(Exception):
    """Base class for application-owned terminal failures."""


class TerminalUnavailableError(TerminalError):
    """Terminal Sessions cannot currently perform the requested operation."""


class TerminalNotFoundError(TerminalError):
    """A requested process-lifetime terminal does not exist."""


class TerminalConflictError(TerminalError):
    """The requested action conflicts with the terminal lifecycle."""


class TerminalValidationError(TerminalError):
    """Terminal input or metadata is invalid."""


class TerminalAttachmentOverflowError(TerminalError):
    """One slow terminal attachment must reconnect from retained output."""

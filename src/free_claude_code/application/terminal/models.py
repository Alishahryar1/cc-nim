"""Domain values for process-lifetime Terminal Sessions."""

from dataclasses import dataclass
from enum import StrEnum


class TerminalStatus(StrEnum):
    """Lifecycle states exposed to attached terminal views."""

    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"


class TerminalClientRole(StrEnum):
    """A browser view's relationship to the shared terminal pane."""

    CONTROLLER = "controller"
    OBSERVER = "observer"


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


@dataclass(frozen=True, slots=True)
class TerminalAttachmentSnapshot:
    """Atomic rendered state delivered before one view follows live output."""

    session: TerminalSession
    output: bytes
    role: TerminalClientRole


@dataclass(frozen=True, slots=True)
class TerminalEngineSnapshot:
    """Terminal-engine render split into off-screen history and live viewport."""

    scrollback: bytes
    viewport: bytes

    @property
    def rendered(self) -> bytes:
        return self.scrollback + self.viewport


@dataclass(frozen=True, slots=True)
class TerminalOutputEvent:
    data: bytes


@dataclass(frozen=True, slots=True)
class TerminalResetEvent:
    """Replace a view after reconnect or controller transfer."""

    output: bytes
    role: TerminalClientRole


@dataclass(frozen=True, slots=True)
class TerminalStateEvent:
    session: TerminalSession


@dataclass(frozen=True, slots=True)
class TerminalDeletedEvent:
    pass


type TerminalAttachmentEvent = (
    TerminalOutputEvent | TerminalResetEvent | TerminalStateEvent | TerminalDeletedEvent
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

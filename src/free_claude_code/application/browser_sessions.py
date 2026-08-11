"""Application boundary for locally managed browser terminal sessions."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class BrowserSessionHarness(StrEnum):
    """Native coding harnesses supported by the browser terminal."""

    CLAUDE = "claude"
    CODEX = "codex"
    PI = "pi"


class BrowserSessionState(StrEnum):
    """Observable lifecycle state for one native harness process."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HarnessAvailability:
    """Whether one native harness can be launched on this machine."""

    harness: BrowserSessionHarness
    available: bool
    message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.harness.value,
            "available": self.available,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BrowserSessionView:
    """Safe public state for one persisted session."""

    session_id: str
    name: str
    harness: BrowserSessionHarness
    state: BrowserSessionState
    project_name: str
    path: str
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.session_id,
            "name": self.name,
            "harness": self.harness.value,
            "state": self.state.value,
            "project_name": self.project_name,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class BrowserSessionSnapshot:
    """Complete local Sessions catalog returned to the Admin UI."""

    available: bool
    sessions: tuple[BrowserSessionView, ...]
    harnesses: tuple[HarnessAvailability, ...]
    message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "message": self.message,
            "harnesses": [harness.as_dict() for harness in self.harnesses],
            "sessions": [session.as_dict() for session in self.sessions],
        }


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """One terminal output chunk or lifecycle control notification."""

    kind: str
    data: bytes | None = None
    state: BrowserSessionState | None = None
    detail: str | None = None


class BrowserTerminalPort(Protocol):
    """One exclusive browser attachment to a managed terminal."""

    async def receive(self) -> TerminalEvent: ...

    async def write(self, data: str) -> None: ...

    async def resize(self, columns: int, rows: int) -> None: ...

    async def close(self) -> None: ...


class BrowserSessionPort(Protocol):
    """Browser-session operations exposed to the HTTP adapter."""

    async def snapshot(self) -> BrowserSessionSnapshot: ...

    async def create_session(
        self,
        path: str,
        harness: BrowserSessionHarness,
        name: str | None = None,
    ) -> BrowserSessionView: ...

    async def rename_session(
        self, session_id: str, name: str
    ) -> BrowserSessionView: ...

    async def start_session(self, session_id: str) -> BrowserSessionView: ...

    async def stop_session(self, session_id: str) -> BrowserSessionView: ...

    async def delete_session(self, session_id: str) -> None: ...

    async def attach_terminal(self, session_id: str) -> BrowserTerminalPort: ...


class BrowserSessionLifecyclePort(Protocol):
    """Browser-session work owned by the application resource graph."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...


class BrowserSessionError(Exception):
    """Base exception mapped by the browser-session HTTP adapter."""


class BrowserSessionValidationError(BrowserSessionError):
    """Customer input does not satisfy the session contract."""


class BrowserSessionNotFoundError(BrowserSessionError):
    """Requested project or session does not exist."""


class BrowserSessionConflictError(BrowserSessionError):
    """Requested operation conflicts with current state or identity."""


class BrowserSessionUnavailableError(BrowserSessionError):
    """The Sessions capability or selected harness is unavailable."""

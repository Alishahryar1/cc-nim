"""Narrow ports owned by process-lifetime Terminal Sessions."""

from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Protocol

from .models import (
    TerminalAttachmentEvent,
    TerminalAttachmentSnapshot,
    TerminalSession,
)


class TerminalProcessPort(Protocol):
    """One platform PTY process owned by the Terminal application."""

    @property
    def pid(self) -> int: ...

    @property
    def alive(self) -> bool: ...

    async def read(self) -> bytes: ...

    async def write(self, data: bytes) -> None: ...

    async def resize(self, rows: int, columns: int) -> None: ...

    async def wait(self) -> int | None: ...

    async def terminate_tree(self) -> None: ...

    async def close(self) -> None: ...


class TerminalProcessFactoryPort(Protocol):
    """Platform boundary that creates an interactive shell inside a PTY."""

    async def spawn(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> TerminalProcessPort: ...


class TerminalAttachmentPort(Protocol):
    """One transient browser attachment to a server-owned terminal."""

    @property
    def initial(self) -> TerminalAttachmentSnapshot: ...

    def __aiter__(self) -> AsyncIterator[TerminalAttachmentEvent]: ...

    async def aclose(self) -> None: ...


class TerminalApplicationPort(Protocol):
    """Complete Terminal Sessions capability consumed by the Admin API."""

    async def create_session(self) -> TerminalSession: ...

    async def list_sessions(self) -> tuple[TerminalSession, ...]: ...

    async def get_session(self, session_id: str) -> TerminalSession: ...

    async def rename_session(self, session_id: str, name: str) -> TerminalSession: ...

    async def attach(self, session_id: str) -> TerminalAttachmentPort: ...

    async def write(self, session_id: str, data: bytes) -> None: ...

    async def resize(
        self, session_id: str, *, rows: int, columns: int
    ) -> TerminalSession: ...

    async def stop_session(self, session_id: str) -> TerminalSession: ...

    async def delete_session(self, session_id: str) -> None: ...

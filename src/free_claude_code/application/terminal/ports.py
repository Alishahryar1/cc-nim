"""Narrow ports owned by process-lifetime Terminal Sessions."""

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .models import (
    TerminalAttachmentEvent,
    TerminalAttachmentSnapshot,
    TerminalClientRole,
    TerminalEngineSnapshot,
    TerminalSession,
)


class TerminalClientPort(Protocol):
    """One transient terminal-engine client hosted inside a platform PTY."""

    async def read(self) -> bytes: ...

    async def write(self, data: str) -> None: ...

    async def resize(self, rows: int, columns: int) -> None: ...

    async def wait(self) -> int | None: ...

    async def close(self) -> None: ...


class TerminalEngineSessionPort(Protocol):
    """One canonical terminal pane owned by the runtime engine."""

    async def open_client(
        self,
        role: TerminalClientRole,
        *,
        rows: int,
        columns: int,
    ) -> TerminalClientPort: ...

    async def snapshot(self) -> TerminalEngineSnapshot: ...

    async def resize(self, rows: int, columns: int) -> None: ...

    async def wait_root(self) -> int | None: ...

    async def terminate_tree(self) -> None: ...

    async def close(self) -> None: ...


class TerminalEngineHostPort(Protocol):
    """Runtime-owned terminal multiplexer installation and process namespace."""

    async def start(self) -> None: ...

    async def create_session(
        self,
        *,
        session_name: str,
        command: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> TerminalEngineSessionPort: ...

    async def close(self) -> None: ...


class TerminalAttachmentPort(Protocol):
    """One transient browser attachment to a server-owned terminal."""

    @property
    def initial(self) -> TerminalAttachmentSnapshot: ...

    def __aiter__(self) -> AsyncIterator[TerminalAttachmentEvent]: ...

    async def claim(self) -> None: ...

    async def write(self, data: str) -> None: ...

    async def resize(self, *, rows: int, columns: int) -> None: ...

    async def aclose(self) -> None: ...


class TerminalApplicationPort(Protocol):
    """Complete Terminal Sessions capability consumed by the Admin API."""

    @property
    def availability_error(self) -> str | None: ...

    async def create_session(self) -> TerminalSession: ...

    async def list_sessions(self) -> tuple[TerminalSession, ...]: ...

    async def get_session(self, session_id: str) -> TerminalSession: ...

    async def rename_session(self, session_id: str, name: str) -> TerminalSession: ...

    async def attach(
        self, session_id: str, *, rows: int, columns: int
    ) -> TerminalAttachmentPort: ...

    async def stop_session(self, session_id: str) -> TerminalSession: ...

    async def delete_session(self, session_id: str) -> None: ...

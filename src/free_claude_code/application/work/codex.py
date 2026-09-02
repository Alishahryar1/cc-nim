"""Concrete application-facing contract for Codex Direct mode."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from free_claude_code.core.json_types import JsonObject, JsonValue

type CodexRequestId = int | str
type CodexApprovalPolicy = str | JsonObject


@dataclass(frozen=True, slots=True)
class CodexAvailability:
    """Installed Codex availability without starting its app-server."""

    available: bool
    binary_path: str | None
    version: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CodexInitialization:
    """Connection metadata returned by the app-server handshake."""

    connection_id: str
    user_agent: str
    codex_home: str
    platform_family: str
    platform_os: str


@dataclass(frozen=True, slots=True)
class CodexControlCatalog:
    """Native controls exposed by the installed Codex version."""

    models: tuple[JsonObject, ...] | None
    collaboration_modes: tuple[JsonObject, ...] | None
    permission_profiles: tuple[JsonObject, ...] | None
    config: JsonObject | None


@dataclass(frozen=True, slots=True)
class CodexThreadSettings:
    """Codex-native settings applied when a thread is started or resumed."""

    cwd: str
    model: str | None = None
    approval_policy: CodexApprovalPolicy | None = None
    permission_profile: str | None = None


@dataclass(frozen=True, slots=True)
class CodexTurnSettings:
    """Codex-native settings that may be changed between turns."""

    model: str | None = None
    effort: str | None = None
    collaboration_mode: JsonObject | None = None
    approval_policy: CodexApprovalPolicy | None = None
    permission_profile: str | None = None


@dataclass(frozen=True, slots=True)
class CodexThreadHandle:
    """Native thread identity plus the complete start/resume response."""

    connection_id: str
    thread_id: str
    response: JsonObject


@dataclass(frozen=True, slots=True)
class CodexThreadSnapshot:
    """Native thread metadata returned without deprecated history hydration."""

    thread_id: str
    thread: JsonObject


@dataclass(frozen=True, slots=True)
class CodexObjectPage:
    """One opaque native page with Codex-owned continuation cursors."""

    records: tuple[JsonObject, ...]
    next_cursor: str | None
    backwards_cursor: str | None


@dataclass(frozen=True, slots=True)
class CodexTurnHandle:
    """Native turn identity plus the complete start response."""

    connection_id: str
    thread_id: str
    turn_id: str
    response: JsonObject


@dataclass(frozen=True, slots=True)
class CodexNotification:
    """A native Codex notification with its complete params payload."""

    connection_id: str
    method: str
    params: JsonValue


@dataclass(frozen=True, slots=True)
class CodexServerRequest:
    """An interactive app-server request that requires a client response."""

    connection_id: str
    request_id: CodexRequestId
    method: str
    params: JsonValue


@dataclass(frozen=True, slots=True)
class CodexConnectionLost:
    """Terminal event for one app-server connection generation."""

    connection_id: str
    message: str


@dataclass(frozen=True, slots=True)
class CodexUnsupportedInteraction:
    """A server request whose response contract FCC does not implement."""

    connection_id: str
    method: str


type CodexAppServerEvent = (
    CodexNotification
    | CodexServerRequest
    | CodexUnsupportedInteraction
    | CodexConnectionLost
)


class CodexDirectError(Exception):
    """Base class for the concrete Codex Direct boundary."""


class CodexUnavailableError(CodexDirectError):
    """Codex is missing, closed, or cannot be launched."""


class CodexCompatibilityError(CodexDirectError):
    """The installed Codex lacks a required app-server contract."""


class CodexProtocolError(CodexDirectError):
    """The child emitted malformed or invalid protocol data."""


class CodexConnectionError(CodexDirectError):
    """The app-server connection ended before an operation completed."""


class CodexRequestError(CodexDirectError):
    """Codex rejected one otherwise valid app-server request."""

    def __init__(self, *, method: str, code: int, message: str) -> None:
        super().__init__(f"Codex {method} failed ({code}): {message}")
        self.method = method
        self.code = code
        self.message = message


class CodexAppServerPort(Protocol):
    """Native Codex operations required by the future Work application."""

    async def availability(self) -> CodexAvailability: ...

    async def initialize(self) -> CodexInitialization: ...

    async def controls(self, *, cwd: str) -> CodexControlCatalog: ...

    async def start_thread(
        self, settings: CodexThreadSettings
    ) -> CodexThreadHandle: ...

    async def resume_thread(
        self, thread_id: str, settings: CodexThreadSettings
    ) -> CodexThreadHandle: ...

    async def read_thread(self, thread_id: str) -> CodexThreadSnapshot: ...

    async def list_threads_page(
        self, *, cursor: str | None, limit: int
    ) -> CodexObjectPage: ...

    async def list_turns_page(
        self,
        *,
        thread_id: str,
        cursor: str | None,
        limit: int,
    ) -> CodexObjectPage: ...

    async def set_thread_name(self, *, thread_id: str, name: str) -> None: ...

    async def delete_thread(self, thread_id: str) -> None: ...

    async def start_turn(
        self,
        *,
        thread_id: str,
        text: str,
        settings: CodexTurnSettings,
        client_user_message_id: str | None = None,
    ) -> CodexTurnHandle: ...

    async def interrupt_turn(self, *, thread_id: str, turn_id: str) -> None: ...

    async def respond(
        self,
        *,
        connection_id: str,
        request_id: CodexRequestId,
        result: JsonValue,
    ) -> None: ...

    def events(self) -> AsyncIterator[CodexAppServerEvent]: ...

    async def close(self) -> None: ...

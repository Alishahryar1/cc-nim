"""Concrete application-facing contract for Codex Direct mode."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from free_claude_code.core.json_types import JsonObject, JsonValue

type CodexRequestId = int | str


class CodexDelivery(StrEnum):
    """What is known after a JSON-RPC call loses its response."""

    DEFINITELY_NOT_WRITTEN = "definitely_not_written"
    POSSIBLY_WRITTEN = "possibly_written"


class CodexInteractionKind(StrEnum):
    COMMAND_APPROVAL = "command_approval"
    FILE_CHANGE_APPROVAL = "file_change_approval"
    PERMISSION_APPROVAL = "permission_approval"
    USER_INPUT = "user_input"


@dataclass(frozen=True, slots=True)
class CodexAvailability:
    available: bool
    binary_path: str | None
    version: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CodexInitialization:
    connection_id: str
    user_agent: str
    codex_home: str
    platform_family: str
    platform_os: str


@dataclass(frozen=True, slots=True)
class CodexControlCatalog:
    """Optional read-only controls exposed by the installed Codex version."""

    models: tuple[JsonObject, ...] | None
    config: JsonObject | None


@dataclass(frozen=True, slots=True)
class CodexThreadSettings:
    cwd: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class CodexTurnSettings:
    model: str
    effort: str | None = None


@dataclass(frozen=True, slots=True)
class CodexThreadHandle:
    connection_id: str
    thread_id: str
    response: JsonObject


@dataclass(frozen=True, slots=True)
class CodexThreadSnapshot:
    thread_id: str
    thread: JsonObject


@dataclass(frozen=True, slots=True)
class CodexObjectPage:
    records: tuple[JsonObject, ...]
    next_cursor: str | None
    backwards_cursor: str | None


@dataclass(frozen=True, slots=True)
class CodexTurnHandle:
    connection_id: str
    thread_id: str
    turn_id: str
    response: JsonObject


@dataclass(frozen=True, slots=True)
class CodexNotification:
    connection_id: str
    method: str
    params: JsonValue


@dataclass(frozen=True, slots=True)
class CodexInteractionRequest:
    """Validated interactive request that requires a typed response."""

    connection_id: str
    request_id: CodexRequestId
    method: str
    thread_id: str
    turn_id: str | None
    kind: CodexInteractionKind
    params: JsonObject


@dataclass(frozen=True, slots=True)
class CodexInteractionResponse:
    kind: CodexInteractionKind
    result: JsonObject


@dataclass(frozen=True, slots=True)
class CodexConnectionLost:
    connection_id: str
    message: str


@dataclass(frozen=True, slots=True)
class CodexUnsupportedInteraction:
    connection_id: str
    method: str


type CodexAppServerEvent = (
    CodexNotification
    | CodexInteractionRequest
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

    def __init__(
        self,
        message: str,
        *,
        delivery: CodexDelivery = CodexDelivery.DEFINITELY_NOT_WRITTEN,
    ) -> None:
        super().__init__(message)
        self.delivery = delivery


class CodexRequestError(CodexDirectError):
    """Codex rejected one otherwise valid app-server request."""

    def __init__(self, *, method: str, code: int, message: str) -> None:
        super().__init__(f"Codex {method} failed ({code}): {message}")
        self.method = method
        self.code = code
        self.message = message


class CodexAppServerPort(Protocol):
    async def availability(self) -> CodexAvailability: ...

    async def initialize(self) -> CodexInitialization: ...

    async def controls(self, *, cwd: str) -> CodexControlCatalog: ...

    async def start_thread(
        self, settings: CodexThreadSettings
    ) -> CodexThreadHandle: ...

    async def materialize_thread(self, thread_id: str) -> None: ...

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
        response: CodexInteractionResponse,
    ) -> None: ...

    def events(self) -> AsyncIterator[CodexAppServerEvent]: ...

    async def close(self) -> None: ...

"""Persistence and native execution required by Code sessions."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from free_claude_code.application.session_events import EventSubscription
from free_claude_code.core.json_types import JsonObject

from .models import (
    CodeDetail,
    CodeItem,
    CodePage,
    CodePrompt,
    CodeRun,
    CodeSession,
    HarnessEvent,
    NativeThread,
)

type EventSink = Callable[[HarnessEvent], Awaitable[None]]


class HarnessConnection(Protocol):
    generation: str
    thread_id: str | None

    def supports(self, selection: HarnessSelection) -> bool: ...

    async def create_thread(self) -> NativeThread: ...

    async def resume_thread(self, thread_id: str) -> NativeThread: ...

    async def read_thread(self, thread_id: str) -> NativeThread: ...

    async def start_turn(
        self, text: str, selection: HarnessSelection, client_id: str
    ) -> str: ...

    async def interrupt(self, turn_id: str) -> None: ...

    def prepare_answer(
        self, request_id: str | int, answer: JsonObject
    ) -> JsonObject: ...

    async def respond(self, request_id: str | int, response: JsonObject) -> None: ...

    async def delete_thread(self, thread_id: str) -> None: ...

    async def close(self) -> None: ...


class HarnessSelection(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def configuration_key(self) -> str: ...

    async def open(self, cwd: str, sink: EventSink) -> HarnessConnection: ...


class HarnessFactory(Protocol):
    def availability(self) -> tuple[bool, str | None]: ...

    def prepare(self) -> HarnessSelection: ...

    async def open_history(self, cwd: str, sink: EventSink) -> HarnessConnection: ...


class CodeStore(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def create(self, session: CodeSession) -> CodeSession: ...

    async def get_session(self, session_id: str) -> CodeSession: ...

    async def list_sessions(
        self, cursor: tuple[int, str] | None, limit: int
    ) -> CodePage: ...

    async def pending_deletions(self) -> tuple[CodeSession, ...]: ...

    async def is_deleted(self, session_id: str) -> bool: ...

    async def get_run(self, run_id: str) -> CodeRun | None: ...

    async def latest_run(self, session_id: str) -> CodeRun | None: ...

    async def items(
        self, session_id: str, before: int | None, limit: int | None
    ) -> tuple[CodeItem, ...]: ...

    async def prompts(self, session_id: str) -> tuple[CodePrompt, ...]: ...

    async def save(
        self,
        session: CodeSession,
        *,
        run: CodeRun | None = None,
        items: Sequence[CodeItem] = (),
        prompts: Sequence[CodePrompt] = (),
    ) -> None: ...

    async def delete(self, session_id: str) -> None: ...


class CodeApplicationPort(Protocol):
    @property
    def epoch(self) -> str: ...

    @property
    def cursor(self) -> int: ...

    def availability(self) -> tuple[bool, str | None]: ...

    async def create_session(self, session_id: str, cwd: str) -> CodeSession: ...

    async def list_sessions(
        self, cursor: tuple[int, str] | None = None, limit: int = 25
    ) -> CodePage: ...

    async def get_detail(
        self, session_id: str, *, before: int | None = None
    ) -> CodeDetail: ...

    async def subscribe(self) -> tuple[EventSubscription, JsonObject]: ...

    async def rename(
        self, session_id: str, revision: int, title: str
    ) -> CodeSession: ...

    async def send(
        self,
        session_id: str,
        operation_id: str,
        revision: int,
        text: str,
        *,
        expected_epoch: str,
    ) -> CodeRun: ...

    async def stop(self, session_id: str, operation_id: str) -> CodeRun: ...

    async def answer(
        self, session_id: str, prompt_id: str, response_id: str, answer: JsonObject
    ) -> CodePrompt: ...

    async def delete_session(
        self, session_id: str, revision: int
    ) -> CodeSession | None: ...

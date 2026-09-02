"""Narrow ports owned by the Codex-backed Work application."""

from collections.abc import AsyncIterator
from typing import Protocol

from free_claude_code.application.event_feed import PublishedEvent
from free_claude_code.core.json_types import JsonObject, JsonValue

from .models import (
    WorkBootstrap,
    WorkOperation,
    WorkOperationAcknowledgement,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionDetail,
    WorkSessionPage,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkTurnPage,
)


class WorkStorePort(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def list_sessions(self) -> tuple[WorkSessionRecord, ...]: ...

    async def get_session(self, thread_id: str) -> WorkSessionRecord: ...

    async def create_session(self, record: WorkSessionRecord) -> WorkSessionRecord: ...

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        settings: WorkSessionSettings,
    ) -> WorkSessionRecord: ...

    async def bump_revision(
        self, thread_id: str, *, expected_revision: int
    ) -> WorkSessionRecord: ...

    async def delete_session(self, thread_id: str) -> None: ...

    async def reserve_operation(
        self,
        *,
        operation_id: str,
        kind: WorkOperationKind,
        session_id: str | None,
        intent_digest: str,
    ) -> tuple[WorkOperation, bool]: ...

    async def update_operation(
        self,
        operation_id: str,
        *,
        state: WorkOperationState,
        result_thread_id: str | None = None,
        result_turn_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkOperation: ...

    async def prune_deleted_session_operations(
        self, thread_id: str, *, keep_operation_id: str
    ) -> None: ...

    async def recent_projects(self, *, limit: int) -> tuple[str, ...]: ...


class WorkEventSubscriptionPort(Protocol):
    cursor: int

    def __aiter__(self) -> AsyncIterator[PublishedEvent]: ...

    async def aclose(self) -> None: ...


class WorkApplicationPort(Protocol):
    async def bootstrap(self) -> WorkBootstrap: ...

    async def subscribe(self) -> WorkEventSubscriptionPort: ...

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> WorkSessionPage: ...

    async def create_session(
        self, *, cwd: str, operation_id: str
    ) -> WorkOperationAcknowledgement: ...

    async def get_detail(self, thread_id: str) -> WorkSessionDetail: ...

    async def get_turn_page(
        self, thread_id: str, *, cursor: str | None, limit: int
    ) -> WorkTurnPage: ...

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        updates: JsonObject,
    ) -> WorkSessionRecord: ...

    async def rename(
        self, thread_id: str, *, expected_revision: int, name: str
    ) -> WorkSessionRecord: ...

    async def send(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
    ) -> WorkOperationAcknowledgement: ...

    async def stop(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement: ...

    async def delete(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement: ...

    async def remove_missing(self, thread_id: str) -> None: ...

    async def respond(
        self,
        thread_id: str,
        interaction_id: str,
        *,
        value: JsonValue,
    ) -> None: ...

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
)


class WorkStorePort(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def list_sessions(self) -> tuple[WorkSessionRecord, ...]: ...

    async def get_session(self, thread_id: str) -> WorkSessionRecord: ...

    async def create_session_from_operation(
        self, operation_id: str, record: WorkSessionRecord
    ) -> tuple[WorkOperation, WorkSessionRecord]: ...

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        settings: WorkSessionSettings,
    ) -> WorkSessionRecord: ...

    async def delete_session(self, thread_id: str) -> None: ...

    async def complete_delete(
        self, operation_id: str, thread_id: str
    ) -> WorkOperation: ...

    async def admit_operation(
        self,
        *,
        operation_id: str,
        kind: WorkOperationKind,
        session_id: str | None,
        interaction_id: str | None,
        intent_digest: str,
        payload: JsonObject,
        expected_revision: int | None = None,
    ) -> tuple[WorkOperation, bool]: ...

    async def get_operation(self, operation_id: str) -> WorkOperation: ...

    async def list_operations(
        self, *, states: tuple[WorkOperationState, ...]
    ) -> tuple[WorkOperation, ...]: ...

    async def claim_operation(self, operation_id: str) -> WorkOperation | None: ...

    async def record_operation_evidence(
        self,
        operation_id: str,
        *,
        native_thread_id: str | None = None,
        native_turn_id: str | None = None,
        native_connection_id: str | None = None,
        captured_model: str | None = None,
        captured_reasoning_effort: str | None = None,
    ) -> WorkOperation: ...

    async def transition_operation(
        self,
        operation_id: str,
        *,
        expected_states: tuple[WorkOperationState, ...],
        state: WorkOperationState,
        native_thread_id: str | None = None,
        native_turn_id: str | None = None,
        native_connection_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkOperation: ...

    async def abandon_unknown_operations(
        self, session_id: str
    ) -> tuple[WorkOperation, ...]: ...

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

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        updates: JsonObject,
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
        operation_id: str,
        value: JsonValue,
    ) -> WorkOperationAcknowledgement: ...

    async def get_operation(
        self, operation_id: str
    ) -> WorkOperationAcknowledgement: ...

    async def acknowledge_unknown(
        self, thread_id: str
    ) -> tuple[WorkOperationAcknowledgement, ...]: ...

    async def dismiss_unknown_create(
        self, operation_id: str
    ) -> WorkOperationAcknowledgement: ...

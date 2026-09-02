"""Application owner for local Codex-backed Work Sessions."""

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections.abc import Coroutine
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path

import anyio.to_thread
from loguru import logger

from free_claude_code.application.event_feed import EventPublisher, EventSubscription
from free_claude_code.core.json_types import JsonObject, JsonValue

from .codex import (
    CodexAppServerEvent,
    CodexAppServerPort,
    CodexCompatibilityError,
    CodexConnectionError,
    CodexConnectionLost,
    CodexControlCatalog,
    CodexNotification,
    CodexRequestError,
    CodexServerRequest,
    CodexThreadSettings,
    CodexTurnSettings,
    CodexUnavailableError,
    CodexUnsupportedInteraction,
)
from .models import (
    WorkBootstrap,
    WorkCompatibilityError,
    WorkConflictError,
    WorkInteraction,
    WorkInteractionKind,
    WorkNotFoundError,
    WorkOperation,
    WorkOperationAcknowledgement,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionDetail,
    WorkSessionPage,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkSessionSummary,
    WorkStatus,
    WorkTimelineItem,
    WorkTurnPage,
    WorkUnavailableError,
    WorkValidationError,
)
from .ports import WorkStorePort

_SESSION_PAGE_LIMIT = 25
_TURN_PAGE_LIMIT = 50
_NATIVE_PAGE_LIMIT = 100
_RECENT_PROJECT_LIMIT = 8
_MAX_MESSAGE_LENGTH = 1_000_000
_MAX_NAME_LENGTH = 200

_COMMAND_APPROVAL = "item/commandExecution/requestApproval"
_FILE_APPROVAL = "item/fileChange/requestApproval"
_PERMISSION_APPROVAL = "item/permissions/requestApproval"
_USER_INPUT = "item/tool/requestUserInput"
_MCP_ELICITATION = "mcpServer/elicitation/request"
_LEGACY_INTERACTIONS = frozenset({"applyPatchApproval", "execCommandApproval"})

_DELTA_METHOD_FIELDS = {
    "item/agentMessage/delta": "text",
    "item/commandExecution/outputDelta": "aggregatedOutput",
    "item/fileChange/outputDelta": "output",
    "item/mcpToolCall/progress": "progress",
    "item/plan/delta": "text",
    "item/reasoning/summaryTextDelta": "summary",
    "item/reasoning/textDelta": "content",
}


@dataclass(slots=True)
class _ActiveTurn:
    thread_id: str
    operation_id: str
    turn_id: str | None = None
    stop_operation_id: str | None = None
    stop_requested: bool = False
    interrupt_sent: bool = False
    terminal: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True)
class _PendingInteraction:
    public: WorkInteraction
    connection_id: str
    request_id: int | str
    method: str
    params: JsonObject
    claimed: bool = False


class WorkService:
    """Own Work use cases, native projection, and browser event fan-out."""

    def __init__(self, codex: CodexAppServerPort, store: WorkStorePort) -> None:
        self._codex = codex
        self._store = store
        self._events = EventPublisher()
        self._generation = str(uuid.uuid4())
        self._state_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._records: dict[str, WorkSessionRecord] = {}
        self._native_threads: dict[str, JsonObject] = {}
        self._missing_threads: set[str] = set()
        self._native_status: dict[str, WorkStatus] = {}
        self._live_items: dict[tuple[str, str, str], WorkTimelineItem] = {}
        self._active_turns: dict[str, _ActiveTurn] = {}
        self._deleting: set[str] = set()
        self._interactions: dict[str, _PendingInteraction] = {}
        self._interaction_keys: dict[tuple[str, int | str], str] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._event_task: asyncio.Task[None] | None = None
        self._started = False
        self._accepting = False
        self._unavailable_message: str | None = "Work Sessions is starting."

    async def start(self) -> None:
        if self._started:
            return
        try:
            await self._store.start()
            records = await self._store.list_sessions()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._unavailable_message = (
                str(exc)
                if isinstance(exc, WorkUnavailableError)
                else "Work storage could not be opened."
            )
            logger.warning("Work Sessions unavailable: exc_type={}", type(exc).__name__)
            return
        self._records = {record.thread_id: record for record in records}
        self._started = True
        self._accepting = True
        self._unavailable_message = None
        self._event_task = asyncio.create_task(
            self._pump_events(),
            name="fcc-work-codex-events",
        )

    async def close(self) -> None:
        self._accepting = False
        async with self._state_lock:
            active = tuple(self._active_turns.values())
            interactions = tuple(self._interactions.values())
            tasks = tuple(self._tasks)
        await asyncio.gather(
            *(
                self._settle_operation(
                    item.operation_id,
                    WorkOperationState.INTERRUPTED,
                    result_thread_id=item.thread_id,
                    result_turn_id=item.turn_id,
                    error_code="server_shutdown",
                    error_message="FCC stopped before Codex confirmed completion.",
                )
                for item in active
            ),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(self._decline_on_shutdown(item) for item in interactions),
            return_exceptions=True,
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        event_task = self._event_task
        if event_task is not None:
            event_task.cancel()
        await self._codex.close()
        if event_task is not None:
            await asyncio.gather(event_task, return_exceptions=True)
        self._events.close()
        if self._started:
            await self._store.close()
        self._started = False
        self._unavailable_message = "Work Sessions is stopped."

    async def bootstrap(self) -> WorkBootstrap:
        self._require_store()
        availability = await self._codex.availability()
        reason = availability.reason
        if not availability.available and not reason:
            reason = "Install or update Codex to use Work Sessions."
        return WorkBootstrap(
            available=availability.available,
            reason=reason,
            codex_version=availability.version,
            recent_projects=await self._store.recent_projects(
                limit=_RECENT_PROJECT_LIMIT
            ),
            event_generation=self._generation,
            event_cursor=self._events.cursor,
        )

    async def subscribe(self) -> EventSubscription:
        self._require_store()
        return self._events.subscribe()

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> WorkSessionPage:
        self._require_store()
        await self._synchronize_native_index()
        summaries = await self._summaries()
        normalized_query = query.strip().casefold()
        if normalized_query:
            summaries = tuple(
                summary
                for summary in summaries
                if normalized_query
                in " ".join((summary.title, summary.preview, summary.cwd)).casefold()
            )
        if cursor is not None:
            summaries = tuple(
                summary
                for summary in summaries
                if (_summary_time(summary), summary.thread_id) < cursor
            )
        page_limit = max(1, min(_SESSION_PAGE_LIMIT, limit))
        selected = summaries[: page_limit + 1]
        has_more = len(selected) > page_limit
        selected = selected[:page_limit]
        next_cursor = None
        if has_more and selected:
            last = selected[-1]
            next_cursor = (_summary_time(last), last.thread_id)
        return WorkSessionPage(sessions=selected, next_cursor=next_cursor)

    async def create_session(
        self, *, cwd: str, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        canonical_path, path_key = await _canonical_project_path(cwd)
        digest = _intent_digest("create", canonical_path)
        operation, created = await self._store.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            intent_digest=digest,
        )
        if created:
            self._spawn(
                self._run_create(operation_id, canonical_path, path_key),
                name=f"fcc-work-create-{operation_id}",
            )
        return _acknowledgement(operation)

    async def get_detail(self, thread_id: str) -> WorkSessionDetail:
        self._require_store()
        record = self._record(thread_id)
        project_available = await _project_is_available(record.cwd)
        try:
            snapshot = await self._codex.read_thread(thread_id)
        except CodexUnavailableError, CodexConnectionError:
            async with self._state_lock:
                self._native_status[thread_id] = WorkStatus.DISCONNECTED
            return await self._placeholder_detail(record)
        except CodexRequestError as exc:
            if _is_native_not_found(exc):
                async with self._state_lock:
                    self._missing_threads.add(thread_id)
                return await self._placeholder_detail(record)
            raise _work_error(exc) from exc
        except Exception as exc:
            raise _work_error(exc) from exc
        try:
            if project_available:
                catalog, page = await asyncio.gather(
                    self._codex.controls(cwd=record.cwd),
                    self._codex.list_turns_page(
                        thread_id=thread_id,
                        cursor=None,
                        limit=_TURN_PAGE_LIMIT,
                    ),
                )
            else:
                catalog = CodexControlCatalog(
                    models=None,
                    collaboration_modes=None,
                    permission_profiles=None,
                    config=None,
                )
                page = await self._codex.list_turns_page(
                    thread_id=thread_id,
                    cursor=None,
                    limit=_TURN_PAGE_LIMIT,
                )
        except CodexUnavailableError, CodexConnectionError:
            async with self._state_lock:
                self._native_status[thread_id] = WorkStatus.DISCONNECTED
            return await self._placeholder_detail(record)
        except Exception as exc:
            raise _work_error(exc) from exc
        async with self._state_lock:
            self._native_threads[thread_id] = snapshot.thread
            self._missing_threads.discard(thread_id)
            if self._native_status.get(thread_id) is WorkStatus.DISCONNECTED:
                self._native_status[thread_id] = WorkStatus.READY
            live_items = tuple(
                item for key, item in self._live_items.items() if key[0] == thread_id
            )
            interactions = tuple(
                pending.public
                for pending in self._interactions.values()
                if pending.public.thread_id == thread_id and not pending.claimed
            )
            event_cursor = self._events.cursor
        turns = _turn_page(thread_id, page.records, page.next_cursor)
        persisted_keys = {
            (item.thread_id, item.turn_id, item.item_id) for item in turns.items
        }
        summary = await self._summary(record)
        return WorkSessionDetail(
            summary=summary,
            settings=record.settings,
            controls=_controls_payload(catalog),
            turns=turns,
            live_items=tuple(
                item
                for item in sorted(
                    live_items,
                    key=lambda candidate: (candidate.turn_id, candidate.item_id),
                )
                if (item.thread_id, item.turn_id, item.item_id) not in persisted_keys
            ),
            interactions=interactions,
            event_cursor=event_cursor,
        )

    async def _placeholder_detail(self, record: WorkSessionRecord) -> WorkSessionDetail:
        return WorkSessionDetail(
            summary=await self._summary(record),
            settings=record.settings,
            controls={},
            turns=WorkTurnPage(items=(), next_cursor=None),
            live_items=(),
            interactions=(),
            event_cursor=self._events.cursor,
        )

    async def get_turn_page(
        self, thread_id: str, *, cursor: str | None, limit: int
    ) -> WorkTurnPage:
        self._require_store()
        self._record(thread_id)
        try:
            page = await self._codex.list_turns_page(
                thread_id=thread_id,
                cursor=cursor,
                limit=max(1, min(_TURN_PAGE_LIMIT, limit)),
            )
        except Exception as exc:
            raise _work_error(exc) from exc
        return _turn_page(thread_id, page.records, page.next_cursor)

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        updates: JsonObject,
    ) -> WorkSessionRecord:
        self._require_accepting()
        current = self._record(thread_id)
        self._require_mutable(thread_id)
        if current.revision != expected_revision:
            raise WorkConflictError("Work session changed in another tab. Reload it.")
        try:
            catalog = await self._codex.controls(cwd=current.cwd)
        except Exception as exc:
            raise _work_error(exc) from exc
        settings = _updated_settings(current.settings, updates, catalog)
        updated = await self._store.update_settings(
            thread_id,
            expected_revision=expected_revision,
            settings=settings,
        )
        async with self._state_lock:
            self._records[thread_id] = updated
        self._publish_session("session.updated", thread_id)
        return updated

    async def rename(
        self, thread_id: str, *, expected_revision: int, name: str
    ) -> WorkSessionRecord:
        self._require_accepting()
        current = self._record(thread_id)
        self._require_mutable(thread_id)
        normalized = " ".join(name.split())
        if not normalized:
            raise WorkValidationError("Work session name cannot be empty.")
        if len(normalized) > _MAX_NAME_LENGTH:
            raise WorkValidationError("Work session name cannot exceed 200 characters.")
        if current.revision != expected_revision:
            async with self._state_lock:
                native_name = _optional_string(
                    self._native_threads.get(thread_id, {}).get("name")
                )
            if native_name == normalized:
                return current
            raise WorkConflictError("Work session changed in another tab. Reload it.")
        try:
            await self._codex.set_thread_name(thread_id=thread_id, name=normalized)
        except Exception as exc:
            raise _work_error(exc) from exc
        updated = await self._store.bump_revision(
            thread_id,
            expected_revision=expected_revision,
        )
        async with self._state_lock:
            self._records[thread_id] = updated
            native = self._native_threads.get(thread_id)
            if native is not None:
                native["name"] = normalized
        self._publish_session("session.updated", thread_id)
        return updated

    async def send(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        record = self._record(thread_id)
        await self._require_project(record.cwd)
        if record.revision != expected_revision:
            raise WorkConflictError("Work session changed in another tab. Reload it.")
        if not text.strip():
            raise WorkValidationError("Write a message before sending.")
        if len(text) > _MAX_MESSAGE_LENGTH:
            raise WorkValidationError(
                "Work message cannot exceed 1,000,000 characters."
            )
        digest = _intent_digest("send", thread_id, str(expected_revision), text)
        operation, created = await self._store.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.SEND,
            session_id=thread_id,
            intent_digest=digest,
        )
        if not created:
            return _acknowledgement(operation)
        async with self._state_lock:
            if thread_id in self._deleting:
                conflict = "This Work session is being deleted."
            elif thread_id in self._active_turns:
                conflict = "This Work session already has an active turn."
            else:
                conflict = None
                self._active_turns[thread_id] = _ActiveTurn(
                    thread_id=thread_id,
                    operation_id=operation_id,
                )
        if conflict is not None:
            await self._settle_operation(
                operation_id,
                WorkOperationState.FAILED,
                result_thread_id=thread_id,
                error_code="conflict",
                error_message=conflict,
            )
            raise WorkConflictError(conflict)
        self._publish_status(thread_id)
        self._spawn(
            self._run_send(record, operation_id, text),
            name=f"fcc-work-send-{operation_id}",
        )
        return _acknowledgement(operation)

    async def stop(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        self._record(thread_id)
        operation, created = await self._store.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.STOP,
            session_id=thread_id,
            intent_digest=_intent_digest("stop", thread_id),
        )
        if not created:
            return _acknowledgement(operation)
        async with self._state_lock:
            active = self._active_turns.get(thread_id)
            if active is not None:
                active.stop_requested = True
                active.stop_operation_id = operation_id
        if active is None:
            operation = await self._store.update_operation(
                operation_id,
                state=WorkOperationState.COMPLETED,
                result_thread_id=thread_id,
            )
            return _acknowledgement(operation)
        await self._store.update_operation(
            operation_id,
            state=WorkOperationState.SUBMITTED,
            result_thread_id=thread_id,
            result_turn_id=active.turn_id,
        )
        self._publish_status(thread_id)
        if active.turn_id is not None:
            self._spawn(
                self._interrupt(active),
                name=f"fcc-work-stop-{operation_id}",
            )
        return _acknowledgement(operation)

    async def delete(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        operation, created = await self._store.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.DELETE,
            session_id=thread_id,
            intent_digest=_intent_digest("delete", thread_id),
        )
        if not created:
            return _acknowledgement(operation)
        try:
            self._record(thread_id)
        except WorkNotFoundError:
            await self._settle_operation(
                operation_id,
                WorkOperationState.FAILED,
                result_thread_id=thread_id,
                error_code="not_found",
                error_message="Work session was not found.",
            )
            raise
        async with self._state_lock:
            if thread_id in self._deleting:
                conflict = True
                active = None
            else:
                conflict = False
                self._deleting.add(thread_id)
                active = self._active_turns.get(thread_id)
                if active is not None:
                    active.stop_requested = True
        if conflict:
            await self._settle_operation(
                operation_id,
                WorkOperationState.FAILED,
                result_thread_id=thread_id,
                error_code="conflict",
                error_message="This Work session is already being deleted.",
            )
            raise WorkConflictError("This Work session is already being deleted.")
        self._publish_status(thread_id)
        self._spawn(
            self._run_delete(thread_id, operation_id, active),
            name=f"fcc-work-delete-{operation_id}",
        )
        return _acknowledgement(operation)

    async def remove_missing(self, thread_id: str) -> None:
        self._require_accepting()
        self._record(thread_id)
        async with self._state_lock:
            if thread_id not in self._missing_threads:
                raise WorkConflictError(
                    "Only a confirmed missing Codex session can be removed from Work."
                )
        await self._store.delete_session(thread_id)
        async with self._state_lock:
            self._records.pop(thread_id, None)
            self._native_threads.pop(thread_id, None)
            self._missing_threads.discard(thread_id)
            self._native_status.pop(thread_id, None)
            self._clear_thread_projection(thread_id)
        self._events.publish("session.deleted", {"thread_id": thread_id})

    async def respond(
        self,
        thread_id: str,
        interaction_id: str,
        *,
        value: JsonValue,
    ) -> None:
        self._require_accepting()
        self._record(thread_id)
        async with self._state_lock:
            pending = self._interactions.get(interaction_id)
            if (
                pending is None
                or pending.public.thread_id != thread_id
                or pending.claimed
            ):
                raise WorkConflictError("This Codex request was already answered.")
            result = _interaction_response(pending, value)
            pending.claimed = True
        try:
            await self._codex.respond(
                connection_id=pending.connection_id,
                request_id=pending.request_id,
                result=result,
            )
        except Exception as exc:
            async with self._state_lock:
                self._retire_interaction(interaction_id)
            self._publish_status(thread_id)
            raise _work_error(exc) from exc
        self._events.publish(
            "interaction.resolved",
            {"thread_id": thread_id, "interaction_id": interaction_id},
        )

    async def _run_create(self, operation_id: str, cwd: str, cwd_key: str) -> None:
        native_thread_id: str | None = None
        try:
            await self._store.update_operation(
                operation_id,
                state=WorkOperationState.SUBMITTED,
            )
            handle = await self._codex.start_thread(CodexThreadSettings(cwd=cwd))
            native_thread_id = handle.thread_id
            settings = _settings_from_start(handle.response)
            record = WorkSessionRecord(
                thread_id=native_thread_id,
                cwd=cwd,
                cwd_key=cwd_key,
                settings=settings,
                revision=1,
                registered_at_ms=_now_ms(),
            )
            await self._store.create_session(record)
            async with self._state_lock:
                self._records[native_thread_id] = record
                thread = handle.response.get("thread")
                if isinstance(thread, dict):
                    self._native_threads[native_thread_id] = dict(thread)
                self._missing_threads.discard(native_thread_id)
            await self._settle_operation(
                operation_id,
                WorkOperationState.COMPLETED,
                result_thread_id=native_thread_id,
            )
            self._publish_session("session.created", native_thread_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if native_thread_id is not None:
                try:
                    await self._codex.delete_thread(native_thread_id)
                except Exception as cleanup_exc:
                    logger.warning(
                        "Could not compensate unregistered Codex thread: exc_type={}",
                        type(cleanup_exc).__name__,
                    )
            await self._settle_operation(
                operation_id,
                WorkOperationState.FAILED,
                result_thread_id=native_thread_id,
                error_code=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )

    async def _run_send(
        self,
        record: WorkSessionRecord,
        operation_id: str,
        text: str,
    ) -> None:
        active = self._active_turns[record.thread_id]
        try:
            await self._codex.resume_thread(
                record.thread_id,
                CodexThreadSettings(
                    cwd=record.cwd,
                    model=record.settings.model,
                    permission_profile=record.settings.permission_profile,
                ),
            )
            catalog = await self._codex.controls(cwd=record.cwd)
            _validate_settings(record.settings, catalog)
            collaboration = _collaboration_value(
                record.settings.collaboration_mode,
                catalog,
            )
            await self._store.update_operation(
                operation_id,
                state=WorkOperationState.SUBMITTED,
                result_thread_id=record.thread_id,
            )
            handle = await self._codex.start_turn(
                thread_id=record.thread_id,
                text=text,
                settings=CodexTurnSettings(
                    model=record.settings.model,
                    effort=record.settings.reasoning_effort,
                    collaboration_mode=collaboration,
                    permission_profile=record.settings.permission_profile,
                ),
                client_user_message_id=operation_id,
            )
            async with self._state_lock:
                current = self._active_turns.get(record.thread_id)
                if current is active:
                    current.turn_id = handle.turn_id
                    should_interrupt = current.stop_requested
                else:
                    should_interrupt = False
            operation = await self._store.update_operation(
                operation_id,
                state=WorkOperationState.SUBMITTED,
                result_thread_id=record.thread_id,
                result_turn_id=handle.turn_id,
            )
            if operation.state is WorkOperationState.SUBMITTED:
                self._events.publish(
                    "operation.updated",
                    {
                        "thread_id": record.thread_id,
                        "operation_id": operation_id,
                        "state": WorkOperationState.SUBMITTED.value,
                        "turn_id": handle.turn_id,
                    },
                )
            if should_interrupt:
                await self._interrupt(active)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_active_turn(active, exc)

    async def _interrupt(self, active: _ActiveTurn) -> None:
        async with self._state_lock:
            if (
                active.interrupt_sent
                or active.turn_id is None
                or active.terminal.is_set()
            ):
                return
            active.interrupt_sent = True
            turn_id = active.turn_id
        try:
            await self._codex.interrupt_turn(
                thread_id=active.thread_id,
                turn_id=turn_id,
            )
        except Exception as exc:
            stop_id = active.stop_operation_id
            if stop_id is not None:
                await self._settle_operation(
                    stop_id,
                    WorkOperationState.FAILED,
                    result_thread_id=active.thread_id,
                    result_turn_id=turn_id,
                    error_code=type(exc).__name__,
                    error_message=_safe_error_message(exc),
                )
            async with self._state_lock:
                active.stop_requested = False
                active.stop_operation_id = None
                active.interrupt_sent = False
            self._publish_status(active.thread_id)

    async def _run_delete(
        self,
        thread_id: str,
        operation_id: str,
        active: _ActiveTurn | None,
    ) -> None:
        try:
            await self._store.update_operation(
                operation_id,
                state=WorkOperationState.SUBMITTED,
                result_thread_id=thread_id,
                result_turn_id=active.turn_id if active is not None else None,
            )
            if active is not None:
                if active.turn_id is not None:
                    await self._interrupt(active)
                await active.terminal.wait()
            try:
                await self._codex.delete_thread(thread_id)
            except CodexRequestError as exc:
                if not _is_native_not_found(exc):
                    raise
            await self._store.delete_session(thread_id)
            await self._store.prune_deleted_session_operations(
                thread_id,
                keep_operation_id=operation_id,
            )
            await self._settle_operation(
                operation_id,
                WorkOperationState.COMPLETED,
                result_thread_id=thread_id,
            )
            async with self._state_lock:
                self._records.pop(thread_id, None)
                self._native_threads.pop(thread_id, None)
                self._missing_threads.discard(thread_id)
                self._native_status.pop(thread_id, None)
                self._deleting.discard(thread_id)
                self._clear_thread_projection(thread_id)
            self._events.publish("session.deleted", {"thread_id": thread_id})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._state_lock:
                self._deleting.discard(thread_id)
            await self._settle_operation(
                operation_id,
                WorkOperationState.FAILED,
                result_thread_id=thread_id,
                error_code=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )
            self._publish_status(thread_id)

    async def _pump_events(self) -> None:
        try:
            async for event in self._codex.events():
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Work Codex event pump failed: exc_type={}", type(exc).__name__
            )
            await self._handle_connection_lost(
                CodexConnectionLost(
                    connection_id="unknown",
                    message="Codex event stream stopped.",
                )
            )

    async def _handle_event(self, event: CodexAppServerEvent) -> None:
        if isinstance(event, CodexConnectionLost):
            await self._handle_connection_lost(event)
        elif isinstance(event, CodexServerRequest):
            await self._handle_server_request(event)
        elif isinstance(event, CodexUnsupportedInteraction):
            self._events.publish(
                "work.warning",
                {
                    "message": f"Codex requested an unsupported interaction: {event.method}",
                },
            )
        elif isinstance(event, CodexNotification):
            await self._handle_notification(event)

    async def _handle_connection_lost(self, event: CodexConnectionLost) -> None:
        async with self._state_lock:
            active = tuple(self._active_turns.values())
            affected_threads = set(self._records)
            for thread_id in affected_threads:
                self._native_status[thread_id] = WorkStatus.DISCONNECTED
            self._interactions.clear()
            self._interaction_keys.clear()
            self._active_turns.clear()
            for item in active:
                item.terminal.set()
        for item in active:
            await self._settle_operation(
                item.operation_id,
                WorkOperationState.INTERRUPTED,
                result_thread_id=item.thread_id,
                result_turn_id=item.turn_id,
                error_code="connection_lost",
                error_message=event.message,
            )
            if item.stop_operation_id is not None:
                await self._settle_operation(
                    item.stop_operation_id,
                    WorkOperationState.INTERRUPTED,
                    result_thread_id=item.thread_id,
                    result_turn_id=item.turn_id,
                    error_code="connection_lost",
                    error_message=event.message,
                )
        self._events.publish(
            "work.disconnected",
            {"message": event.message},
        )

    async def _handle_server_request(self, event: CodexServerRequest) -> None:
        params = _as_object(event.params)
        if event.method == _MCP_ELICITATION:
            await self._codex.respond(
                connection_id=event.connection_id,
                request_id=event.request_id,
                result={"action": "cancel"},
            )
            self._events.publish(
                "work.warning",
                {
                    "message": "Codex requested an unsupported MCP form; it was cancelled."
                },
            )
            return
        if event.method in _LEGACY_INTERACTIONS:
            await self._codex.respond(
                connection_id=event.connection_id,
                request_id=event.request_id,
                result={"decision": "abort"},
            )
            self._events.publish(
                "work.warning",
                {
                    "message": "Codex requested a legacy approval; it was safely declined."
                },
            )
            return
        kind = _interaction_kind(event.method)
        if kind is None:
            return
        thread_id = _optional_string(params.get("threadId"))
        if thread_id is None or thread_id not in self._records:
            await self._codex.respond(
                connection_id=event.connection_id,
                request_id=event.request_id,
                result=_safe_decline_result(event.method, params),
            )
            return
        key = (event.connection_id, event.request_id)
        async with self._state_lock:
            existing_id = self._interaction_keys.get(key)
            if existing_id is not None:
                return
            interaction_id = str(uuid.uuid4())
            public = WorkInteraction(
                interaction_id=interaction_id,
                thread_id=thread_id,
                turn_id=_optional_string(params.get("turnId")),
                kind=kind,
                title=_interaction_title(kind),
                payload=_public_interaction_payload(kind, params),
            )
            self._interactions[interaction_id] = _PendingInteraction(
                public=public,
                connection_id=event.connection_id,
                request_id=event.request_id,
                method=event.method,
                params=params,
            )
            self._interaction_keys[key] = interaction_id
        self._events.publish("interaction.created", _interaction_payload(public))
        self._publish_status(thread_id)

    async def _handle_notification(self, event: CodexNotification) -> None:
        params = _as_object(event.params)
        method = event.method
        if method == "serverRequest/resolved":
            request_id = params.get("requestId")
            if not isinstance(request_id, (int, str)) or isinstance(request_id, bool):
                return
            key = (event.connection_id, request_id)
            async with self._state_lock:
                interaction_id = self._interaction_keys.get(key)
                pending = (
                    self._interactions.get(interaction_id)
                    if interaction_id is not None
                    else None
                )
                if interaction_id is not None:
                    self._retire_interaction(interaction_id)
            if pending is not None:
                self._events.publish(
                    "interaction.resolved",
                    {
                        "thread_id": pending.public.thread_id,
                        "interaction_id": pending.public.interaction_id,
                    },
                )
                self._publish_status(pending.public.thread_id)
            return
        thread_id = _event_thread_id(method, params)
        if thread_id is None or thread_id not in self._records:
            return
        if method == "thread/name/updated":
            async with self._state_lock:
                native = self._native_threads.get(thread_id)
                if native is not None:
                    native["name"] = params.get("threadName")
            self._publish_session("session.updated", thread_id)
            return
        if method == "thread/status/changed":
            status = _native_status(params.get("status"))
            async with self._state_lock:
                self._native_status[thread_id] = status
            self._publish_status(thread_id)
            return
        if method == "thread/deleted":
            async with self._state_lock:
                self._missing_threads.add(thread_id)
            self._publish_session("session.updated", thread_id)
            return
        if method == "turn/started":
            turn = _as_object(params.get("turn"))
            turn_id = _required_string(turn, "id")
            async with self._state_lock:
                active = self._active_turns.get(thread_id)
                if active is not None and active.turn_id is None:
                    active.turn_id = turn_id
                    should_interrupt = active.stop_requested
                else:
                    should_interrupt = False
                self._native_status[thread_id] = WorkStatus.WORKING
            self._publish_status(thread_id)
            if active is not None and should_interrupt:
                self._spawn(
                    self._interrupt(active),
                    name=f"fcc-work-stop-{active.operation_id}",
                )
            return
        if method == "turn/completed":
            await self._complete_turn(thread_id, _as_object(params.get("turn")))
            return
        if method in {"item/started", "item/completed"}:
            turn_id = _optional_string(params.get("turnId"))
            item = _as_object(params.get("item"))
            if turn_id is None or not item:
                return
            projected = _timeline_item(thread_id, turn_id, item)
            async with self._state_lock:
                self._live_items[(thread_id, turn_id, projected.item_id)] = projected
            self._events.publish("timeline.item", _timeline_payload(projected))
            return
        if method in _DELTA_METHOD_FIELDS:
            await self._apply_delta(thread_id, method, params)
            return
        if method == "item/fileChange/patchUpdated":
            await self._replace_item_field(thread_id, params, "changes")
            return
        if method == "turn/diff/updated":
            await self._upsert_synthetic_item(
                thread_id,
                params,
                item_id="turn-diff",
                kind="diff",
                text=_optional_string(params.get("diff")),
            )
            return
        if method == "turn/plan/updated":
            await self._upsert_synthetic_item(
                thread_id,
                params,
                item_id="turn-plan",
                kind="plan",
                text=_json_text(params.get("plan")),
            )
            return
        await self._record_unknown_notification(thread_id, method, params)

    async def _apply_delta(
        self, thread_id: str, method: str, params: JsonObject
    ) -> None:
        turn_id = _optional_string(params.get("turnId"))
        item_id = _optional_string(params.get("itemId"))
        delta = _optional_string(params.get("delta"))
        if turn_id is None or item_id is None or delta is None:
            return
        key = (thread_id, turn_id, item_id)
        async with self._state_lock:
            current = self._live_items.get(key)
            if current is None:
                current = WorkTimelineItem(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                    kind=_kind_from_delta(method),
                    status="inProgress",
                    text="",
                    payload={"id": item_id, "type": _kind_from_delta(method)},
                )
            field_name = _DELTA_METHOD_FIELDS[method]
            payload = dict(current.payload)
            previous_field = _optional_string(payload.get(field_name)) or ""
            payload[field_name] = previous_field + delta
            text = (current.text or "") + delta
            updated = replace(current, text=text, payload=payload)
            self._live_items[key] = updated
        self._events.publish("timeline.item", _timeline_payload(updated))

    async def _replace_item_field(
        self, thread_id: str, params: JsonObject, field_name: str
    ) -> None:
        turn_id = _optional_string(params.get("turnId"))
        item_id = _optional_string(params.get("itemId"))
        if turn_id is None or item_id is None:
            return
        key = (thread_id, turn_id, item_id)
        async with self._state_lock:
            current = self._live_items.get(key)
            if current is None:
                current = WorkTimelineItem(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                    kind="fileChange",
                    status="inProgress",
                    text=None,
                    payload={"id": item_id, "type": "fileChange"},
                )
            payload = dict(current.payload)
            payload[field_name] = params.get(field_name)
            updated = replace(current, payload=payload)
            self._live_items[key] = updated
        self._events.publish("timeline.item", _timeline_payload(updated))

    async def _upsert_synthetic_item(
        self,
        thread_id: str,
        params: JsonObject,
        *,
        item_id: str,
        kind: str,
        text: str | None,
    ) -> None:
        turn_id = _optional_string(params.get("turnId"))
        if turn_id is None:
            return
        item = WorkTimelineItem(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=item_id,
            kind=kind,
            status="inProgress",
            text=text,
            payload=dict(params),
        )
        async with self._state_lock:
            self._live_items[(thread_id, turn_id, item_id)] = item
        self._events.publish("timeline.item", _timeline_payload(item))

    async def _record_unknown_notification(
        self, thread_id: str, method: str, params: JsonObject
    ) -> None:
        turn_id = _optional_string(params.get("turnId"))
        if turn_id is None:
            return
        item_id = (
            _optional_string(params.get("itemId"))
            or _intent_digest(
                method,
                json.dumps(params, sort_keys=True, separators=(",", ":"), default=str),
            )[:24]
        )
        item = WorkTimelineItem(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=f"event-{item_id}",
            kind="codexActivity",
            status=None,
            text=method,
            payload={"method": method, "params": params},
        )
        async with self._state_lock:
            self._live_items[(thread_id, turn_id, item.item_id)] = item
        self._events.publish("timeline.item", _timeline_payload(item))

    async def _complete_turn(self, thread_id: str, turn: JsonObject) -> None:
        turn_id = _optional_string(turn.get("id"))
        status = _turn_status(turn)
        async with self._state_lock:
            active = self._active_turns.pop(thread_id, None)
            self._native_status[thread_id] = status
            if active is not None:
                active.terminal.set()
        if active is not None:
            operation_state = (
                WorkOperationState.COMPLETED
                if status in {WorkStatus.COMPLETED, WorkStatus.INTERRUPTED}
                else WorkOperationState.FAILED
            )
            await self._settle_operation(
                active.operation_id,
                operation_state,
                result_thread_id=thread_id,
                result_turn_id=turn_id or active.turn_id,
                error_code=(
                    "turn_failed"
                    if operation_state is WorkOperationState.FAILED
                    else None
                ),
                error_message=(
                    _turn_error(turn)
                    if operation_state is WorkOperationState.FAILED
                    else None
                ),
            )
            if active.stop_operation_id is not None:
                await self._settle_operation(
                    active.stop_operation_id,
                    WorkOperationState.COMPLETED,
                    result_thread_id=thread_id,
                    result_turn_id=turn_id or active.turn_id,
                )
        self._publish_status(thread_id)
        self._publish_session("session.updated", thread_id)

    async def _fail_active_turn(self, active: _ActiveTurn, exc: Exception) -> None:
        async with self._state_lock:
            current = self._active_turns.get(active.thread_id)
            if current is active:
                self._active_turns.pop(active.thread_id, None)
            active.terminal.set()
            self._native_status[active.thread_id] = WorkStatus.FAILED
        await self._settle_operation(
            active.operation_id,
            WorkOperationState.FAILED,
            result_thread_id=active.thread_id,
            result_turn_id=active.turn_id,
            error_code=type(exc).__name__,
            error_message=_safe_error_message(exc),
        )
        if active.stop_operation_id is not None:
            await self._settle_operation(
                active.stop_operation_id,
                WorkOperationState.COMPLETED,
                result_thread_id=active.thread_id,
                result_turn_id=active.turn_id,
            )
        self._publish_status(active.thread_id)

    async def _settle_operation(
        self,
        operation_id: str,
        state: WorkOperationState,
        *,
        result_thread_id: str | None = None,
        result_turn_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkOperation:
        operation = await self._store.update_operation(
            operation_id,
            state=state,
            result_thread_id=result_thread_id,
            result_turn_id=result_turn_id,
            error_code=error_code,
            error_message=error_message,
        )
        self._events.publish(
            "operation.updated",
            {
                "operation_id": operation.operation_id,
                "kind": operation.kind.value,
                "state": operation.state.value,
                "thread_id": operation.result_thread_id or operation.session_id,
                "turn_id": operation.result_turn_id,
                "error_message": operation.error_message,
            },
        )
        return operation

    async def _synchronize_native_index(self) -> None:
        async with self._sync_lock:
            async with self._state_lock:
                scanned_ids = set(self._records)
            native: dict[str, JsonObject] = {}
            cursor: str | None = None
            try:
                while True:
                    page = await self._codex.list_threads_page(
                        cursor=cursor,
                        limit=_NATIVE_PAGE_LIMIT,
                    )
                    for thread in page.records:
                        thread_id = _optional_string(thread.get("id"))
                        if thread_id is not None and thread_id in scanned_ids:
                            native[thread_id] = dict(thread)
                    cursor = page.next_cursor
                    if cursor is None:
                        break
            except CodexUnavailableError, CodexConnectionError:
                async with self._state_lock:
                    for thread_id in scanned_ids.intersection(self._records):
                        self._native_status[thread_id] = WorkStatus.DISCONNECTED
                return
            except Exception as exc:
                raise _work_error(exc) from exc
            async with self._state_lock:
                current_ids = set(self._records)
                synchronized_ids = scanned_ids.intersection(current_ids)
                for thread_id in synchronized_ids:
                    thread = native.get(thread_id)
                    if thread is None:
                        self._native_threads.pop(thread_id, None)
                        self._missing_threads.add(thread_id)
                        continue
                    self._native_threads[thread_id] = thread
                    self._missing_threads.discard(thread_id)
                    if self._native_status.get(thread_id) is WorkStatus.DISCONNECTED:
                        self._native_status[thread_id] = WorkStatus.READY

    async def _summaries(self) -> tuple[WorkSessionSummary, ...]:
        summaries = await asyncio.gather(
            *(self._summary(record) for record in self._records.values())
        )
        return tuple(
            sorted(
                summaries,
                key=lambda summary: (_summary_time(summary), summary.thread_id),
                reverse=True,
            )
        )

    async def _summary(self, record: WorkSessionRecord) -> WorkSessionSummary:
        project_available = await _project_is_available(record.cwd)
        async with self._state_lock:
            native = self._native_threads.get(record.thread_id, {})
            session_available = record.thread_id not in self._missing_threads
            status = self._status_for(record.thread_id)
        name = _optional_string(native.get("name"))
        preview = _optional_string(native.get("preview")) or ""
        updated_at = _native_timestamp_ms(native.get("recencyAt"))
        if updated_at is None:
            updated_at = _native_timestamp_ms(native.get("updatedAt"))
        title = name or _title_from_preview(preview) or "New Work Session"
        return WorkSessionSummary(
            thread_id=record.thread_id,
            cwd=record.cwd,
            title=title,
            preview=preview,
            status=status,
            revision=record.revision,
            registered_at_ms=record.registered_at_ms,
            updated_at_ms=updated_at,
            project_available=project_available,
            session_available=session_available,
        )

    def _status_for(self, thread_id: str) -> WorkStatus:
        if thread_id in self._deleting:
            return WorkStatus.DELETING
        active = self._active_turns.get(thread_id)
        if active is not None and active.stop_requested:
            return WorkStatus.STOPPING
        kinds = {
            pending.public.kind
            for pending in self._interactions.values()
            if pending.public.thread_id == thread_id and not pending.claimed
        }
        if WorkInteractionKind.USER_INPUT in kinds:
            return WorkStatus.WAITING_FOR_INPUT
        if kinds:
            return WorkStatus.WAITING_FOR_APPROVAL
        if active is not None:
            return WorkStatus.WORKING
        if thread_id in self._missing_threads:
            return WorkStatus.DISCONNECTED
        return self._native_status.get(thread_id, WorkStatus.READY)

    def _record(self, thread_id: str) -> WorkSessionRecord:
        record = self._records.get(thread_id)
        if record is None:
            raise WorkNotFoundError("Work session was not found.")
        return record

    def _require_store(self) -> None:
        if not self._started:
            raise WorkUnavailableError(
                self._unavailable_message or "Work Sessions is unavailable."
            )

    def _require_accepting(self) -> None:
        self._require_store()
        if not self._accepting:
            raise WorkUnavailableError("Work Sessions is shutting down.")

    def _require_mutable(self, thread_id: str) -> None:
        if thread_id in self._deleting:
            raise WorkConflictError("This Work session is being deleted.")
        if thread_id in self._active_turns:
            raise WorkConflictError(
                "Work settings cannot change while this session is running."
            )

    async def _require_project(self, cwd: str) -> None:
        if not await _project_is_available(cwd):
            raise WorkConflictError(
                "This Work session's project folder is unavailable."
            )

    def _publish_session(self, event: str, thread_id: str) -> None:
        if thread_id not in self._records:
            return
        self._events.publish(event, {"thread_id": thread_id})

    def _publish_status(self, thread_id: str) -> None:
        if thread_id not in self._records:
            return
        self._events.publish(
            "session.status",
            {"thread_id": thread_id, "status": self._status_for(thread_id).value},
        )

    def _spawn(self, coroutine: Coroutine[object, object, None], *, name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            logger.error("Work background task failed: exc_type={}", type(exc).__name__)

    def _clear_thread_projection(self, thread_id: str) -> None:
        for key in tuple(self._live_items):
            if key[0] == thread_id:
                self._live_items.pop(key, None)
        for interaction_id, pending in tuple(self._interactions.items()):
            if pending.public.thread_id == thread_id:
                self._retire_interaction(interaction_id)
        active = self._active_turns.pop(thread_id, None)
        if active is not None:
            active.terminal.set()

    def _retire_interaction(self, interaction_id: str) -> None:
        pending = self._interactions.pop(interaction_id, None)
        if pending is not None:
            self._interaction_keys.pop(
                (pending.connection_id, pending.request_id),
                None,
            )

    async def _decline_on_shutdown(self, pending: _PendingInteraction) -> None:
        if pending.claimed:
            return
        with suppress(Exception):
            await self._codex.respond(
                connection_id=pending.connection_id,
                request_id=pending.request_id,
                result=_safe_decline_result(pending.method, pending.params),
            )


async def _canonical_project_path(value: str) -> tuple[str, str]:
    stripped = value.strip()
    if not stripped:
        raise WorkValidationError("Enter an absolute project folder path.")

    def resolve() -> tuple[str, str]:
        path = Path(stripped).expanduser()
        if not path.is_absolute():
            raise WorkValidationError("Project folder must be an absolute path.")
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise WorkValidationError("Project folder does not exist.") from exc
        if not resolved.is_dir():
            raise WorkValidationError("Project path must be a folder.")
        display = str(resolved)
        return display, os.path.normcase(display)

    return await anyio.to_thread.run_sync(resolve)


async def _project_is_available(value: str) -> bool:
    return await anyio.to_thread.run_sync(lambda: Path(value).is_dir())


def _turn_page(
    thread_id: str,
    turns: tuple[JsonObject, ...],
    next_cursor: str | None,
) -> WorkTurnPage:
    items: list[WorkTimelineItem] = []
    for turn in reversed(turns):
        turn_id = _optional_string(turn.get("id"))
        native_items = turn.get("items")
        if turn_id is None or not isinstance(native_items, list):
            continue
        items.extend(
            _timeline_item(thread_id, turn_id, native_item)
            for native_item in native_items
            if isinstance(native_item, dict)
        )
    return WorkTurnPage(items=tuple(items), next_cursor=next_cursor)


def _timeline_item(
    thread_id: str,
    turn_id: str,
    item: JsonObject,
) -> WorkTimelineItem:
    item_id = (
        _optional_string(item.get("id"))
        or _intent_digest(
            thread_id,
            turn_id,
            json.dumps(item, sort_keys=True, separators=(",", ":"), default=str),
        )[:24]
    )
    kind = _optional_string(item.get("type")) or "codexActivity"
    status = _optional_string(item.get("status"))
    return WorkTimelineItem(
        thread_id=thread_id,
        turn_id=turn_id,
        item_id=item_id,
        kind=kind,
        status=status,
        text=_item_text(kind, item),
        payload=dict(item),
    )


def _item_text(kind: str, item: JsonObject) -> str | None:
    direct_fields = {
        "agentMessage": "text",
        "plan": "text",
        "commandExecution": "aggregatedOutput",
        "webSearch": "query",
        "imageView": "path",
    }
    field_name = direct_fields.get(kind)
    if field_name is not None:
        direct = _optional_string(item.get(field_name))
        if direct:
            return direct
    if kind == "userMessage":
        content = item.get("content")
        if isinstance(content, list):
            values: list[str] = []
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                text = _optional_string(entry.get("text"))
                if text:
                    values.append(text)
                elif entry.get("type") in {"image", "inputImage"}:
                    values.append("[Image]")
            return "\n".join(values) or None
    if kind == "reasoning":
        return _json_text(item.get("summary")) or _json_text(item.get("content"))
    if kind == "commandExecution":
        command = item.get("command")
        if isinstance(command, list):
            return " ".join(value for value in command if isinstance(value, str))
        return _optional_string(command)
    if kind in {"mcpToolCall", "dynamicToolCall"}:
        return _optional_string(item.get("tool"))
    if kind == "fileChange":
        return _json_text(item.get("changes"))
    return None


def _updated_settings(
    current: WorkSessionSettings,
    updates: JsonObject,
    catalog: CodexControlCatalog,
) -> WorkSessionSettings:
    allowed = {
        "model",
        "reasoning_effort",
        "collaboration_mode",
        "permission_profile",
    }
    if not updates or set(updates).difference(allowed):
        raise WorkValidationError("Choose only supported Work settings to update.")
    values: dict[str, str | None] = {
        "model": current.model,
        "reasoning_effort": current.reasoning_effort,
        "collaboration_mode": current.collaboration_mode,
        "permission_profile": current.permission_profile,
    }
    for key, value in updates.items():
        if value is not None and not isinstance(value, str):
            raise WorkValidationError(f"Work setting {key!r} must be text or null.")
        values[key] = value
    settings = WorkSessionSettings(
        model=values["model"],
        reasoning_effort=values["reasoning_effort"],
        collaboration_mode=values["collaboration_mode"],
        permission_profile=values["permission_profile"],
    )
    _validate_settings(settings, catalog)
    return settings


def _validate_settings(
    settings: WorkSessionSettings,
    catalog: CodexControlCatalog,
) -> None:
    model = settings.model
    if model is not None and model not in _model_ids(catalog):
        raise WorkValidationError("Choose a model advertised by Codex.")
    effort = settings.reasoning_effort
    efforts = _reasoning_efforts(catalog, model)
    if effort is not None and effort not in efforts:
        raise WorkValidationError(
            "Choose a reasoning effort advertised for this model."
        )
    collaboration = settings.collaboration_mode
    if collaboration is not None and collaboration not in _collaboration_names(catalog):
        raise WorkValidationError("Choose a collaboration mode advertised by Codex.")
    permission = settings.permission_profile
    if permission is not None and permission not in _permission_ids(catalog):
        raise WorkValidationError("Choose a permission profile allowed by Codex.")


def _controls_payload(catalog: CodexControlCatalog) -> JsonObject:
    return {
        "models": list(catalog.models or ()),
        "collaboration_modes": list(catalog.collaboration_modes or ()),
        "permission_profiles": list(catalog.permission_profiles or ()),
    }


def _model_ids(catalog: CodexControlCatalog) -> set[str]:
    result: set[str] = set()
    for model in catalog.models or ():
        value = _optional_string(model.get("model")) or _optional_string(
            model.get("id")
        )
        if value:
            result.add(value)
    return result


def _reasoning_efforts(catalog: CodexControlCatalog, model_id: str | None) -> set[str]:
    result: set[str] = set()
    for model in catalog.models or ():
        value = _optional_string(model.get("model")) or _optional_string(
            model.get("id")
        )
        if model_id is not None and value != model_id:
            continue
        options = model.get("supportedReasoningEfforts")
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict):
                    effort = _optional_string(option.get("reasoningEffort"))
                else:
                    effort = _optional_string(option)
                if effort:
                    result.add(effort)
    return result


def _collaboration_names(catalog: CodexControlCatalog) -> set[str]:
    return {
        name
        for item in catalog.collaboration_modes or ()
        if (name := _optional_string(item.get("name"))) is not None
    }


def _permission_ids(catalog: CodexControlCatalog) -> set[str]:
    return {
        profile_id
        for item in catalog.permission_profiles or ()
        if item.get("allowed") is not False
        if (profile_id := _optional_string(item.get("id"))) is not None
    }


def _collaboration_value(
    name: str | None,
    catalog: CodexControlCatalog,
) -> JsonObject | None:
    if name is None:
        return None
    for item in catalog.collaboration_modes or ():
        if item.get("name") == name:
            return {
                key: value
                for key, value in item.items()
                if key in {"mode", "model", "reasoning_effort"}
            }
    raise WorkValidationError("The selected collaboration mode is no longer available.")


def _settings_from_start(response: JsonObject) -> WorkSessionSettings:
    return WorkSessionSettings(
        model=_optional_string(response.get("model")),
        reasoning_effort=_optional_string(response.get("reasoningEffort")),
        collaboration_mode=None,
        permission_profile=None,
    )


def _interaction_kind(method: str) -> WorkInteractionKind | None:
    return {
        _COMMAND_APPROVAL: WorkInteractionKind.COMMAND_APPROVAL,
        _FILE_APPROVAL: WorkInteractionKind.FILE_CHANGE_APPROVAL,
        _PERMISSION_APPROVAL: WorkInteractionKind.PERMISSION_APPROVAL,
        _USER_INPUT: WorkInteractionKind.USER_INPUT,
    }.get(method)


def _interaction_title(kind: WorkInteractionKind) -> str:
    return {
        WorkInteractionKind.COMMAND_APPROVAL: "Command approval",
        WorkInteractionKind.FILE_CHANGE_APPROVAL: "File change approval",
        WorkInteractionKind.PERMISSION_APPROVAL: "Additional permission",
        WorkInteractionKind.USER_INPUT: "Codex needs your input",
    }[kind]


def _public_interaction_payload(
    kind: WorkInteractionKind, params: JsonObject
) -> JsonObject:
    if kind is WorkInteractionKind.COMMAND_APPROVAL:
        return {
            "command": params.get("command"),
            "cwd": params.get("cwd"),
            "reason": params.get("reason"),
            "available_decisions": params.get("availableDecisions"),
        }
    if kind is WorkInteractionKind.FILE_CHANGE_APPROVAL:
        return {
            "reason": params.get("reason"),
            "grant_root": params.get("grantRoot"),
        }
    if kind is WorkInteractionKind.PERMISSION_APPROVAL:
        return {
            "reason": params.get("reason"),
            "permissions": params.get("permissions"),
        }
    return {"questions": params.get("questions")}


def _interaction_response(
    pending: _PendingInteraction,
    value: JsonValue,
) -> JsonObject:
    body = _as_object(value)
    if pending.method == _COMMAND_APPROVAL:
        decision = _required_string(body, "decision")
        available = pending.params.get("availableDecisions")
        allowed = (
            {candidate for candidate in available if isinstance(candidate, str)}
            if isinstance(available, list)
            else {"accept", "decline", "cancel"}
        )
        if decision not in allowed:
            raise WorkValidationError("Choose an approval decision offered by Codex.")
        return {"decision": decision}
    if pending.method == _FILE_APPROVAL:
        decision = _required_string(body, "decision")
        if decision not in {"accept", "acceptForSession", "decline", "cancel"}:
            raise WorkValidationError("Choose a valid file-change decision.")
        return {"decision": decision}
    if pending.method == _PERMISSION_APPROVAL:
        decision = _required_string(body, "decision")
        scope = _optional_string(body.get("scope")) or "turn"
        if decision not in {"accept", "decline"} or scope not in {"turn", "session"}:
            raise WorkValidationError("Choose a valid permission decision and scope.")
        permissions = pending.params.get("permissions") if decision == "accept" else {}
        if not isinstance(permissions, dict):
            raise WorkValidationError("Codex supplied an invalid permission request.")
        return {"permissions": permissions, "scope": scope}
    if pending.method == _USER_INPUT:
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise WorkValidationError("Answer every Codex question.")
        questions = pending.params.get("questions")
        if not isinstance(questions, list):
            raise WorkValidationError("Codex supplied invalid questions.")
        allowed_ids = {
            question_id
            for question in questions
            if isinstance(question, dict)
            if (question_id := _optional_string(question.get("id"))) is not None
        }
        if set(answers) != allowed_ids:
            raise WorkValidationError("Answer exactly the questions Codex asked.")
        questions_by_id = {
            question_id: question
            for question in questions
            if isinstance(question, dict)
            if (question_id := _optional_string(question.get("id"))) is not None
        }
        normalized: JsonObject = {}
        for question_id, answer in answers.items():
            if not isinstance(answer, list) or not all(
                isinstance(entry, str) for entry in answer
            ):
                raise WorkValidationError("Codex answers must be lists of text values.")
            question = questions_by_id[question_id]
            options = question.get("options")
            if isinstance(options, list) and question.get("isOther") is not True:
                labels = {
                    label
                    for option in options
                    if isinstance(option, dict)
                    if (label := _optional_string(option.get("label"))) is not None
                }
                if any(entry not in labels for entry in answer):
                    raise WorkValidationError("Choose only answers offered by Codex.")
            normalized[question_id] = {"answers": answer}
        return {"answers": normalized}
    raise WorkValidationError("This Codex interaction is not supported.")


def _safe_decline_result(method: str, params: JsonObject) -> JsonObject:
    del params
    if method == _COMMAND_APPROVAL:
        return {"decision": "decline"}
    if method == _FILE_APPROVAL:
        return {"decision": "decline"}
    if method == _PERMISSION_APPROVAL:
        return {"permissions": {}, "scope": "turn"}
    if method == _USER_INPUT:
        return {"answers": {}}
    if method == _MCP_ELICITATION:
        return {"action": "cancel"}
    return {"decision": "abort"}


def _interaction_payload(interaction: WorkInteraction) -> JsonObject:
    return {
        "interaction_id": interaction.interaction_id,
        "thread_id": interaction.thread_id,
        "turn_id": interaction.turn_id,
        "kind": interaction.kind.value,
        "title": interaction.title,
        "payload": interaction.payload,
    }


def _timeline_payload(item: WorkTimelineItem) -> JsonObject:
    return {
        "thread_id": item.thread_id,
        "turn_id": item.turn_id,
        "item_id": item.item_id,
        "kind": item.kind,
        "status": item.status,
        "text": item.text,
        "payload": item.payload,
    }


def _event_thread_id(method: str, params: JsonObject) -> str | None:
    thread_id = _optional_string(params.get("threadId"))
    if thread_id is not None:
        return thread_id
    if method == "thread/started":
        return _optional_string(_as_object(params.get("thread")).get("id"))
    return None


def _native_status(value: JsonValue) -> WorkStatus:
    status = _as_object(value)
    kind = status.get("type")
    flags = status.get("activeFlags")
    if kind == "active" and isinstance(flags, list):
        if "waitingOnUserInput" in flags:
            return WorkStatus.WAITING_FOR_INPUT
        if "waitingOnApproval" in flags:
            return WorkStatus.WAITING_FOR_APPROVAL
        return WorkStatus.WORKING
    if kind == "systemError":
        return WorkStatus.FAILED
    if kind == "notLoaded":
        return WorkStatus.READY
    return WorkStatus.READY


def _turn_status(turn: JsonObject) -> WorkStatus:
    status = _optional_string(turn.get("status"))
    if status in {"interrupted", "cancelled", "canceled"}:
        return WorkStatus.INTERRUPTED
    if status in {"failed", "error"}:
        return WorkStatus.FAILED
    return WorkStatus.COMPLETED


def _turn_error(turn: JsonObject) -> str:
    error = turn.get("error")
    if isinstance(error, dict):
        message = _optional_string(error.get("message"))
        if message:
            return message
    return "Codex could not complete this turn."


def _kind_from_delta(method: str) -> str:
    if "agentMessage" in method:
        return "agentMessage"
    if "commandExecution" in method:
        return "commandExecution"
    if "fileChange" in method:
        return "fileChange"
    if "reasoning" in method:
        return "reasoning"
    if "plan" in method:
        return "plan"
    return "mcpToolCall"


def _as_object(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _required_string(value: JsonObject, key: str) -> str:
    result = _optional_string(value.get(key))
    if result is None:
        raise WorkValidationError(f"Codex value {key!r} is missing.")
    return result


def _optional_string(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _json_text(value: JsonValue) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _native_timestamp_ms(value: JsonValue) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = int(value)
    return timestamp if timestamp >= 1_000_000_000_000 else timestamp * 1_000


def _title_from_preview(preview: str) -> str:
    collapsed = " ".join(preview.split())
    if len(collapsed) <= 64:
        return collapsed
    return f"{collapsed[:61].rstrip()}..."


def _summary_time(summary: WorkSessionSummary) -> int:
    return summary.updated_at_ms or summary.registered_at_ms


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise WorkValidationError(f"Invalid {label} ID.") from exc
    canonical = str(parsed)
    if parsed.version != 4 or canonical != value.lower():
        raise WorkValidationError(f"Invalid {label} ID.")
    return canonical


def _intent_digest(*values: str) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _acknowledgement(operation: WorkOperation) -> WorkOperationAcknowledgement:
    return WorkOperationAcknowledgement(
        operation_id=operation.operation_id,
        kind=operation.kind,
        state=operation.state,
        thread_id=operation.result_thread_id or operation.session_id,
        turn_id=operation.result_turn_id,
    )


def _is_native_not_found(exc: CodexRequestError) -> bool:
    message = exc.message.casefold()
    return exc.code in {-32001, -32602} and "not found" in message


def _work_error(exc: Exception) -> WorkUnavailableError | WorkCompatibilityError:
    if isinstance(exc, WorkCompatibilityError):
        return exc
    if isinstance(exc, CodexCompatibilityError) or (
        isinstance(exc, CodexRequestError) and exc.code == -32601
    ):
        return WorkCompatibilityError("Update Codex to use this Work Sessions feature.")
    if isinstance(exc, (CodexUnavailableError, CodexConnectionError)):
        return WorkUnavailableError(_safe_error_message(exc))
    return WorkUnavailableError("Codex could not complete this Work operation.")


def _safe_error_message(exc: Exception) -> str:
    if isinstance(
        exc,
        (
            WorkValidationError,
            WorkConflictError,
            WorkUnavailableError,
            WorkCompatibilityError,
            CodexUnavailableError,
            CodexConnectionError,
            CodexCompatibilityError,
            CodexRequestError,
        ),
    ):
        return str(exc)
    return f"Work operation failed ({type(exc).__name__})."


def _now_ms() -> int:
    return time.time_ns() // 1_000_000

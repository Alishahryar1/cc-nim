"""Durable Work command coordination and per-session state ownership."""

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial

from loguru import logger

from free_claude_code.application.event_feed import EventPublisher, EventSubscription
from free_claude_code.core.json_types import JsonObject, JsonValue

from .codex import (
    CodexAppServerEvent,
    CodexAppServerPort,
    CodexCompatibilityError,
    CodexConnectionError,
    CodexConnectionLost,
    CodexDelivery,
    CodexInteractionKind,
    CodexInteractionRequest,
    CodexInteractionResponse,
    CodexNotification,
    CodexObjectPage,
    CodexProtocolError,
    CodexRequestError,
    CodexThreadHandle,
    CodexThreadSettings,
    CodexTurnHandle,
    CodexTurnSettings,
    CodexUnavailableError,
    CodexUnsupportedInteraction,
)
from .models import (
    WorkConflictError,
    WorkInteraction,
    WorkOperation,
    WorkOperationAcknowledgement,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkStatus,
    WorkTimelineItem,
    WorkUnavailableError,
    WorkValidationError,
)
from .operation_policy import ACTIVE_OPERATION_STATES, derive_work_status
from .ports import WorkStorePort
from .projection import (
    ProjectionState,
    apply_notification,
    as_object,
    begin_turn,
    clear_completed,
    complete_turn,
    history_contains_turn,
    native_status,
    notification_thread_id,
    optional_string,
    public_interaction,
    turn_error,
    turn_status,
)

_NATIVE_PAGE_LIMIT = 100
_EARLY_CREATE_EVENT_GRACE_SECONDS = 1.0
_READ_ONLY_RECONCILE_DELAY_SECONDS = 0.5
_TERMINAL_TURN_STATUSES = frozenset(
    {"completed", "failed", "interrupted", "cancelled", "canceled"}
)


@dataclass(frozen=True, slots=True)
class CoordinatorSessionSnapshot:
    record: WorkSessionRecord
    status: WorkStatus
    native_thread: JsonObject | None
    missing: bool
    projection: ProjectionState
    interactions: tuple[WorkInteraction, ...]
    active_turn_id: str | None
    operations: tuple[WorkOperation, ...]


@dataclass(slots=True)
class _PendingInteraction:
    public: WorkInteraction
    request: CodexInteractionRequest
    operation_id: str | None = None
    response_written: bool = False
    resolved: bool = False


@dataclass(slots=True)
class _SessionState:
    record: WorkSessionRecord
    native_status: WorkStatus = WorkStatus.READY
    native_thread: JsonObject | None = None
    missing: bool = False
    projection: ProjectionState = field(default_factory=ProjectionState)
    interactions: dict[str, _PendingInteraction] = field(default_factory=dict)
    request_keys: dict[tuple[str, int | str], str] = field(default_factory=dict)
    operations: dict[str, WorkOperation] = field(default_factory=dict)
    active_turn_id: str | None = None
    interrupt_in_flight: bool = False
    delete_in_flight: bool = False
    disconnected: bool = False
    retired: bool = False


@dataclass(frozen=True, slots=True)
class _Dispatch:
    operation_id: str


@dataclass(frozen=True, slots=True)
class _Recover:
    operation: WorkOperation
    completed: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _RecordChanged:
    record: WorkSessionRecord


@dataclass(frozen=True, slots=True)
class _OperationChanged:
    operation: WorkOperation
    completed: asyncio.Future[None] | None = None


@dataclass(frozen=True, slots=True)
class _AcknowledgeUnknown:
    completed: asyncio.Future[tuple[WorkOperation, ...]]


@dataclass(frozen=True, slots=True)
class _Native:
    event: CodexAppServerEvent


@dataclass(frozen=True, slots=True)
class _EffectDone:
    operation_id: str
    phase: str
    value: object | None = None
    error: Exception | None = None


type _SessionMessage = (
    _Dispatch
    | _Recover
    | _RecordChanged
    | _OperationChanged
    | _AcknowledgeUnknown
    | _Native
    | _EffectDone
)


class WorkCoordinator:
    """Own durable dispatch, native events, and immutable session snapshots."""

    def __init__(
        self,
        codex: CodexAppServerPort,
        store: WorkStorePort,
        *,
        dispatch_interval_seconds: float = 1.0,
    ) -> None:
        if dispatch_interval_seconds <= 0:
            raise ValueError("Work dispatch interval must be positive.")
        self.codex = codex
        self.store = store
        self.events = EventPublisher()
        self.generation = str(uuid.uuid4())
        self._dispatch_interval = dispatch_interval_seconds
        self._wake = asyncio.Event()
        self._create_queue: asyncio.Queue[str] = asyncio.Queue()
        self._actors: dict[str, _SessionActor] = {}
        self._snapshots: dict[str, CoordinatorSessionSnapshot] = {}
        self._buffered_events: dict[str, list[CodexAppServerEvent]] = {}
        self._active_create_operation_id: str | None = None
        self._active_create_evidence = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._create_task: asyncio.Task[None] | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False

    async def start(
        self,
        records: tuple[WorkSessionRecord, ...],
        operations: tuple[WorkOperation, ...],
    ) -> None:
        if self._started:
            return
        self._started = True
        self._closing = False
        operations_by_session: dict[str, list[WorkOperation]] = {}
        for operation in operations:
            if operation.session_id is not None:
                operations_by_session.setdefault(operation.session_id, []).append(
                    operation
                )
        for record in records:
            self._register_actor(
                record, tuple(operations_by_session.get(record.thread_id, ()))
            )
        self._event_task = asyncio.create_task(
            self._event_loop(), name="fcc-work-events"
        )
        await self._recover_startup(operations)
        self._dispatcher_task = asyncio.create_task(
            self._dispatch_loop(), name="fcc-work-dispatch"
        )
        self._create_task = asyncio.create_task(
            self._create_loop(), name="fcc-work-create"
        )
        self._wake.set()

    async def close(self) -> None:
        if not self._started:
            return
        self._closing = True
        tasks = tuple(
            task
            for task in (
                self._dispatcher_task,
                self._create_task,
                self._event_task,
                *(actor.task for actor in self._actors.values()),
                *self._tasks,
            )
            if task is not None
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.codex.close()
        self.events.close()
        self._actors.clear()
        self._tasks.clear()
        self._started = False

    def nudge(self) -> None:
        if self._started and not self._closing:
            self._wake.set()

    def subscribe(self) -> EventSubscription:
        return self.events.subscribe()

    def snapshot(self, thread_id: str) -> CoordinatorSessionSnapshot | None:
        return self._snapshots.get(thread_id)

    def snapshots(self) -> tuple[CoordinatorSessionSnapshot, ...]:
        return tuple(self._snapshots.values())

    async def replace_record(self, record: WorkSessionRecord) -> None:
        actor = self._actors.get(record.thread_id)
        if actor is None:
            actor = self._register_actor(record)
        await actor.queue.put(_RecordChanged(record))

    async def acknowledge_unknown(self, thread_id: str) -> tuple[WorkOperation, ...]:
        actor = self._actors.get(thread_id)
        if actor is None:
            raise WorkConflictError("Work session is not active.")
        return await actor.acknowledge_unknown()

    async def interaction_response(
        self,
        thread_id: str,
        interaction_id: str,
        value: JsonValue,
    ) -> tuple[CodexInteractionResponse, str]:
        actor = self._actors.get(thread_id)
        if actor is None:
            raise WorkConflictError("Work session is not active.")
        return actor.interaction_response(interaction_id, value)

    async def confirmed_missing(self, thread_id: str) -> bool:
        try:
            await self.codex.read_thread(thread_id)
        except CodexRequestError as exc:
            return _is_native_not_found(exc)
        except CodexUnavailableError, CodexConnectionError:
            return False
        return False

    async def unregister(self, thread_id: str) -> None:
        actor = self._actors.pop(thread_id, None)
        if actor is not None:
            actor.task.cancel()
            await asyncio.gather(actor.task, return_exceptions=True)
        self._snapshots.pop(thread_id, None)

    def retire_actor(self, thread_id: str, actor: _SessionActor) -> None:
        if self._actors.get(thread_id) is actor:
            self._actors.pop(thread_id, None)
        self._snapshots.pop(thread_id, None)

    def launch_effect(
        self,
        queue: asyncio.Queue[_SessionMessage],
        *,
        operation_id: str,
        phase: str,
        effect: Callable[[], Awaitable[object | None]],
    ) -> None:
        async def run() -> None:
            try:
                value = await effect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._closing:
                    await queue.put(_EffectDone(operation_id, phase, error=exc))
            else:
                if not self._closing:
                    await queue.put(_EffectDone(operation_id, phase, value=value))

        task = asyncio.create_task(
            run(), name=f"fcc-work-effect-{phase}-{operation_id}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._effect_finished)

    async def settle(
        self,
        operation_id: str,
        state: WorkOperationState,
        *,
        expected_states: tuple[WorkOperationState, ...] = (
            WorkOperationState.EXECUTING,
            WorkOperationState.UNKNOWN,
        ),
        native_thread_id: str | None = None,
        native_turn_id: str | None = None,
        native_connection_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkOperation:
        operation = await self.store.transition_operation(
            operation_id,
            expected_states=expected_states,
            state=state,
            native_thread_id=native_thread_id,
            native_turn_id=native_turn_id,
            native_connection_id=native_connection_id,
            error_code=error_code,
            error_message=error_message,
        )
        self.publish_operation(operation)
        return operation

    def publish_status(self, snapshot: CoordinatorSessionSnapshot) -> None:
        self.events.publish(
            "session.status",
            {
                "thread_id": snapshot.record.thread_id,
                "status": snapshot.status.value,
            },
        )

    def publish_timeline(self, item: WorkTimelineItem) -> None:
        self.events.publish(
            "timeline.item",
            {
                "thread_id": item.thread_id,
                "turn_id": item.turn_id,
                "item_id": item.item_id,
                "kind": item.kind,
                "status": item.status,
                "text": item.text,
                "payload": item.payload,
            },
        )

    def update_snapshot(self, state: _SessionState) -> CoordinatorSessionSnapshot:
        operations = tuple(
            sorted(
                (
                    operation
                    for operation in state.operations.values()
                    if operation.state in ACTIVE_OPERATION_STATES
                ),
                key=lambda operation: (operation.created_at_ms, operation.operation_id),
            )
        )
        all_interactions = tuple(
            pending.public for pending in state.interactions.values()
        )
        snapshot = CoordinatorSessionSnapshot(
            record=state.record,
            status=derive_work_status(
                operations,
                all_interactions,
                native_status=state.native_status,
                disconnected=state.disconnected,
            ),
            native_thread=(
                dict(state.native_thread) if state.native_thread is not None else None
            ),
            missing=state.missing,
            projection=state.projection,
            interactions=tuple(
                pending.public
                for pending in state.interactions.values()
                if pending.operation_id is None
            ),
            active_turn_id=state.active_turn_id,
            operations=operations,
        )
        self._snapshots[state.record.thread_id] = snapshot
        return snapshot

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                self._wake.clear()
                operations = await self.store.list_operations(
                    states=(WorkOperationState.ACCEPTED,)
                )
                for operation in operations:
                    if operation.kind is WorkOperationKind.CREATE:
                        await self._create_queue.put(operation.operation_id)
                        continue
                    session_id = operation.session_id
                    actor = self._actors.get(session_id or "")
                    if actor is None:
                        await self.settle(
                            operation.operation_id,
                            WorkOperationState.FAILED,
                            expected_states=(WorkOperationState.ACCEPTED,),
                            error_code="session_missing",
                            error_message="Work session was removed before dispatch.",
                        )
                        continue
                    await actor.queue.put(_Dispatch(operation.operation_id))
                try:
                    async with asyncio.timeout(self._dispatch_interval):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Work dispatcher stopped unexpectedly: exc_type={}",
                type(exc).__name__,
            )

    async def _create_loop(self) -> None:
        while True:
            operation_id = await self._create_queue.get()
            try:
                await self._execute_create(operation_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Work create coordinator failed: exc_type={}",
                    type(exc).__name__,
                )

    async def _execute_create(self, operation_id: str) -> None:
        operation = await self.store.claim_operation(operation_id)
        if operation is None:
            return
        payload = operation.payload or {}
        cwd = optional_string(payload.get("cwd"))
        cwd_key = optional_string(payload.get("cwd_key"))
        if cwd is None or cwd_key is None:
            await self.settle(
                operation_id,
                WorkOperationState.FAILED,
                error_code="invalid_payload",
                error_message="Create operation payload is invalid.",
            )
            return
        self._active_create_operation_id = operation_id
        self._active_create_evidence.clear()
        try:
            await self._execute_claimed_create(operation, cwd=cwd, cwd_key=cwd_key)
        finally:
            if self._active_create_operation_id == operation_id:
                self._active_create_operation_id = None

    async def _execute_claimed_create(
        self,
        operation: WorkOperation,
        *,
        cwd: str,
        cwd_key: str,
    ) -> None:
        handle: CodexThreadHandle | None = None
        try:
            handle = await self.codex.start_thread(CodexThreadSettings(cwd=cwd))
            model, effort = _settings_from_start(handle.response)
            await _finish_store_write(
                self.store.record_operation_evidence(
                    operation.operation_id,
                    native_thread_id=handle.thread_id,
                    native_connection_id=handle.connection_id,
                    captured_model=model,
                    captured_reasoning_effort=effort,
                )
            )
            materialized = await self._materialize_or_confirm(handle.thread_id)
            if not materialized:
                raise WorkUnavailableError(
                    "Codex did not make the new Work session discoverable."
                )
            record = WorkSessionRecord(
                thread_id=handle.thread_id,
                cwd=cwd,
                cwd_key=cwd_key,
                settings=WorkSessionSettings(model=model, reasoning_effort=effort),
                revision=1,
                registered_at_ms=_now_ms(),
            )
            completed, persisted = await _finish_store_write(
                self.store.create_session_from_operation(operation.operation_id, record)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_or_reconcile_create(operation.operation_id, handle, exc)
            return
        actor = self._register_actor(persisted)
        native = handle.response.get("thread")
        if isinstance(native, dict):
            await actor.queue.put(
                _Native(
                    CodexNotification(
                        connection_id=handle.connection_id,
                        method="thread/started",
                        params={"thread": native},
                    )
                )
            )
        for event in self._buffered_events.pop(handle.thread_id, []):
            await actor.queue.put(_Native(event))
        self.publish_operation(completed)
        self.events.publish("session.created", {"thread_id": handle.thread_id})

    async def _materialize_or_confirm(self, thread_id: str) -> bool:
        try:
            await self.codex.materialize_thread(thread_id)
        except CodexConnectionError as exc:
            if exc.delivery is CodexDelivery.DEFINITELY_NOT_WRITTEN:
                raise
        except CodexRequestError, CodexCompatibilityError:
            raise
        return await self._thread_is_materialized(thread_id)

    async def _fail_or_reconcile_create(
        self,
        operation_id: str,
        handle: CodexThreadHandle | None,
        exc: Exception,
    ) -> None:
        if (
            handle is None
            and _mutation_failure_state(exc) is WorkOperationState.UNKNOWN
            and not self._active_create_evidence.is_set()
        ):
            try:
                async with asyncio.timeout(_EARLY_CREATE_EVENT_GRACE_SECONDS):
                    await self._active_create_evidence.wait()
            except TimeoutError:
                pass
        operation = await self.store.get_operation(operation_id)
        thread_id = (
            handle.thread_id if handle is not None else operation.native_thread_id
        )
        connection_id = (
            handle.connection_id
            if handle is not None
            else operation.native_connection_id
        )
        if thread_id is None:
            target = _mutation_failure_state(exc)
            await self.settle(
                operation_id,
                target,
                error_code=_error_code(exc),
                error_message=_safe_error_message(exc),
            )
            return
        materialized: bool | None
        try:
            materialized = await self._thread_is_materialized(thread_id)
        except Exception:
            materialized = None
        if materialized:
            try:
                await self._register_reconciled_create(
                    operation,
                    thread_id=thread_id,
                    handle=handle,
                )
            except Exception:
                await self.settle(
                    operation_id,
                    WorkOperationState.UNKNOWN,
                    native_thread_id=thread_id,
                    native_connection_id=connection_id,
                    error_code=_error_code(exc),
                    error_message=(
                        "The native thread exists, but FCC could not register it."
                    ),
                )
                return
            return
        if materialized is False:
            cleanup = await self._delete_exact_thread(thread_id)
            if cleanup is True:
                await self.settle(
                    operation_id,
                    WorkOperationState.FAILED,
                    native_thread_id=thread_id,
                    native_connection_id=connection_id,
                    error_code=_error_code(exc),
                    error_message=_safe_error_message(exc),
                )
                return
        await self.settle(
            operation_id,
            WorkOperationState.UNKNOWN,
            native_thread_id=thread_id,
            native_connection_id=connection_id,
            error_code=_error_code(exc),
            error_message=(
                "FCC could not prove whether Codex created this Work session."
            ),
        )

    async def _event_loop(self) -> None:
        try:
            async for event in self.codex.events():
                await self._route_native_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Work Codex event pump failed: exc_type={}", type(exc).__name__
            )
            await self._route_native_event(
                CodexConnectionLost("unknown", "Codex event stream stopped.")
            )

    async def _route_native_event(self, event: CodexAppServerEvent) -> None:
        if isinstance(event, CodexConnectionLost):
            for actor in tuple(self._actors.values()):
                await actor.queue.put(_Native(event))
            self.events.publish("work.disconnected", {"message": event.message})
            return
        if isinstance(event, CodexUnsupportedInteraction):
            self.events.publish(
                "work.warning",
                {
                    "message": f"Codex requested an unsupported interaction: {event.method}"
                },
            )
            return
        if isinstance(event, CodexInteractionRequest):
            thread_id = event.thread_id
        else:
            thread_id = notification_thread_id(event)
        if thread_id is None:
            return
        actor = self._actors.get(thread_id)
        if actor is None:
            if (
                isinstance(event, CodexNotification)
                and event.method == "thread/started"
                and self._active_create_operation_id is not None
            ):
                await _finish_store_write(
                    self.store.record_operation_evidence(
                        self._active_create_operation_id,
                        native_thread_id=thread_id,
                        native_connection_id=event.connection_id,
                    )
                )
                self._active_create_evidence.set()
            self._buffered_events.setdefault(thread_id, []).append(event)
            return
        await actor.queue.put(_Native(event))

    async def _recover_startup(self, operations: tuple[WorkOperation, ...]) -> None:
        for operation in operations:
            if self._closing:
                return
            if operation.state not in {
                WorkOperationState.EXECUTING,
                WorkOperationState.UNKNOWN,
            }:
                continue
            if operation.kind is WorkOperationKind.CREATE:
                await self._recover_create(operation)
                continue
            actor = self._actors.get(operation.session_id or "")
            if actor is None:
                await self.settle(
                    operation.operation_id,
                    WorkOperationState.UNKNOWN,
                    error_code="session_missing",
                    error_message="FCC cannot reconcile this operation without its session.",
                )
                continue
            await actor.recover(operation)

        for operation in operations:
            if (
                operation.state is WorkOperationState.ACCEPTED
                and operation.kind is WorkOperationKind.RESPOND
            ):
                failed = await self.settle(
                    operation.operation_id,
                    WorkOperationState.FAILED,
                    expected_states=(WorkOperationState.ACCEPTED,),
                    error_code="interaction_expired",
                    error_message="The Codex interaction expired when FCC restarted.",
                )
                actor = self._actors.get(operation.session_id or "")
                if actor is not None:
                    await actor.replace_operation(failed)

    async def _recover_create(self, operation: WorkOperation) -> None:
        thread_id = operation.native_thread_id
        if thread_id is None:
            await self.settle(
                operation.operation_id,
                WorkOperationState.UNKNOWN,
                expected_states=(operation.state,),
                error_code="create_outcome_unknown",
                error_message="Codex may have created an unregistered thread.",
            )
            return
        try:
            materialized = await self._thread_is_materialized(thread_id)
        except Exception as exc:
            await self.settle(
                operation.operation_id,
                WorkOperationState.UNKNOWN,
                expected_states=(operation.state,),
                native_thread_id=thread_id,
                error_code=_error_code(exc),
                error_message="FCC could not reconcile the interrupted create.",
            )
            return
        if not materialized:
            cleanup = await self._delete_exact_thread(thread_id)
            await self.settle(
                operation.operation_id,
                (
                    WorkOperationState.FAILED
                    if cleanup is True
                    else WorkOperationState.UNKNOWN
                ),
                expected_states=(operation.state,),
                native_thread_id=thread_id,
                error_code="create_interrupted",
                error_message=(
                    "The incomplete native thread was removed."
                    if cleanup is True
                    else "FCC could not prove cleanup of the incomplete native thread."
                ),
            )
            return
        try:
            await self._register_reconciled_create(
                operation,
                thread_id=thread_id,
                handle=None,
            )
        except Exception as exc:
            await self.settle(
                operation.operation_id,
                WorkOperationState.UNKNOWN,
                expected_states=(operation.state,),
                native_thread_id=thread_id,
                error_code=_error_code(exc),
                error_message="The native thread exists, but FCC could not register it.",
            )

    async def _register_reconciled_create(
        self,
        operation: WorkOperation,
        *,
        thread_id: str,
        handle: CodexThreadHandle | None,
    ) -> None:
        payload = operation.payload or {}
        cwd = optional_string(payload.get("cwd"))
        cwd_key = optional_string(payload.get("cwd_key"))
        if cwd is None or cwd_key is None:
            raise WorkValidationError("Create operation payload is invalid.")
        model = operation.captured_model
        effort = operation.captured_reasoning_effort
        if handle is not None:
            model, effort = _settings_from_start(handle.response)
        elif model is None:
            resumed = await self.codex.resume_thread(
                thread_id,
                CodexThreadSettings(cwd=cwd),
            )
            model, effort = _settings_from_start(resumed.response)
            await self.store.record_operation_evidence(
                operation.operation_id,
                native_thread_id=thread_id,
                native_connection_id=resumed.connection_id,
                captured_model=model,
                captured_reasoning_effort=effort,
            )
        completed, record = await self.store.create_session_from_operation(
            operation.operation_id,
            WorkSessionRecord(
                thread_id=thread_id,
                cwd=cwd,
                cwd_key=cwd_key,
                settings=WorkSessionSettings(model=model, reasoning_effort=effort),
                revision=1,
                registered_at_ms=operation.created_at_ms,
            ),
        )
        actor = self._register_actor(record)
        for event in self._buffered_events.pop(thread_id, []):
            await actor.queue.put(_Native(event))
        self.publish_operation(completed)
        self.events.publish("session.created", {"thread_id": thread_id})

    async def _thread_is_materialized(self, thread_id: str) -> bool:
        try:
            await self.codex.list_turns_page(
                thread_id=thread_id,
                cursor=None,
                limit=1,
            )
        except CodexRequestError as exc:
            if _is_unmaterialized_thread(exc):
                return False
            raise
        return True

    async def _delete_exact_thread(self, thread_id: str) -> bool | None:
        try:
            await self.codex.delete_thread(thread_id)
            return True
        except CodexRequestError as exc:
            return bool(_is_native_not_found(exc))
        except CodexConnectionError as exc:
            if exc.delivery is CodexDelivery.DEFINITELY_NOT_WRITTEN:
                return False
        except CodexUnavailableError:
            return None
        except CodexCompatibilityError:
            return False
        return None

    def _register_actor(
        self,
        record: WorkSessionRecord,
        operations: tuple[WorkOperation, ...] = (),
    ) -> _SessionActor:
        existing = self._actors.get(record.thread_id)
        if existing is not None:
            return existing
        actor = _SessionActor(self, record, operations)
        self._actors[record.thread_id] = actor
        self.update_snapshot(actor.state)
        return actor

    def publish_operation(self, operation: WorkOperation) -> None:
        self.events.publish(
            "operation.updated",
            _operation_payload(operation),
        )

    def _effect_finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.error(
                "Work native effect escaped supervision: exc_type={}",
                type(error).__name__,
            )


class _SessionActor:
    """Single FIFO owner for one registered Work session."""

    def __init__(
        self,
        coordinator: WorkCoordinator,
        record: WorkSessionRecord,
        operations: tuple[WorkOperation, ...],
    ) -> None:
        self.coordinator = coordinator
        self.state = _SessionState(
            record=record,
            operations={operation.operation_id: operation for operation in operations},
        )
        self.queue: asyncio.Queue[_SessionMessage] = asyncio.Queue()
        self.task = asyncio.create_task(
            self._run(), name=f"fcc-work-session-{record.thread_id}"
        )

    async def recover(self, operation: WorkOperation) -> None:
        completed = asyncio.get_running_loop().create_future()
        await self.queue.put(_Recover(operation, completed))
        await completed

    async def replace_operation(self, operation: WorkOperation) -> None:
        completed = asyncio.get_running_loop().create_future()
        await self.queue.put(_OperationChanged(operation, completed))
        await completed

    async def acknowledge_unknown(self) -> tuple[WorkOperation, ...]:
        completed: asyncio.Future[tuple[WorkOperation, ...]] = (
            asyncio.get_running_loop().create_future()
        )
        await self.queue.put(_AcknowledgeUnknown(completed))
        return await completed

    def interaction_response(
        self, interaction_id: str, value: JsonValue
    ) -> tuple[CodexInteractionResponse, str]:
        pending = self.state.interactions.get(interaction_id)
        if pending is None or pending.operation_id is not None:
            raise WorkConflictError("This Codex request was already answered.")
        response = _interaction_response(pending.request, value)
        canonical = json.dumps(
            response.result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return response, canonical

    def _active_operation(self, kind: WorkOperationKind) -> WorkOperation | None:
        return next(
            (
                operation
                for operation in self.state.operations.values()
                if operation.kind is kind and operation.state in ACTIVE_OPERATION_STATES
            ),
            None,
        )

    def _remember_operation(self, operation: WorkOperation) -> None:
        if operation.state.terminal:
            self.state.operations.pop(operation.operation_id, None)
        else:
            self.state.operations[operation.operation_id] = operation

    async def _settle(
        self,
        operation_id: str,
        state: WorkOperationState,
        *,
        expected_states: tuple[WorkOperationState, ...] = (
            WorkOperationState.EXECUTING,
            WorkOperationState.UNKNOWN,
        ),
        native_thread_id: str | None = None,
        native_turn_id: str | None = None,
        native_connection_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkOperation:
        operation = await self.coordinator.settle(
            operation_id,
            state,
            expected_states=expected_states,
            native_thread_id=native_thread_id,
            native_turn_id=native_turn_id,
            native_connection_id=native_connection_id,
            error_code=error_code,
            error_message=error_message,
        )
        self._remember_operation(operation)
        return operation

    async def _record_evidence(
        self,
        operation_id: str,
        *,
        native_thread_id: str | None = None,
        native_turn_id: str | None = None,
        native_connection_id: str | None = None,
    ) -> WorkOperation:
        operation = await self.coordinator.store.record_operation_evidence(
            operation_id,
            native_thread_id=native_thread_id,
            native_turn_id=native_turn_id,
            native_connection_id=native_connection_id,
        )
        self._remember_operation(operation)
        self.coordinator.publish_operation(operation)
        return operation

    async def _acknowledge_unknown(self) -> tuple[WorkOperation, ...]:
        active = tuple(
            operation
            for operation in self.state.operations.values()
            if operation.state in ACTIVE_OPERATION_STATES
        )
        if not active:
            return ()
        if any(
            operation.state is not WorkOperationState.UNKNOWN for operation in active
        ):
            raise WorkConflictError(
                "Wait for active Work operations before resolving uncertainty."
            )
        try:
            native, page = await asyncio.gather(
                self.coordinator.codex.read_thread(self.state.record.thread_id),
                self.coordinator.codex.list_turns_page(
                    thread_id=self.state.record.thread_id,
                    cursor=None,
                    limit=_NATIVE_PAGE_LIMIT,
                ),
            )
        except (CodexUnavailableError, CodexConnectionError) as exc:
            raise WorkConflictError(
                "Codex must be reachable before resolving uncertainty."
            ) from exc
        except CodexRequestError as exc:
            raise WorkConflictError(
                "The native Codex session must be readable before resolving uncertainty."
            ) from exc
        latest = page.records[0] if page.records else None
        if (
            isinstance(latest, dict)
            and optional_string(latest.get("status")) not in _TERMINAL_TURN_STATUSES
        ):
            raise WorkConflictError(
                "Wait for Codex to become idle before resolving uncertainty."
            )
        abandoned = await self.coordinator.store.abandon_unknown_operations(
            self.state.record.thread_id
        )
        self.state.native_thread = dict(native.thread)
        self.state.projection = ProjectionState()
        self.state.interactions.clear()
        self.state.request_keys.clear()
        self.state.active_turn_id = None
        self.state.disconnected = False
        self.state.native_status = (
            turn_status(latest) if isinstance(latest, dict) else WorkStatus.READY
        )
        for operation in abandoned:
            self._remember_operation(operation)
            self.coordinator.publish_operation(operation)
        self._changed("session.updated")
        return abandoned

    async def _run(self) -> None:
        while True:
            message = await self.queue.get()
            try:
                await self._handle(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Work session owner failed: thread_id={} exc_type={}",
                    self.state.record.thread_id,
                    type(exc).__name__,
                )
            if self.state.retired:
                return

    async def _handle(self, message: _SessionMessage) -> None:
        if isinstance(message, _Dispatch):
            await self._dispatch(message.operation_id)
        elif isinstance(message, _Recover):
            try:
                await self._recover(message.operation)
            except BaseException as exc:
                if message.completed is not None and not message.completed.done():
                    message.completed.set_exception(exc)
                raise
            else:
                if message.completed is not None and not message.completed.done():
                    message.completed.set_result(None)
        elif isinstance(message, _RecordChanged):
            self.state.record = message.record
            self._changed("session.updated")
        elif isinstance(message, _OperationChanged):
            self._remember_operation(message.operation)
            self._changed("session.updated")
            if message.completed is not None and not message.completed.done():
                message.completed.set_result(None)
        elif isinstance(message, _AcknowledgeUnknown):
            try:
                operations = await self._acknowledge_unknown()
            except BaseException as exc:
                if not message.completed.done():
                    message.completed.set_exception(exc)
                raise
            else:
                if not message.completed.done():
                    message.completed.set_result(operations)
        elif isinstance(message, _Native):
            await self._native(message.event)
        elif isinstance(message, _EffectDone):
            await self._effect_done(message)

    async def _dispatch(self, operation_id: str) -> None:
        operation = await self.coordinator.store.claim_operation(operation_id)
        if operation is None:
            return
        self._remember_operation(operation)
        self.coordinator.publish_operation(operation)
        if operation.kind is WorkOperationKind.SEND:
            await self._begin_send(operation)
        elif operation.kind is WorkOperationKind.STOP:
            await self._begin_stop(operation)
        elif operation.kind is WorkOperationKind.DELETE:
            await self._begin_delete(operation)
        elif operation.kind is WorkOperationKind.RESPOND:
            await self._begin_response(operation)

    async def _begin_send(self, operation: WorkOperation) -> None:
        other_active = tuple(
            item
            for item in self.state.operations.values()
            if item.operation_id != operation.operation_id
            and item.state in ACTIVE_OPERATION_STATES
        )
        if (
            other_active
            or self.state.active_turn_id is not None
            or self.state.projection.items
        ):
            await self._fail(
                operation,
                "conflict",
                "This Work session is not ready for another turn.",
            )
            return
        payload = operation.payload or {}
        text = optional_string(payload.get("text"))
        model = operation.captured_model
        if text is None or model is None:
            await self._fail(operation, "invalid_payload", "Send payload is invalid.")
            return
        self._changed()
        self.coordinator.launch_effect(
            self.queue,
            operation_id=operation.operation_id,
            phase="send_resume",
            effect=partial(
                self.coordinator.codex.resume_thread,
                self.state.record.thread_id,
                CodexThreadSettings(cwd=self.state.record.cwd, model=model),
            ),
        )

    async def _begin_stop(self, operation: WorkOperation) -> None:
        self._changed()
        if self.state.active_turn_id is not None:
            self._launch_interrupt(operation.operation_id)
        elif self._active_operation(WorkOperationKind.SEND) is None:
            self.coordinator.launch_effect(
                self.queue,
                operation_id=operation.operation_id,
                phase="stop_discover",
                effect=partial(
                    self.coordinator.codex.list_turns_page,
                    thread_id=self.state.record.thread_id,
                    cursor=None,
                    limit=1,
                ),
            )

    async def _begin_delete(self, operation: WorkOperation) -> None:
        other_active = tuple(
            item
            for item in self.state.operations.values()
            if item.operation_id != operation.operation_id
            and item.state in ACTIVE_OPERATION_STATES
        )
        if (
            other_active
            or self.state.active_turn_id is not None
            or self.state.interactions
        ):
            await self._fail(
                operation,
                "conflict",
                "Stop active Codex work before deleting this session.",
            )
            return
        self._changed()
        self._launch_delete(operation.operation_id)

    async def _begin_response(self, operation: WorkOperation) -> None:
        interaction_id = operation.interaction_id
        pending = self.state.interactions.get(interaction_id or "")
        payload = operation.payload or {}
        result = payload.get("result")
        kind_value = optional_string(payload.get("kind"))
        if (
            pending is None
            or pending.operation_id is not None
            or not isinstance(result, dict)
        ):
            await self._fail(
                operation,
                "interaction_expired",
                "This Codex request is no longer awaiting a response.",
            )
            return
        if kind_value != pending.request.kind.value:
            await self._fail(
                operation,
                "invalid_payload",
                "Interaction response kind does not match Codex.",
            )
            return
        pending.operation_id = operation.operation_id
        self._changed()
        self.coordinator.launch_effect(
            self.queue,
            operation_id=operation.operation_id,
            phase=f"respond:{pending.public.interaction_id}",
            effect=partial(
                self.coordinator.codex.respond,
                connection_id=pending.request.connection_id,
                request_id=pending.request.request_id,
                response=CodexInteractionResponse(pending.request.kind, dict(result)),
            ),
        )

    async def _recover(self, operation: WorkOperation) -> None:
        self._remember_operation(operation)
        if (
            operation.kind is WorkOperationKind.RESPOND
            and operation.state is WorkOperationState.EXECUTING
        ):
            await self._settle(
                operation.operation_id,
                WorkOperationState.UNKNOWN,
                error_code="interaction_outcome_unknown",
                error_message=(
                    "FCC restarted before it could confirm whether Codex accepted "
                    "this response."
                ),
            )
            self._changed("session.updated")
            return
        if operation.kind is WorkOperationKind.RESPOND:
            self._changed()
            return
        if operation.kind is WorkOperationKind.SEND:
            self._changed()
            try:
                page = await self.coordinator.codex.list_turns_page(
                    thread_id=self.state.record.thread_id,
                    cursor=None,
                    limit=_NATIVE_PAGE_LIMIT,
                )
            except Exception as exc:
                await self._send_reconciled(
                    _EffectDone(operation.operation_id, "send_reconcile", error=exc)
                )
            else:
                await self._send_reconciled(
                    _EffectDone(operation.operation_id, "send_reconcile", value=page)
                )
            return
        if operation.kind is WorkOperationKind.STOP:
            self.state.active_turn_id = operation.native_turn_id
            self._changed()
            try:
                page = await self.coordinator.codex.list_turns_page(
                    thread_id=self.state.record.thread_id,
                    cursor=None,
                    limit=1,
                )
            except Exception as exc:
                await self._stop_recovered(
                    _EffectDone(operation.operation_id, "stop_recover", error=exc)
                )
            else:
                await self._stop_recovered(
                    _EffectDone(operation.operation_id, "stop_recover", value=page)
                )
            return
        if operation.kind is WorkOperationKind.DELETE:
            self._changed()
            try:
                snapshot = await self.coordinator.codex.read_thread(
                    self.state.record.thread_id
                )
            except Exception as exc:
                await self._delete_reconciled(
                    _EffectDone(operation.operation_id, "delete_reconcile", error=exc)
                )
            else:
                await self._delete_reconciled(
                    _EffectDone(
                        operation.operation_id,
                        "delete_reconcile",
                        value=snapshot,
                    )
                )

    async def _native(self, event: CodexAppServerEvent) -> None:
        if isinstance(event, CodexConnectionLost):
            await self._connection_lost(event)
            return
        if isinstance(event, CodexInteractionRequest):
            await self._interaction(event)
            return
        if not isinstance(event, CodexNotification):
            return
        params = as_object(event.params)
        if event.method == "serverRequest/resolved":
            await self._interaction_resolved(event.connection_id, params)
            return
        if event.method == "thread/started":
            thread = as_object(params.get("thread"))
            if thread:
                self.state.native_thread = thread
                self.state.missing = False
                self._changed("session.updated")
            return
        if event.method == "thread/name/updated":
            if self.state.native_thread is not None:
                native = dict(self.state.native_thread)
                native["name"] = params.get("threadName")
                self.state.native_thread = native
            self._changed("session.updated")
            return
        if event.method == "thread/status/changed":
            self.state.native_status = native_status(params.get("status"))
            self._changed()
            return
        if event.method == "thread/deleted":
            self.state.missing = True
            self._changed("session.updated")
            return
        if event.method == "turn/started":
            await self._turn_started(params)
            return
        if event.method == "turn/completed":
            await self._turn_completed(as_object(params.get("turn")))
            return
        projected, item = apply_notification(
            self.state.projection,
            self.state.record.thread_id,
            event,
        )
        if item is not None:
            self.state.projection = projected
            self.coordinator.update_snapshot(self.state)
            self.coordinator.publish_timeline(item)

    async def _turn_started(self, params: JsonObject) -> None:
        turn = as_object(params.get("turn"))
        turn_id = optional_string(turn.get("id"))
        if turn_id is None:
            return
        active_send = self._active_operation(WorkOperationKind.SEND)
        if active_send is not None:
            self.state.active_turn_id = turn_id
            if not self.state.projection.items:
                self.state.projection = begin_turn(self.state.projection, turn_id)
            await self._record_evidence(
                active_send.operation_id,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=turn_id,
                native_connection_id=optional_string(params.get("connectionId")),
            )
            if (stop := self._active_operation(WorkOperationKind.STOP)) is not None:
                self._launch_interrupt(stop.operation_id)
        self.state.native_status = WorkStatus.WORKING
        self._changed()

    async def _turn_completed(self, turn: JsonObject) -> None:
        turn_id = optional_string(turn.get("id"))
        if turn_id is None:
            return
        status = turn_status(turn)
        active_send = self._active_operation(WorkOperationKind.SEND)
        matches = self.state.active_turn_id == turn_id
        if active_send is not None and not matches:
            return
        if active_send is not None and matches:
            self.state.active_turn_id = turn_id
            self.state.projection = complete_turn(self.state.projection, turn_id)
            await self._settle(
                active_send.operation_id,
                (
                    WorkOperationState.FAILED
                    if status is WorkStatus.FAILED
                    else WorkOperationState.SUCCEEDED
                ),
                native_thread_id=self.state.record.thread_id,
                native_turn_id=turn_id,
                error_code="turn_failed" if status is WorkStatus.FAILED else None,
                error_message=turn_error(turn) if status is WorkStatus.FAILED else None,
            )
        stop = self._active_operation(WorkOperationKind.STOP)
        if stop is not None and matches:
            await self._settle(
                stop.operation_id,
                WorkOperationState.SUCCEEDED,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=turn_id,
            )
            self.state.interrupt_in_flight = False
        self.state.active_turn_id = None
        self.state.native_status = status
        await self._retire_turn_interactions(turn_id)
        self._changed("session.updated")
        if self.state.projection.completed:
            self.coordinator.launch_effect(
                self.queue,
                operation_id=(
                    active_send.operation_id
                    if active_send is not None
                    else f"history-{turn_id}"
                ),
                phase=f"history:{turn_id}",
                effect=partial(
                    self.coordinator.codex.list_turns_page,
                    thread_id=self.state.record.thread_id,
                    cursor=None,
                    limit=_NATIVE_PAGE_LIMIT,
                ),
            )

    async def _interaction(self, request: CodexInteractionRequest) -> None:
        key = (request.connection_id, request.request_id)
        if key in self.state.request_keys:
            return
        interaction_id = str(uuid.uuid4())
        pending = _PendingInteraction(
            public=public_interaction(interaction_id, request),
            request=request,
        )
        self.state.interactions[interaction_id] = pending
        self.state.request_keys[key] = interaction_id
        self._changed()
        self.coordinator.events.publish(
            "interaction.created", _interaction_payload(pending.public)
        )

    async def _interaction_resolved(
        self, connection_id: str, params: JsonObject
    ) -> None:
        request_id = params.get("requestId")
        if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
            return
        interaction_id = self.state.request_keys.get((connection_id, request_id))
        pending = self.state.interactions.get(interaction_id or "")
        if pending is None:
            return
        pending.resolved = True
        if pending.operation_id is not None and pending.response_written:
            await self._settle(
                pending.operation_id,
                WorkOperationState.SUCCEEDED,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=pending.public.turn_id,
                native_connection_id=connection_id,
            )
            self._retire_interaction(pending.public.interaction_id)
        elif pending.operation_id is None:
            self._retire_interaction(pending.public.interaction_id)
        self._changed()

    async def _connection_lost(self, event: CodexConnectionLost) -> None:
        executing = tuple(
            operation
            for operation in self.state.operations.values()
            if operation.state is WorkOperationState.EXECUTING
        )
        for operation in executing:
            await self._settle(
                operation.operation_id,
                WorkOperationState.UNKNOWN,
                error_code="connection_lost",
                error_message=event.message,
            )
        self.state.active_turn_id = None
        self.state.interactions.clear()
        self.state.request_keys.clear()
        self.state.disconnected = True
        self._changed()

    async def _effect_done(self, effect: _EffectDone) -> None:
        if effect.phase == "send_resume":
            await self._send_resumed(effect)
        elif effect.phase == "send_start":
            await self._send_started(effect)
        elif effect.phase == "send_reconcile":
            await self._send_reconciled(effect)
        elif effect.phase == "stop_discover":
            await self._stop_discovered(effect)
        elif effect.phase == "stop_recover":
            await self._stop_recovered(effect)
        elif effect.phase == "stop_confirm":
            await self._stop_confirmed(effect)
        elif effect.phase == "stop_interrupt":
            await self._interrupt_done(effect)
        elif effect.phase == "delete_native":
            await self._delete_done(effect)
        elif effect.phase == "delete_reconcile":
            await self._delete_reconciled(effect)
        elif effect.phase.startswith("respond:"):
            await self._response_done(effect)
        elif effect.phase.startswith("history:"):
            await self._history_done(effect)

    async def _send_resumed(self, effect: _EffectDone) -> None:
        active_send = self._active_operation(WorkOperationKind.SEND)
        if active_send is None or active_send.operation_id != effect.operation_id:
            return
        if effect.error is not None:
            await self._settle_send_failure(effect, always_failed=True)
            return
        operation = await self.coordinator.store.get_operation(effect.operation_id)
        payload = operation.payload or {}
        text = optional_string(payload.get("text"))
        model = operation.captured_model
        if text is None or model is None:
            await self._fail(operation, "invalid_payload", "Send payload is invalid.")
            await self._clear_failed_send()
            return
        self.coordinator.launch_effect(
            self.queue,
            operation_id=operation.operation_id,
            phase="send_start",
            effect=partial(
                self.coordinator.codex.start_turn,
                thread_id=self.state.record.thread_id,
                text=text,
                settings=CodexTurnSettings(
                    model=model,
                    effort=operation.captured_reasoning_effort,
                ),
                client_user_message_id=operation.operation_id,
            ),
        )

    async def _send_started(self, effect: _EffectDone) -> None:
        active_send = self._active_operation(WorkOperationKind.SEND)
        if active_send is None or active_send.operation_id != effect.operation_id:
            return
        if effect.error is not None:
            await self._settle_send_failure(effect, always_failed=False)
            return
        if not isinstance(effect.value, CodexTurnHandle):
            await self._settle_send_failure(
                replace(effect, error=CodexProtocolError("Invalid turn response.")),
                always_failed=False,
            )
            return
        handle = effect.value
        self.state.active_turn_id = handle.turn_id
        if not self.state.projection.items:
            self.state.projection = begin_turn(self.state.projection, handle.turn_id)
        await self._record_evidence(
            effect.operation_id,
            native_thread_id=handle.thread_id,
            native_turn_id=handle.turn_id,
            native_connection_id=handle.connection_id,
        )
        if (stop := self._active_operation(WorkOperationKind.STOP)) is not None:
            self._launch_interrupt(stop.operation_id)
        self._changed()

    async def _settle_send_failure(
        self, effect: _EffectDone, *, always_failed: bool
    ) -> None:
        error = effect.error or WorkUnavailableError("Codex turn failed.")
        if always_failed or _mutation_failure_state(error) is WorkOperationState.FAILED:
            await self._settle(
                effect.operation_id,
                WorkOperationState.FAILED,
                error_code=_error_code(error),
                error_message=_safe_error_message(error),
            )
            await self._clear_failed_send()
            return
        self.coordinator.launch_effect(
            self.queue,
            operation_id=effect.operation_id,
            phase="send_reconcile",
            effect=partial(
                self.coordinator.codex.list_turns_page,
                thread_id=self.state.record.thread_id,
                cursor=None,
                limit=_NATIVE_PAGE_LIMIT,
            ),
        )

    async def _send_reconciled(self, effect: _EffectDone) -> None:
        if effect.error is not None:
            await self._mark_unknown(effect.operation_id, effect.error)
            return
        records = getattr(effect.value, "records", None)
        if not isinstance(records, tuple):
            await self._mark_unknown(
                effect.operation_id,
                CodexProtocolError("Codex history reconciliation failed."),
            )
            return
        turn = _find_operation_turn(records, effect.operation_id)
        if turn is None:
            await self._mark_unknown(
                effect.operation_id,
                WorkUnavailableError("Codex did not expose a matching turn."),
            )
            return
        turn_id = optional_string(turn.get("id"))
        if turn_id is None:
            await self._mark_unknown(
                effect.operation_id,
                CodexProtocolError("Matching Codex turn has no ID."),
            )
            return
        await self._record_evidence(
            effect.operation_id,
            native_thread_id=self.state.record.thread_id,
            native_turn_id=turn_id,
        )
        self.state.active_turn_id = turn_id
        native_turn_status = optional_string(turn.get("status"))
        if native_turn_status in _TERMINAL_TURN_STATUSES:
            await self._turn_completed(turn)
            return
        self.state.native_status = WorkStatus.WORKING
        self._changed()

    async def _stop_discovered(self, effect: _EffectDone) -> None:
        stop = self._active_operation(WorkOperationKind.STOP)
        if stop is None or stop.operation_id != effect.operation_id:
            return
        if effect.error is not None:
            await self._mark_unknown(effect.operation_id, effect.error)
            return
        records = getattr(effect.value, "records", None)
        latest = records[0] if isinstance(records, tuple) and records else None
        if (
            isinstance(latest, dict)
            and optional_string(latest.get("status")) not in _TERMINAL_TURN_STATUSES
        ):
            turn_id = optional_string(latest.get("id"))
            if turn_id is not None:
                self.state.active_turn_id = turn_id
                self._launch_interrupt(effect.operation_id)
                return
        await self._settle(
            effect.operation_id,
            WorkOperationState.SUCCEEDED,
            native_thread_id=self.state.record.thread_id,
        )
        self.state.native_status = WorkStatus.READY
        self._changed()

    async def _stop_recovered(self, effect: _EffectDone) -> None:
        stop = self._active_operation(WorkOperationKind.STOP)
        if stop is None or stop.operation_id != effect.operation_id:
            return
        if effect.error is not None:
            await self._mark_unknown(effect.operation_id, effect.error)
            return
        records = getattr(effect.value, "records", None)
        if not isinstance(records, tuple):
            await self._mark_unknown(
                effect.operation_id,
                CodexProtocolError("Codex Stop reconciliation failed."),
            )
            return
        target = _turn_record(records, self.state.active_turn_id)
        latest = target or (records[0] if records else None)
        if (
            isinstance(latest, dict)
            and optional_string(latest.get("status")) not in _TERMINAL_TURN_STATUSES
        ):
            await self._settle(
                effect.operation_id,
                WorkOperationState.FAILED,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=optional_string(latest.get("id")),
                error_code="turn_still_active",
                error_message="Codex still reports an active turn; retry Stop.",
            )
            self.state.native_status = WorkStatus.WORKING
            self._changed()
            return
        if self.state.active_turn_id is not None and target is None:
            await self._mark_unknown(
                effect.operation_id,
                WorkUnavailableError("Codex did not expose the interrupted turn."),
            )
            return
        if target is not None:
            await self._turn_completed(target)
            return
        await self._settle(
            effect.operation_id,
            WorkOperationState.SUCCEEDED,
            native_thread_id=self.state.record.thread_id,
        )
        self.state.native_status = WorkStatus.READY
        self._changed()

    async def _stop_confirmed(self, effect: _EffectDone) -> None:
        stop = self._active_operation(WorkOperationKind.STOP)
        if stop is None or stop.operation_id != effect.operation_id:
            return
        if effect.error is not None:
            await self._mark_unknown(effect.operation_id, effect.error)
            return
        records = getattr(effect.value, "records", None)
        if not isinstance(records, tuple):
            await self._mark_unknown(
                effect.operation_id,
                CodexProtocolError("Codex Stop confirmation failed."),
            )
            return
        target = _turn_record(records, self.state.active_turn_id)
        if target is None:
            await self._mark_unknown(
                effect.operation_id,
                WorkUnavailableError("Codex did not expose the interrupted turn."),
            )
            return
        if optional_string(target.get("status")) not in _TERMINAL_TURN_STATUSES:
            await self._settle(
                effect.operation_id,
                WorkOperationState.FAILED,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=optional_string(target.get("id")),
                error_code="turn_still_active",
                error_message="Codex still reports an active turn; retry Stop.",
            )
            self.state.native_status = WorkStatus.WORKING
            self._changed()
            return
        await self._turn_completed(target)

    async def _interrupt_done(self, effect: _EffectDone) -> None:
        self.state.interrupt_in_flight = False
        if effect.error is None:
            if isinstance(effect.value, WorkOperation):
                self._remember_operation(effect.value)
                self.coordinator.publish_operation(effect.value)
            stop = self._active_operation(WorkOperationKind.STOP)
            if stop is not None and stop.operation_id == effect.operation_id:
                self.coordinator.launch_effect(
                    self.queue,
                    operation_id=effect.operation_id,
                    phase="stop_confirm",
                    effect=partial(
                        _delayed_turns_page,
                        self.coordinator.codex,
                        self.state.record.thread_id,
                    ),
                )
            return
        target = _mutation_failure_state(effect.error)
        await self._settle(
            effect.operation_id,
            target,
            native_thread_id=self.state.record.thread_id,
            native_turn_id=self.state.active_turn_id,
            error_code=_error_code(effect.error),
            error_message=_safe_error_message(effect.error),
        )
        if target is WorkOperationState.FAILED:
            self.state.native_status = WorkStatus.WORKING
        self._changed()

    async def _delete_done(self, effect: _EffectDone) -> None:
        self.state.delete_in_flight = False
        if effect.error is None or (
            isinstance(effect.error, CodexRequestError)
            and _is_native_not_found(effect.error)
        ):
            await self._complete_delete(effect.operation_id)
            return
        if _mutation_failure_state(effect.error) is WorkOperationState.UNKNOWN:
            self.coordinator.launch_effect(
                self.queue,
                operation_id=effect.operation_id,
                phase="delete_reconcile",
                effect=partial(
                    self.coordinator.codex.read_thread, self.state.record.thread_id
                ),
            )
            return
        await self._settle(
            effect.operation_id,
            WorkOperationState.FAILED,
            native_thread_id=self.state.record.thread_id,
            error_code=_error_code(effect.error),
            error_message=_safe_error_message(effect.error),
        )
        self.state.native_status = WorkStatus.READY
        self._changed()

    async def _delete_reconciled(self, effect: _EffectDone) -> None:
        if effect.error is not None:
            if isinstance(effect.error, CodexRequestError) and _is_native_not_found(
                effect.error
            ):
                await self._complete_delete(effect.operation_id)
                return
            await self._mark_unknown(effect.operation_id, effect.error)
            return
        await self._settle(
            effect.operation_id,
            WorkOperationState.FAILED,
            native_thread_id=self.state.record.thread_id,
            error_code="delete_not_confirmed",
            error_message="Codex still reports this session; retry Delete.",
        )
        self.state.native_status = WorkStatus.READY
        self._changed()

    async def _complete_delete(self, operation_id: str) -> None:
        completed = await self.coordinator.store.complete_delete(
            operation_id, self.state.record.thread_id
        )
        self._remember_operation(completed)
        self.coordinator.publish_operation(completed)
        self.coordinator.events.publish(
            "session.deleted", {"thread_id": self.state.record.thread_id}
        )
        self.state.retired = True
        self.coordinator.retire_actor(self.state.record.thread_id, self)

    async def _response_done(self, effect: _EffectDone) -> None:
        pending = next(
            (
                item
                for item in self.state.interactions.values()
                if item.operation_id == effect.operation_id
            ),
            None,
        )
        if pending is None:
            return
        if effect.error is not None:
            target = _mutation_failure_state(effect.error)
            await self._settle(
                effect.operation_id,
                target,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=pending.public.turn_id,
                native_connection_id=pending.request.connection_id,
                error_code=_error_code(effect.error),
                error_message=_safe_error_message(effect.error),
            )
            self._retire_interaction(pending.public.interaction_id)
            self._changed()
            return
        pending.response_written = True
        if pending.resolved:
            await self._settle(
                effect.operation_id,
                WorkOperationState.SUCCEEDED,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=pending.public.turn_id,
                native_connection_id=pending.request.connection_id,
            )
            self._retire_interaction(pending.public.interaction_id)
            self._changed()

    async def _history_done(self, effect: _EffectDone) -> None:
        turn_id = effect.phase.partition(":")[2]
        if (
            self.state.projection.turn_id != turn_id
            or not self.state.projection.completed
        ):
            return
        records = getattr(effect.value, "records", None)
        if (
            effect.error is None
            and isinstance(records, tuple)
            and history_contains_turn(records, turn_id)
        ):
            self.state.projection = clear_completed(self.state.projection, turn_id)
            self._changed()
            return
        self.coordinator.launch_effect(
            self.queue,
            operation_id=effect.operation_id,
            phase=effect.phase,
            effect=partial(
                _delayed_turns_page,
                self.coordinator.codex,
                self.state.record.thread_id,
            ),
        )

    def _launch_interrupt(self, operation_id: str) -> None:
        if self.state.interrupt_in_flight or self.state.active_turn_id is None:
            return
        self.state.interrupt_in_flight = True
        self.coordinator.launch_effect(
            self.queue,
            operation_id=operation_id,
            phase="stop_interrupt",
            effect=partial(
                _interrupt_with_evidence,
                self.coordinator.store,
                self.coordinator.codex,
                operation_id=operation_id,
                thread_id=self.state.record.thread_id,
                turn_id=self.state.active_turn_id,
            ),
        )

    def _launch_delete(self, operation_id: str) -> None:
        if self.state.delete_in_flight:
            return
        self.state.delete_in_flight = True
        self.coordinator.launch_effect(
            self.queue,
            operation_id=operation_id,
            phase="delete_native",
            effect=partial(
                self.coordinator.codex.delete_thread, self.state.record.thread_id
            ),
        )

    async def _fail(self, operation: WorkOperation, code: str, message: str) -> None:
        await self._settle(
            operation.operation_id,
            WorkOperationState.FAILED,
            error_code=code,
            error_message=message,
        )

    async def _mark_unknown(self, operation_id: str, error: Exception) -> None:
        operation = await self._settle(
            operation_id,
            WorkOperationState.UNKNOWN,
            native_thread_id=self.state.record.thread_id,
            native_turn_id=self.state.active_turn_id,
            error_code=_error_code(error),
            error_message=_safe_error_message(error),
        )
        if (
            operation.kind is WorkOperationKind.SEND
            and self.state.active_turn_id is None
        ):
            stop = self._active_operation(WorkOperationKind.STOP)
            if stop is not None and stop.state in {
                WorkOperationState.ACCEPTED,
                WorkOperationState.EXECUTING,
            }:
                await self._settle(
                    stop.operation_id,
                    WorkOperationState.FAILED,
                    expected_states=(
                        WorkOperationState.ACCEPTED,
                        WorkOperationState.EXECUTING,
                    ),
                    native_thread_id=self.state.record.thread_id,
                    error_code="stop_not_attempted",
                    error_message=(
                        "FCC could not identify a turn to stop because the "
                        "send outcome is uncertain."
                    ),
                )
        if operation.kind is WorkOperationKind.STOP:
            self.state.interrupt_in_flight = False
        if operation.kind is WorkOperationKind.DELETE:
            self.state.delete_in_flight = False
        self._changed()

    async def _clear_failed_send(self) -> None:
        self.state.active_turn_id = None
        self.state.interrupt_in_flight = False
        stop = self._active_operation(WorkOperationKind.STOP)
        if stop is not None:
            if stop.state is WorkOperationState.ACCEPTED:
                claimed = await self.coordinator.store.claim_operation(
                    stop.operation_id
                )
                if claimed is not None:
                    stop = claimed
                    self._remember_operation(claimed)
                    self.coordinator.publish_operation(claimed)
            await self._settle(
                stop.operation_id,
                WorkOperationState.SUCCEEDED,
                native_thread_id=self.state.record.thread_id,
            )
        self.state.native_status = WorkStatus.FAILED
        self._changed()

    async def _retire_turn_interactions(self, turn_id: str) -> None:
        for interaction_id in tuple(self.state.interactions):
            pending = self.state.interactions[interaction_id]
            if pending.public.turn_id != turn_id:
                continue
            if pending.operation_id is None:
                self._retire_interaction(interaction_id)
                continue
            pending.resolved = True
            if not pending.response_written:
                continue
            await self._settle(
                pending.operation_id,
                WorkOperationState.SUCCEEDED,
                native_thread_id=self.state.record.thread_id,
                native_turn_id=turn_id,
                native_connection_id=pending.request.connection_id,
            )
            self._retire_interaction(interaction_id)

    def _retire_interaction(self, interaction_id: str) -> None:
        pending = self.state.interactions.pop(interaction_id, None)
        if pending is None:
            return
        self.state.request_keys.pop(
            (pending.request.connection_id, pending.request.request_id), None
        )
        self.coordinator.events.publish(
            "interaction.resolved",
            {
                "thread_id": self.state.record.thread_id,
                "interaction_id": interaction_id,
            },
        )

    def _changed(self, event: str | None = None) -> None:
        snapshot = self.coordinator.update_snapshot(self.state)
        self.coordinator.publish_status(snapshot)
        if event is not None:
            self.coordinator.events.publish(
                event, {"thread_id": self.state.record.thread_id}
            )


def _settings_from_start(response: JsonObject) -> tuple[str, str | None]:
    model = optional_string(response.get("model"))
    if model is None:
        raise CodexProtocolError("Codex thread/start omitted its model.")
    effort_value = response.get("reasoningEffort")
    if effort_value is not None and not isinstance(effort_value, str):
        raise CodexProtocolError(
            "Codex thread/start returned invalid reasoning effort."
        )
    return model, effort_value


def _interaction_response(
    request: CodexInteractionRequest,
    value: JsonValue,
) -> CodexInteractionResponse:
    body = as_object(value)
    if request.kind is CodexInteractionKind.COMMAND_APPROVAL:
        decision = _required_string(body, "decision")
        available = request.params.get("availableDecisions")
        allowed = (
            {candidate for candidate in available if isinstance(candidate, str)}
            if isinstance(available, list) and available
            else {"accept", "acceptForSession", "decline", "cancel"}
        )
        if decision not in allowed:
            raise WorkValidationError("Choose an approval decision offered by Codex.")
        result: JsonObject = {"decision": decision}
        amendment = body.get("execpolicy_amendment")
        proposed = request.params.get("proposedExecpolicyAmendment")
        if amendment is not None:
            if amendment != proposed:
                raise WorkValidationError(
                    "Use only the exec-policy amendment Codex proposed."
                )
            result["execpolicy_amendment"] = amendment
        network = body.get("network_policy_amendments")
        proposed_network = request.params.get("proposedNetworkPolicyAmendments")
        if network is not None:
            if network != proposed_network:
                raise WorkValidationError(
                    "Use only the network amendments Codex proposed."
                )
            result["network_policy_amendments"] = network
        return CodexInteractionResponse(request.kind, result)
    if request.kind is CodexInteractionKind.FILE_CHANGE_APPROVAL:
        decision = _required_string(body, "decision")
        if decision not in {"accept", "acceptForSession", "decline", "cancel"}:
            raise WorkValidationError("Choose a valid file-change decision.")
        return CodexInteractionResponse(request.kind, {"decision": decision})
    if request.kind is CodexInteractionKind.PERMISSION_APPROVAL:
        decision = _required_string(body, "decision")
        scope = optional_string(body.get("scope")) or "turn"
        if decision not in {"accept", "decline"} or scope not in {"turn", "session"}:
            raise WorkValidationError("Choose a valid permission decision and scope.")
        requested = request.params.get("permissions")
        if not isinstance(requested, dict):
            raise WorkValidationError("Codex supplied an invalid permission request.")
        permissions = body.get("permissions", requested if decision == "accept" else {})
        if not isinstance(permissions, dict):
            raise WorkValidationError("Permission grant must be an object.")
        if decision == "accept" and not _json_subset(permissions, requested):
            raise WorkValidationError("Permissions cannot exceed what Codex requested.")
        result = {"permissions": permissions, "scope": scope}
        strict = body.get("strictAutoReview")
        if strict is not None:
            if not isinstance(strict, bool):
                raise WorkValidationError("strictAutoReview must be true or false.")
            result["strictAutoReview"] = strict
        return CodexInteractionResponse(request.kind, result)
    answers = body.get("answers")
    questions = request.params.get("questions")
    if not isinstance(answers, dict) or not isinstance(questions, list):
        raise WorkValidationError("Answer every Codex question.")
    question_ids = {
        question_id
        for question in questions
        if isinstance(question, dict)
        if (question_id := optional_string(question.get("id"))) is not None
    }
    if set(answers) != question_ids:
        raise WorkValidationError("Answer exactly the questions Codex asked.")
    normalized: JsonObject = {}
    for question_id, answer in answers.items():
        if not isinstance(answer, list) or not all(
            isinstance(entry, str) for entry in answer
        ):
            raise WorkValidationError("Codex answers must be lists of text values.")
        normalized[question_id] = {"answers": answer}
    return CodexInteractionResponse(request.kind, {"answers": normalized})


def _find_operation_turn(
    turns: tuple[JsonObject, ...], operation_id: str
) -> JsonObject | None:
    for turn in turns:
        if _contains_client_message_id(turn, operation_id):
            return turn
    return None


def _turn_record(
    turns: tuple[JsonObject, ...], turn_id: str | None
) -> JsonObject | None:
    if turn_id is None:
        return None
    return next(
        (turn for turn in turns if optional_string(turn.get("id")) == turn_id),
        None,
    )


def _contains_client_message_id(value: JsonValue, operation_id: str) -> bool:
    if isinstance(value, dict):
        if value.get("clientUserMessageId") == operation_id:
            return True
        return any(
            _contains_client_message_id(nested, operation_id)
            for nested in value.values()
        )
    if isinstance(value, list):
        return any(
            _contains_client_message_id(nested, operation_id) for nested in value
        )
    return False


def _mutation_failure_state(exc: Exception) -> WorkOperationState:
    if isinstance(exc, (CodexRequestError, CodexCompatibilityError)):
        return WorkOperationState.FAILED
    if isinstance(exc, CodexConnectionError):
        return (
            WorkOperationState.FAILED
            if exc.delivery is CodexDelivery.DEFINITELY_NOT_WRITTEN
            else WorkOperationState.UNKNOWN
        )
    if isinstance(exc, (CodexUnavailableError, WorkValidationError)):
        return WorkOperationState.FAILED
    return WorkOperationState.UNKNOWN


def _is_native_not_found(exc: CodexRequestError) -> bool:
    if exc.method not in {"thread/read", "thread/delete"}:
        return False
    message = exc.message.casefold()
    return exc.code in {-32001, -32600, -32602} and "not found" in message


def _is_unmaterialized_thread(exc: CodexRequestError) -> bool:
    if exc.method != "thread/turns/list":
        return False
    message = exc.message.casefold()
    return exc.code in {-32000, -32001, -32600, -32602} and any(
        marker in message for marker in ("not found", "not materialized", "rollout")
    )


def _required_string(value: JsonObject, key: str) -> str:
    result = optional_string(value.get(key))
    if result is None:
        raise WorkValidationError(f"Codex value {key!r} is missing.")
    return result


def _json_subset(candidate: JsonValue, requested: JsonValue) -> bool:
    if isinstance(candidate, dict) and isinstance(requested, dict):
        return all(
            key in requested and _json_subset(value, requested[key])
            for key, value in candidate.items()
        )
    if isinstance(candidate, list) and isinstance(requested, list):
        return all(item in requested for item in candidate)
    return candidate == requested


def _operation_payload(operation: WorkOperation) -> JsonObject:
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind.value,
        "state": operation.state.value,
        "thread_id": operation.native_thread_id or operation.session_id,
        "turn_id": operation.native_turn_id,
        "error_code": operation.error_code,
        "error_message": operation.error_message,
    }


def _interaction_payload(interaction: WorkInteraction) -> JsonObject:
    return {
        "interaction_id": interaction.interaction_id,
        "thread_id": interaction.thread_id,
        "turn_id": interaction.turn_id,
        "kind": interaction.kind.value,
        "title": interaction.title,
        "payload": interaction.payload,
    }


def acknowledgement(operation: WorkOperation) -> WorkOperationAcknowledgement:
    return WorkOperationAcknowledgement(
        operation_id=operation.operation_id,
        kind=operation.kind,
        state=operation.state,
        thread_id=operation.native_thread_id or operation.session_id,
        turn_id=operation.native_turn_id,
        error_code=operation.error_code,
        error_message=operation.error_message,
    )


def _error_code(exc: Exception) -> str:
    return type(exc).__name__


def _safe_error_message(exc: Exception) -> str:
    if isinstance(
        exc,
        (
            WorkValidationError,
            WorkConflictError,
            WorkUnavailableError,
            CodexUnavailableError,
            CodexConnectionError,
            CodexCompatibilityError,
            CodexRequestError,
        ),
    ):
        return str(exc)
    return f"Work operation failed ({type(exc).__name__})."


async def _finish_store_write[T](awaitable: Awaitable[T]) -> T:
    """Finish a short durable write even if graceful shutdown cancels its caller."""

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


async def _interrupt_with_evidence(
    store: WorkStorePort,
    codex: CodexAppServerPort,
    *,
    operation_id: str,
    thread_id: str,
    turn_id: str,
) -> WorkOperation:
    operation = await store.record_operation_evidence(
        operation_id,
        native_thread_id=thread_id,
        native_turn_id=turn_id,
    )
    await codex.interrupt_turn(thread_id=thread_id, turn_id=turn_id)
    return operation


async def _delayed_turns_page(
    codex: CodexAppServerPort,
    thread_id: str,
) -> CodexObjectPage:
    await asyncio.sleep(_READ_ONLY_RECONCILE_DELAY_SECONDS)
    return await codex.list_turns_page(
        thread_id=thread_id,
        cursor=None,
        limit=_NATIVE_PAGE_LIMIT,
    )


def _now_ms() -> int:
    return time.time_ns() // 1_000_000

"""Application facade for local Codex-backed Work Sessions."""

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import suppress
from pathlib import Path

import anyio.to_thread
from loguru import logger

from free_claude_code.application.event_feed import EventSubscription
from free_claude_code.core.json_types import JsonObject, JsonValue

from .codex import (
    CodexAppServerPort,
    CodexCompatibilityError,
    CodexConnectionError,
    CodexControlCatalog,
    CodexRequestError,
    CodexUnavailableError,
)
from .models import (
    WorkBootstrap,
    WorkCompatibilityError,
    WorkConflictError,
    WorkOperationAcknowledgement,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionDetail,
    WorkSessionPage,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkSessionSummary,
    WorkStatus,
    WorkTurnPage,
    WorkUnavailableError,
    WorkValidationError,
)
from .operations import WorkCoordinator, acknowledgement
from .ports import WorkStorePort
from .projection import history_page, optional_string, without_persisted

_SESSION_PAGE_LIMIT = 25
_TURN_PAGE_LIMIT = 100
_NATIVE_PAGE_LIMIT = 100
_RECENT_PROJECT_LIMIT = 8
_MAX_MESSAGE_LENGTH = 1_000_000


class WorkService:
    """Own Work admission, read use cases, and lifecycle composition."""

    def __init__(
        self,
        codex: CodexAppServerPort,
        store: WorkStorePort,
        *,
        dispatch_interval_seconds: float = 1.0,
    ) -> None:
        self._codex = codex
        self._store = store
        self._coordinator = WorkCoordinator(
            codex,
            store,
            dispatch_interval_seconds=dispatch_interval_seconds,
        )
        self._started = False
        self._accepting = False
        self._unavailable_message: str | None = "Work Sessions is starting."

    async def start(self) -> None:
        if self._started:
            return
        try:
            await self._store.start()
            records = await self._store.list_sessions()
            operations = await self._store.list_operations(
                states=(
                    WorkOperationState.ACCEPTED,
                    WorkOperationState.EXECUTING,
                    WorkOperationState.UNKNOWN,
                )
            )
            await self._coordinator.start(records, operations)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with suppress(Exception):
                await self._coordinator.close()
            with suppress(Exception):
                await self._store.close()
            self._unavailable_message = (
                str(exc)
                if isinstance(exc, WorkUnavailableError)
                else "Work storage could not be opened."
            )
            logger.warning("Work Sessions unavailable: exc_type={}", type(exc).__name__)
            return
        self._started = True
        self._accepting = True
        self._unavailable_message = None

    async def close(self) -> None:
        self._accepting = False
        if self._started:
            await self._coordinator.close()
            await self._store.close()
        self._started = False
        self._unavailable_message = "Work Sessions is stopped."

    async def bootstrap(self) -> WorkBootstrap:
        self._require_store()
        availability = await self._codex.availability()
        reason = availability.reason
        if not availability.available and not reason:
            reason = "Install or update Codex to use Work Sessions."
        unresolved = await self._store.list_operations(
            states=(
                WorkOperationState.ACCEPTED,
                WorkOperationState.EXECUTING,
                WorkOperationState.UNKNOWN,
            )
        )
        return WorkBootstrap(
            available=availability.available,
            reason=reason,
            codex_version=availability.version,
            recent_projects=await self._store.recent_projects(
                limit=_RECENT_PROJECT_LIMIT
            ),
            unresolved_creates=tuple(
                acknowledgement(operation)
                for operation in unresolved
                if operation.kind is WorkOperationKind.CREATE
            ),
            event_generation=self._coordinator.generation,
            event_cursor=self._coordinator.events.cursor,
        )

    async def subscribe(self) -> EventSubscription:
        self._require_store()
        return self._coordinator.subscribe()

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> WorkSessionPage:
        self._require_store()
        records = await self._store.list_sessions()
        native, disconnected = await self._native_index(
            {record.thread_id for record in records}
        )
        summaries = tuple(
            await asyncio.gather(
                *(
                    self._summary(
                        record,
                        native.get(record.thread_id),
                        session_available=record.thread_id in native,
                        status_override=(
                            WorkStatus.DISCONNECTED if disconnected else None
                        ),
                    )
                    for record in records
                )
            )
        )
        normalized_query = query.strip().casefold()
        if normalized_query:
            summaries = tuple(
                summary
                for summary in summaries
                if normalized_query
                in " ".join((summary.title, summary.preview, summary.cwd)).casefold()
            )
        summaries = tuple(
            sorted(
                summaries,
                key=lambda item: (item.registered_at_ms, item.thread_id),
                reverse=True,
            )
        )
        if cursor is not None:
            summaries = tuple(
                summary
                for summary in summaries
                if (summary.registered_at_ms, summary.thread_id) < cursor
            )
        page_limit = max(1, min(_SESSION_PAGE_LIMIT, limit))
        selected = summaries[: page_limit + 1]
        has_more = len(selected) > page_limit
        selected = selected[:page_limit]
        next_cursor = None
        if has_more and selected:
            last = selected[-1]
            next_cursor = (last.registered_at_ms, last.thread_id)
        return WorkSessionPage(sessions=selected, next_cursor=next_cursor)

    async def create_session(
        self, *, cwd: str, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        canonical_path, path_key = await _canonical_project_path(cwd)
        operation, _created = await self._store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            interaction_id=None,
            intent_digest=_intent_digest("create", canonical_path),
            payload={"cwd": canonical_path, "cwd_key": path_key},
        )
        self._coordinator.nudge()
        return acknowledgement(operation)

    async def get_detail(self, thread_id: str) -> WorkSessionDetail:
        self._require_store()
        record = await self._store.get_session(thread_id)
        project_available = await _project_is_available(record.cwd)
        snapshot = self._coordinator.snapshot(thread_id)
        try:
            native_snapshot = await self._codex.read_thread(thread_id)
            page_task = self._codex.list_turns_page(
                thread_id=thread_id,
                cursor=None,
                limit=_TURN_PAGE_LIMIT,
            )
            if project_available:
                catalog, page = await asyncio.gather(
                    self._codex.controls(cwd=record.cwd),
                    page_task,
                )
            else:
                page = await page_task
                catalog = CodexControlCatalog(models=None, config=None)
        except CodexRequestError as exc:
            if _is_native_not_found(exc):
                return await self._placeholder_detail(
                    record,
                    project_available=project_available,
                    session_available=False,
                )
            raise _work_error(exc) from exc
        except CodexUnavailableError, CodexConnectionError:
            return await self._placeholder_detail(
                record,
                project_available=project_available,
                session_available=True,
                disconnected=True,
            )
        except Exception as exc:
            raise _work_error(exc) from exc
        turns = history_page(thread_id, page.records)
        native = native_snapshot.thread
        summary = await self._summary(record, native, session_available=True)
        live = (
            without_persisted(snapshot.projection, turns)
            if snapshot is not None
            else ()
        )
        return WorkSessionDetail(
            summary=summary,
            settings=record.settings,
            controls=_controls_payload(catalog),
            turns=turns,
            live_items=live,
            interactions=snapshot.interactions if snapshot is not None else (),
            operations=(
                tuple(acknowledgement(operation) for operation in snapshot.operations)
                if snapshot is not None
                else ()
            ),
            event_cursor=self._coordinator.events.cursor,
        )

    async def _placeholder_detail(
        self,
        record: WorkSessionRecord,
        *,
        project_available: bool,
        session_available: bool,
        disconnected: bool = False,
    ) -> WorkSessionDetail:
        snapshot = self._coordinator.snapshot(record.thread_id)
        summary = await self._summary(
            record,
            snapshot.native_thread if snapshot is not None else None,
            session_available=session_available,
            status_override=WorkStatus.DISCONNECTED if disconnected else None,
            project_available=project_available,
        )
        return WorkSessionDetail(
            summary=summary,
            settings=record.settings,
            controls={"models": []},
            turns=WorkTurnPage(items=()),
            live_items=snapshot.projection.items if snapshot is not None else (),
            interactions=snapshot.interactions if snapshot is not None else (),
            operations=(
                tuple(acknowledgement(operation) for operation in snapshot.operations)
                if snapshot is not None
                else ()
            ),
            event_cursor=self._coordinator.events.cursor,
        )

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        updates: JsonObject,
    ) -> WorkSessionRecord:
        self._require_accepting()
        current = await self._store.get_session(thread_id)
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
        await self._coordinator.replace_record(updated)
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
        record = await self._store.get_session(thread_id)
        await _require_project(record.cwd)
        normalized = text.strip()
        if not normalized:
            raise WorkValidationError("Write a message before sending.")
        if len(text) > _MAX_MESSAGE_LENGTH:
            raise WorkValidationError(
                "Work message cannot exceed 1,000,000 characters."
            )
        operation, _created = await self._store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.SEND,
            session_id=thread_id,
            interaction_id=None,
            intent_digest=_intent_digest(
                "send", thread_id, str(expected_revision), text
            ),
            payload={"text": text},
            expected_revision=expected_revision,
        )
        self._coordinator.nudge()
        return acknowledgement(operation)

    async def stop(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        operation, _created = await self._store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.STOP,
            session_id=thread_id,
            interaction_id=None,
            intent_digest=_intent_digest("stop", thread_id),
            payload={},
        )
        self._coordinator.nudge()
        return acknowledgement(operation)

    async def delete(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        snapshot = self._coordinator.snapshot(thread_id)
        if snapshot is not None and (
            snapshot.active_turn_id is not None or snapshot.interactions
        ):
            raise WorkConflictError(
                "Stop active Codex work before deleting this session."
            )
        operation, _created = await self._store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.DELETE,
            session_id=thread_id,
            interaction_id=None,
            intent_digest=_intent_digest("delete", thread_id),
            payload={},
        )
        self._coordinator.nudge()
        return acknowledgement(operation)

    async def remove_missing(self, thread_id: str) -> None:
        self._require_accepting()
        await self._store.get_session(thread_id)
        if not await self._coordinator.confirmed_missing(thread_id):
            raise WorkConflictError(
                "Only a confirmed missing Codex session can be removed from Work."
            )
        await self._store.delete_session(thread_id)
        await self._coordinator.unregister(thread_id)
        self._coordinator.events.publish("session.deleted", {"thread_id": thread_id})

    async def respond(
        self,
        thread_id: str,
        interaction_id: str,
        *,
        operation_id: str,
        value: JsonValue,
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        await self._store.get_session(thread_id)
        response, canonical = await self._coordinator.interaction_response(
            thread_id, interaction_id, value
        )
        operation, _created = await self._store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.RESPOND,
            session_id=thread_id,
            interaction_id=interaction_id,
            intent_digest=_intent_digest(
                "respond", thread_id, interaction_id, canonical
            ),
            payload={"kind": response.kind.value, "result": response.result},
        )
        self._coordinator.nudge()
        return acknowledgement(operation)

    async def get_operation(self, operation_id: str) -> WorkOperationAcknowledgement:
        self._require_store()
        operation = await self._store.get_operation(
            _canonical_uuid(operation_id, "operation")
        )
        return acknowledgement(operation)

    async def acknowledge_unknown(
        self, thread_id: str
    ) -> tuple[WorkOperationAcknowledgement, ...]:
        self._require_accepting()
        await self._store.get_session(thread_id)
        abandoned = await self._coordinator.acknowledge_unknown(thread_id)
        return tuple(acknowledgement(operation) for operation in abandoned)

    async def dismiss_unknown_create(
        self, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._require_accepting()
        operation_id = _canonical_uuid(operation_id, "operation")
        current = await self._store.get_operation(operation_id)
        if (
            current.kind is not WorkOperationKind.CREATE
            or current.state is not WorkOperationState.UNKNOWN
            or current.native_thread_id is not None
        ):
            raise WorkConflictError(
                "Only an unresolved creation without a native ID can be dismissed."
            )
        abandoned = await self._store.transition_operation(
            operation_id,
            expected_states=(WorkOperationState.UNKNOWN,),
            state=WorkOperationState.ABANDONED,
        )
        self._coordinator.publish_operation(abandoned)
        return acknowledgement(abandoned)

    async def _native_index(
        self, registered_ids: set[str]
    ) -> tuple[dict[str, JsonObject], bool]:
        native: dict[str, JsonObject] = {}
        cursor: str | None = None
        try:
            while True:
                page = await self._codex.list_threads_page(
                    cursor=cursor, limit=_NATIVE_PAGE_LIMIT
                )
                for thread in page.records:
                    thread_id = optional_string(thread.get("id"))
                    if thread_id is not None and thread_id in registered_ids:
                        native[thread_id] = dict(thread)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except CodexUnavailableError, CodexConnectionError:
            return (
                {
                    thread_id: snapshot.native_thread or {}
                    for thread_id in registered_ids
                    if (snapshot := self._coordinator.snapshot(thread_id)) is not None
                    and not snapshot.missing
                },
                True,
            )
        except Exception as exc:
            raise _work_error(exc) from exc
        return native, False

    async def _summary(
        self,
        record: WorkSessionRecord,
        native: JsonObject | None,
        *,
        session_available: bool,
        status_override: WorkStatus | None = None,
        project_available: bool | None = None,
    ) -> WorkSessionSummary:
        if project_available is None:
            project_available = await _project_is_available(record.cwd)
        snapshot = self._coordinator.snapshot(record.thread_id)
        native = native or (snapshot.native_thread if snapshot is not None else None)
        values = native or {}
        preview = optional_string(values.get("preview")) or ""
        name = optional_string(values.get("name"))
        title = name or (
            _title_from_preview(preview) if preview else "New Work Session"
        )
        status = status_override or WorkStatus.READY
        if snapshot is not None and (
            status_override is None or snapshot.operations or snapshot.interactions
        ):
            status = snapshot.status
        return WorkSessionSummary(
            thread_id=record.thread_id,
            cwd=record.cwd,
            title=title,
            preview=preview,
            status=status,
            revision=record.revision,
            registered_at_ms=record.registered_at_ms,
            updated_at_ms=_native_timestamp_ms(values.get("recencyAt")),
            project_available=project_available,
            session_available=session_available,
        )

    def _require_store(self) -> None:
        if not self._started:
            raise WorkUnavailableError(
                self._unavailable_message or "Work Sessions is unavailable."
            )

    def _require_accepting(self) -> None:
        self._require_store()
        if not self._accepting:
            raise WorkUnavailableError("Work Sessions is shutting down.")


async def _canonical_project_path(value: str) -> tuple[str, str]:
    raw = value.strip()
    if not raw:
        raise WorkValidationError("Choose a project folder.")

    def resolve() -> tuple[str, str]:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise WorkValidationError("Project path must be absolute.")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise WorkValidationError("Project folder does not exist.") from exc
        if not resolved.is_dir():
            raise WorkValidationError("Project path must be a folder.")
        display = str(resolved)
        return display, os.path.normcase(display)

    return await anyio.to_thread.run_sync(resolve)


async def _project_is_available(value: str) -> bool:
    return await anyio.to_thread.run_sync(lambda: Path(value).is_dir())


async def _require_project(value: str) -> None:
    if not await _project_is_available(value):
        raise WorkConflictError("This Work session's project folder is unavailable.")


def _updated_settings(
    current: WorkSessionSettings,
    updates: JsonObject,
    catalog: CodexControlCatalog,
) -> WorkSessionSettings:
    if not updates or set(updates).difference({"model", "reasoning_effort"}):
        raise WorkValidationError("Choose only model or reasoning to update.")
    models = _model_records(catalog)
    if not models:
        raise WorkCompatibilityError("Codex did not provide a usable model catalog.")
    model_value = updates.get("model", current.model)
    if not isinstance(model_value, str) or not model_value.strip():
        raise WorkValidationError("Choose a model advertised by Codex.")
    model = model_value.strip()
    model_record = models.get(model)
    if model_record is None:
        raise WorkValidationError("Choose a model advertised by Codex.")
    efforts = _reasoning_efforts(model_record)
    requested_effort = updates.get("reasoning_effort", current.reasoning_effort)
    if not efforts:
        effort = None
    elif isinstance(requested_effort, str) and requested_effort in efforts:
        effort = requested_effort
    elif "reasoning_effort" in updates and requested_effort is not None:
        raise WorkValidationError(
            "Choose a reasoning effort advertised for this model."
        )
    else:
        effort = _default_reasoning_effort(model_record, efforts)
    return WorkSessionSettings(model=model, reasoning_effort=effort)


def _controls_payload(catalog: CodexControlCatalog) -> JsonObject:
    return {"models": list(catalog.models or ())}


def _model_records(catalog: CodexControlCatalog) -> dict[str, JsonObject]:
    result: dict[str, JsonObject] = {}
    for model in catalog.models or ():
        model_id = optional_string(model.get("model")) or optional_string(
            model.get("id")
        )
        if model_id is not None:
            result[model_id] = model
    return result


def _reasoning_efforts(model: JsonObject) -> tuple[str, ...]:
    options = model.get("supportedReasoningEfforts")
    if not isinstance(options, list):
        return ()
    result: list[str] = []
    for option in options:
        value = (
            optional_string(option.get("reasoningEffort"))
            if isinstance(option, dict)
            else optional_string(option)
        )
        if value is not None and value not in result:
            result.append(value)
    return tuple(result)


def _default_reasoning_effort(model: JsonObject, efforts: tuple[str, ...]) -> str:
    advertised = optional_string(model.get("defaultReasoningEffort"))
    if advertised in efforts:
        return advertised
    return efforts[0]


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


def _is_native_not_found(exc: CodexRequestError) -> bool:
    if exc.method not in {"thread/read", "thread/delete"}:
        return False
    return (
        exc.code in {-32001, -32600, -32602} and "not found" in exc.message.casefold()
    )


def _work_error(exc: Exception) -> WorkUnavailableError | WorkCompatibilityError:
    if isinstance(exc, WorkCompatibilityError):
        return exc
    if isinstance(exc, CodexCompatibilityError) or (
        isinstance(exc, CodexRequestError) and exc.code == -32601
    ):
        return WorkCompatibilityError("Update Codex to use this Work Sessions feature.")
    if isinstance(exc, (CodexUnavailableError, CodexConnectionError)):
        return WorkUnavailableError(str(exc))
    return WorkUnavailableError("Codex could not complete this Work operation.")


def _native_timestamp_ms(value: JsonValue) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = int(value)
    return timestamp if timestamp >= 1_000_000_000_000 else timestamp * 1_000


def _title_from_preview(preview: str) -> str:
    collapsed = " ".join(preview.split())
    return collapsed if len(collapsed) <= 64 else f"{collapsed[:61].rstrip()}..."


def _now_ms() -> int:
    return time.time_ns() // 1_000_000

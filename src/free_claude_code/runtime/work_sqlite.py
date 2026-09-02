"""SQLite persistence owned exclusively by local Work Sessions."""

import json
import os
import sqlite3
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypeVar

import anyio.to_thread

from free_claude_code.application.work.models import (
    WorkConflictError,
    WorkNotFoundError,
    WorkOperation,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkUnavailableError,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.core.json_types import JsonObject

from .work_migrations import MIGRATIONS

T = TypeVar("T")

_BUSY_TIMEOUT_MS = 5_000
_ACTIVE_STATES = (
    WorkOperationState.ACCEPTED.value,
    WorkOperationState.EXECUTING.value,
    WorkOperationState.UNKNOWN.value,
)
_LEGAL_TRANSITIONS = {
    WorkOperationState.ACCEPTED: {
        WorkOperationState.EXECUTING,
        WorkOperationState.FAILED,
    },
    WorkOperationState.EXECUTING: {
        WorkOperationState.UNKNOWN,
        WorkOperationState.SUCCEEDED,
        WorkOperationState.FAILED,
    },
    WorkOperationState.UNKNOWN: {
        WorkOperationState.SUCCEEDED,
        WorkOperationState.FAILED,
        WorkOperationState.ABANDONED,
    },
}


class SQLiteWorkStore:
    """Short Work-registry transactions behind one process lock."""

    def __init__(self, database_path: Path, lock_path: Path) -> None:
        self._database_path = database_path
        self._lock = InterprocessFileLock(lock_path)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        directory = self._database_path.parent
        await anyio.to_thread.run_sync(
            partial(directory.mkdir, parents=True, exist_ok=True)
        )
        _owner_only(directory, 0o700)
        acquired = await anyio.to_thread.run_sync(self._lock.acquire)
        if not acquired:
            raise WorkUnavailableError(
                "Work Sessions is already open in another FCC server process."
            )
        try:
            await anyio.to_thread.run_sync(self._run_sync, self._migrate)
            _owner_only(self._database_path, 0o600)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{self._database_path}{suffix}")
                if sidecar.exists():
                    _owner_only(sidecar, 0o600)
        except sqlite3.Error as exc:
            self._lock.release()
            raise WorkUnavailableError("Work storage is unavailable.") from exc
        except BaseException:
            self._lock.release()
            raise
        self._started = True

    async def close(self) -> None:
        self._started = False
        await anyio.to_thread.run_sync(self._lock.release)

    async def list_sessions(self) -> tuple[WorkSessionRecord, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[WorkSessionRecord, ...]:
            rows = connection.execute(
                "SELECT * FROM work_sessions ORDER BY registered_at_ms DESC, thread_id"
            ).fetchall()
            return tuple(_session_from_row(row) for row in rows)

        return await self._run(operation)

    async def get_session(self, thread_id: str) -> WorkSessionRecord:
        return await self._run(lambda connection: _get_session(connection, thread_id))

    async def create_session_from_operation(
        self, operation_id: str, record: WorkSessionRecord
    ) -> tuple[WorkOperation, WorkSessionRecord]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[WorkOperation, WorkSessionRecord]:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_operation(connection, operation_id)
            if current.kind is not WorkOperationKind.CREATE:
                connection.rollback()
                raise WorkConflictError(
                    "Only a create operation can register a session."
                )
            if current.state is WorkOperationState.SUCCEEDED:
                result_id = current.native_thread_id
                if result_id is None:
                    connection.rollback()
                    raise WorkUnavailableError(
                        "Completed create has no native thread ID."
                    )
                existing = _get_session(connection, result_id)
                connection.commit()
                return current, existing
            if current.state is not WorkOperationState.EXECUTING:
                connection.rollback()
                raise WorkConflictError("Create operation is not executing.")
            if current.native_thread_id not in {None, record.thread_id}:
                connection.rollback()
                raise WorkConflictError("Create operation changed native thread ID.")
            try:
                _insert_session(connection, record)
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise WorkConflictError("Work session already exists.") from exc
            _update_operation_row(
                connection,
                current,
                state=WorkOperationState.SUCCEEDED,
                native_thread_id=record.thread_id,
                native_turn_id=None,
                native_connection_id=current.native_connection_id,
                error_code=None,
                error_message=None,
                captured_model=record.settings.model,
                captured_reasoning_effort=record.settings.reasoning_effort,
            )
            connection.commit()
            return (
                _get_operation(connection, operation_id),
                _get_session(connection, record.thread_id),
            )

        return await self._run(operation)

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        settings: WorkSessionSettings,
    ) -> WorkSessionRecord:
        def operation(connection: sqlite3.Connection) -> WorkSessionRecord:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_session(connection, thread_id)
            _expect_revision(current, expected_revision)
            active_blocker = connection.execute(
                """
                SELECT 1 FROM work_operations
                WHERE session_id = ?
                    AND (
                        kind IN ('stop', 'delete')
                        OR state = 'unknown'
                    )
                    AND state IN ('accepted', 'executing', 'unknown')
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            if active_blocker is not None:
                connection.rollback()
                raise WorkConflictError(
                    "Wait for the current Work operation before changing settings."
                )
            connection.execute(
                """
                UPDATE work_sessions
                SET model = ?, reasoning_effort = ?, revision = revision + 1
                WHERE thread_id = ?
                """,
                (settings.model, settings.reasoning_effort, thread_id),
            )
            connection.commit()
            return _get_session(connection, thread_id)

        return await self._run(operation)

    async def delete_session(self, thread_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute(
                "DELETE FROM work_sessions WHERE thread_id = ?", (thread_id,)
            ).rowcount
            if deleted != 1:
                connection.rollback()
                raise WorkNotFoundError("Work session was not found.")
            connection.commit()

        await self._run(operation)

    async def complete_delete(self, operation_id: str, thread_id: str) -> WorkOperation:
        def operation(connection: sqlite3.Connection) -> WorkOperation:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_operation(connection, operation_id)
            if current.kind is not WorkOperationKind.DELETE:
                connection.rollback()
                raise WorkConflictError("Only a delete operation can remove a session.")
            if current.state is WorkOperationState.SUCCEEDED:
                connection.commit()
                return current
            if current.state not in {
                WorkOperationState.EXECUTING,
                WorkOperationState.UNKNOWN,
            }:
                connection.rollback()
                raise WorkConflictError("Delete operation cannot be completed now.")
            connection.execute(
                "DELETE FROM work_sessions WHERE thread_id = ?", (thread_id,)
            )
            _update_operation_row(
                connection,
                current,
                state=WorkOperationState.SUCCEEDED,
                native_thread_id=thread_id,
                native_turn_id=current.native_turn_id,
                native_connection_id=current.native_connection_id,
                error_code=None,
                error_message=None,
            )
            connection.commit()
            return _get_operation(connection, operation_id)

        return await self._run(operation)

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
    ) -> tuple[WorkOperation, bool]:
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[WorkOperation, bool]:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is not None:
                existing = _operation_from_row(row)
                if (
                    existing.kind is not kind
                    or existing.session_id != session_id
                    or existing.interaction_id != interaction_id
                    or existing.intent_digest != intent_digest
                ):
                    connection.rollback()
                    raise WorkConflictError(
                        "Operation ID was already used for different Work input."
                    )
                connection.commit()
                return existing, False

            record: WorkSessionRecord | None = None
            if kind is not WorkOperationKind.CREATE:
                if session_id is None:
                    connection.rollback()
                    raise WorkConflictError("Work operation requires a session.")
                record = _get_session(connection, session_id)
                if expected_revision is not None:
                    _expect_revision(record, expected_revision)

            if session_id is not None and kind not in {
                WorkOperationKind.DELETE,
            }:
                deleting = connection.execute(
                    """
                    SELECT 1 FROM work_operations
                    WHERE session_id = ? AND kind = 'delete'
                        AND state IN ('accepted', 'executing', 'unknown')
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if deleting is not None:
                    connection.rollback()
                    raise WorkConflictError("This Work session is being deleted.")

            if kind in {WorkOperationKind.STOP, WorkOperationKind.DELETE}:
                existing_row = connection.execute(
                    """
                    SELECT * FROM work_operations
                    WHERE session_id = ? AND kind = ?
                        AND state IN ('accepted', 'executing', 'unknown')
                    ORDER BY created_at_ms, operation_id
                    LIMIT 1
                    """,
                    (session_id, kind.value),
                ).fetchone()
                if existing_row is not None:
                    connection.commit()
                    return _operation_from_row(existing_row), False

            if kind is WorkOperationKind.SEND:
                active_send = connection.execute(
                    """
                    SELECT 1 FROM work_operations
                    WHERE session_id = ?
                        AND (
                            kind IN ('send', 'stop')
                            OR state = 'unknown'
                        )
                        AND state IN ('accepted', 'executing', 'unknown')
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if active_send is not None:
                    connection.rollback()
                    raise WorkConflictError(
                        "This Work session already has an active or uncertain turn or Stop operation."
                    )

            if kind is WorkOperationKind.RESPOND:
                claimed = connection.execute(
                    """
                    SELECT 1 FROM work_operations
                    WHERE kind = 'respond' AND interaction_id = ?
                    LIMIT 1
                    """,
                    (interaction_id,),
                ).fetchone()
                if claimed is not None:
                    connection.rollback()
                    raise WorkConflictError("This Codex request was already answered.")

            captured_model = record.settings.model if record is not None else None
            captured_effort = (
                record.settings.reasoning_effort if record is not None else None
            )
            now = _now_ms()
            connection.execute(
                """
                INSERT INTO work_operations (
                    operation_id, kind, session_id, interaction_id,
                    intent_digest, payload_json, state, expected_revision,
                    captured_model, captured_reasoning_effort,
                    native_thread_id, native_turn_id, native_connection_id,
                    error_code, error_message, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'accepted', ?, ?, ?, NULL, NULL,
                          NULL, NULL, NULL, ?, ?)
                """,
                (
                    operation_id,
                    kind.value,
                    session_id,
                    interaction_id,
                    intent_digest,
                    payload_json,
                    expected_revision,
                    captured_model,
                    captured_effort,
                    now,
                    now,
                ),
            )
            connection.commit()
            return _get_operation(connection, operation_id), True

        return await self._run(operation)

    async def get_operation(self, operation_id: str) -> WorkOperation:
        return await self._run(
            lambda connection: _get_operation(connection, operation_id)
        )

    async def list_operations(
        self, *, states: tuple[WorkOperationState, ...]
    ) -> tuple[WorkOperation, ...]:
        if not states:
            return ()
        placeholders = ",".join("?" for _ in states)

        def operation(connection: sqlite3.Connection) -> tuple[WorkOperation, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM work_operations
                WHERE state IN ({placeholders})
                ORDER BY created_at_ms, operation_id
                """,
                tuple(state.value for state in states),
            ).fetchall()
            return tuple(_operation_from_row(row) for row in rows)

        return await self._run(operation)

    async def claim_operation(self, operation_id: str) -> WorkOperation | None:
        def operation(connection: sqlite3.Connection) -> WorkOperation | None:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_operation(connection, operation_id)
            if current.state is not WorkOperationState.ACCEPTED:
                connection.commit()
                return None
            _update_operation_row(
                connection,
                current,
                state=WorkOperationState.EXECUTING,
                native_thread_id=current.native_thread_id,
                native_turn_id=current.native_turn_id,
                native_connection_id=current.native_connection_id,
                error_code=None,
                error_message=None,
            )
            connection.commit()
            return _get_operation(connection, operation_id)

        return await self._run(operation)

    async def record_operation_evidence(
        self,
        operation_id: str,
        *,
        native_thread_id: str | None = None,
        native_turn_id: str | None = None,
        native_connection_id: str | None = None,
        captured_model: str | None = None,
        captured_reasoning_effort: str | None = None,
    ) -> WorkOperation:
        def operation(connection: sqlite3.Connection) -> WorkOperation:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_operation(connection, operation_id)
            if current.state is not WorkOperationState.EXECUTING:
                connection.commit()
                return current
            _update_operation_row(
                connection,
                current,
                state=current.state,
                native_thread_id=native_thread_id or current.native_thread_id,
                native_turn_id=native_turn_id or current.native_turn_id,
                native_connection_id=(
                    native_connection_id or current.native_connection_id
                ),
                error_code=current.error_code,
                error_message=current.error_message,
                captured_model=captured_model,
                captured_reasoning_effort=captured_reasoning_effort,
            )
            connection.commit()
            return _get_operation(connection, operation_id)

        return await self._run(operation)

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
    ) -> WorkOperation:
        def operation(connection: sqlite3.Connection) -> WorkOperation:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_operation(connection, operation_id)
            if current.state is state or current.state.terminal:
                connection.commit()
                return current
            if current.state not in expected_states:
                connection.rollback()
                raise WorkConflictError("Work operation changed before it was updated.")
            if state not in _LEGAL_TRANSITIONS.get(current.state, set()):
                connection.rollback()
                raise WorkConflictError("Invalid Work operation state transition.")
            _update_operation_row(
                connection,
                current,
                state=state,
                native_thread_id=native_thread_id or current.native_thread_id,
                native_turn_id=native_turn_id or current.native_turn_id,
                native_connection_id=(
                    native_connection_id or current.native_connection_id
                ),
                error_code=error_code,
                error_message=error_message,
            )
            connection.commit()
            return _get_operation(connection, operation_id)

        return await self._run(operation)

    async def recent_projects(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()

        def operation(connection: sqlite3.Connection) -> tuple[str, ...]:
            rows = connection.execute(
                """
                SELECT cwd, cwd_key FROM work_sessions
                ORDER BY registered_at_ms DESC, thread_id
                """
            ).fetchall()
            seen: set[str] = set()
            projects: list[str] = []
            for row in rows:
                key = _row_str(row, "cwd_key")
                if key in seen:
                    continue
                seen.add(key)
                projects.append(_row_str(row, "cwd"))
                if len(projects) == limit:
                    break
            return tuple(projects)

        return await self._run(operation)

    async def _run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if not self._started:
            raise WorkUnavailableError("Work storage is not available.")
        try:
            return await anyio.to_thread.run_sync(lambda: self._run_sync(operation))
        except sqlite3.Error as exc:
            raise WorkUnavailableError("Work storage is unavailable.") from exc

    def _run_sync(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        connection = sqlite3.connect(
            self._database_path,
            timeout=_BUSY_TIMEOUT_MS / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA journal_mode = WAL")
            return operation(connection)
        finally:
            connection.close()

    def _migrate(self, connection: sqlite3.Connection) -> None:
        _validate_migrations()
        row = connection.execute("PRAGMA user_version").fetchone()
        if row is None or not isinstance(row[0], int):
            raise WorkUnavailableError("Work database schema is invalid.")
        current = row[0]
        latest = MIGRATIONS[-1].version if MIGRATIONS else 0
        if current > latest:
            raise WorkUnavailableError(
                "Work data was created by a newer FCC version. Update FCC to open it."
            )
        for migration in MIGRATIONS:
            if migration.version <= current:
                continue
            connection.execute("BEGIN IMMEDIATE")
            try:
                migration.apply(connection)
                connection.execute(f"PRAGMA user_version = {migration.version}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            current = migration.version


def _validate_migrations() -> None:
    expected = tuple(range(1, len(MIGRATIONS) + 1))
    actual = tuple(migration.version for migration in MIGRATIONS)
    if actual != expected:
        raise RuntimeError("Work migrations must be ordered, unique, and contiguous.")


def _insert_session(connection: sqlite3.Connection, record: WorkSessionRecord) -> None:
    connection.execute(
        """
        INSERT INTO work_sessions (
            thread_id, cwd, cwd_key, model, reasoning_effort,
            revision, registered_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.thread_id,
            record.cwd,
            record.cwd_key,
            record.settings.model,
            record.settings.reasoning_effort,
            record.revision,
            record.registered_at_ms,
        ),
    )


def _get_session(connection: sqlite3.Connection, thread_id: str) -> WorkSessionRecord:
    row = connection.execute(
        "SELECT * FROM work_sessions WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    if row is None:
        raise WorkNotFoundError("Work session was not found.")
    return _session_from_row(row)


def _session_from_row(row: sqlite3.Row) -> WorkSessionRecord:
    model = _row_str(row, "model")
    if not model:
        raise WorkUnavailableError("Work storage contains an empty model.")
    return WorkSessionRecord(
        thread_id=_row_str(row, "thread_id"),
        cwd=_row_str(row, "cwd"),
        cwd_key=_row_str(row, "cwd_key"),
        settings=WorkSessionSettings(
            model=model,
            reasoning_effort=_row_optional_str(row, "reasoning_effort"),
        ),
        revision=_row_int(row, "revision"),
        registered_at_ms=_row_int(row, "registered_at_ms"),
    )


def _get_operation(connection: sqlite3.Connection, operation_id: str) -> WorkOperation:
    row = connection.execute(
        "SELECT * FROM work_operations WHERE operation_id = ?", (operation_id,)
    ).fetchone()
    if row is None:
        raise WorkNotFoundError("Work operation was not found.")
    return _operation_from_row(row)


def _operation_from_row(row: sqlite3.Row) -> WorkOperation:
    return WorkOperation(
        operation_id=_row_str(row, "operation_id"),
        kind=WorkOperationKind(_row_str(row, "kind")),
        session_id=_row_optional_str(row, "session_id"),
        interaction_id=_row_optional_str(row, "interaction_id"),
        intent_digest=_row_str(row, "intent_digest"),
        payload=_row_payload(row),
        state=WorkOperationState(_row_str(row, "state")),
        expected_revision=_row_optional_int(row, "expected_revision"),
        captured_model=_row_optional_str(row, "captured_model"),
        captured_reasoning_effort=_row_optional_str(row, "captured_reasoning_effort"),
        native_thread_id=_row_optional_str(row, "native_thread_id"),
        native_turn_id=_row_optional_str(row, "native_turn_id"),
        native_connection_id=_row_optional_str(row, "native_connection_id"),
        error_code=_row_optional_str(row, "error_code"),
        error_message=_row_optional_str(row, "error_message"),
        created_at_ms=_row_int(row, "created_at_ms"),
        updated_at_ms=_row_int(row, "updated_at_ms"),
    )


def _update_operation_row(
    connection: sqlite3.Connection,
    current: WorkOperation,
    *,
    state: WorkOperationState,
    native_thread_id: str | None,
    native_turn_id: str | None,
    native_connection_id: str | None,
    error_code: str | None,
    error_message: str | None,
    captured_model: str | None = None,
    captured_reasoning_effort: str | None = None,
) -> None:
    payload_json = (
        json.dumps(
            current.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if state in {WorkOperationState.ACCEPTED, WorkOperationState.EXECUTING}
        else None
    )
    updated = connection.execute(
        """
        UPDATE work_operations
        SET payload_json = ?, state = ?, captured_model = ?,
            captured_reasoning_effort = ?, native_thread_id = ?,
            native_turn_id = ?, native_connection_id = ?, error_code = ?,
            error_message = ?, updated_at_ms = ?
        WHERE operation_id = ?
        """,
        (
            payload_json,
            state.value,
            captured_model if captured_model is not None else current.captured_model,
            (
                captured_reasoning_effort
                if captured_reasoning_effort is not None
                else current.captured_reasoning_effort
            ),
            native_thread_id,
            native_turn_id,
            native_connection_id,
            error_code,
            error_message,
            _now_ms(),
            current.operation_id,
        ),
    ).rowcount
    if updated != 1:
        raise WorkUnavailableError("Could not update Work operation.")


def _expect_revision(session: WorkSessionRecord, expected: int) -> None:
    if session.revision != expected:
        raise WorkConflictError("Work session changed in another tab. Reload it.")


def _row_payload(row: sqlite3.Row) -> JsonObject | None:
    text = _row_optional_str(row, "payload_json")
    if text is None:
        return None
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkUnavailableError("Work storage contains invalid JSON.") from exc
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise WorkUnavailableError("Work storage contains an invalid payload.")
    return decoded


def _row_str(row: sqlite3.Row, key: str) -> str:
    value: object = row[key]
    if not isinstance(value, str):
        raise WorkUnavailableError("Work storage contains an invalid text value.")
    return value


def _row_optional_str(row: sqlite3.Row, key: str) -> str | None:
    value: object = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkUnavailableError("Work storage contains an invalid text value.")
    return value


def _row_int(row: sqlite3.Row, key: str) -> int:
    value: object = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkUnavailableError("Work storage contains an invalid integer value.")
    return value


def _row_optional_int(row: sqlite3.Row, key: str) -> int | None:
    value: object = row[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkUnavailableError("Work storage contains an invalid integer value.")
    return value


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _owner_only(path: Path, mode: int) -> None:
    if os.name != "nt":
        path.chmod(mode)

"""SQLite persistence owned exclusively by local Work Sessions."""

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

from .work_migrations import MIGRATIONS

T = TypeVar("T")

_BUSY_TIMEOUT_MS = 5_000


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
            await anyio.to_thread.run_sync(self._run_sync, self._migrate_and_repair)
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

    async def create_session(self, record: WorkSessionRecord) -> WorkSessionRecord:
        def operation(connection: sqlite3.Connection) -> WorkSessionRecord:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO work_sessions (
                        thread_id, cwd, cwd_key, model, reasoning_effort,
                        collaboration_mode, permission_profile, revision,
                        registered_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.thread_id,
                        record.cwd,
                        record.cwd_key,
                        record.settings.model,
                        record.settings.reasoning_effort,
                        record.settings.collaboration_mode,
                        record.settings.permission_profile,
                        record.revision,
                        record.registered_at_ms,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise WorkConflictError("Work session already exists.") from exc
            connection.commit()
            return _get_session(connection, record.thread_id)

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
            connection.execute(
                """
                UPDATE work_sessions
                SET model = ?, reasoning_effort = ?, collaboration_mode = ?,
                    permission_profile = ?, revision = revision + 1
                WHERE thread_id = ?
                """,
                (
                    settings.model,
                    settings.reasoning_effort,
                    settings.collaboration_mode,
                    settings.permission_profile,
                    thread_id,
                ),
            )
            connection.commit()
            return _get_session(connection, thread_id)

        return await self._run(operation)

    async def bump_revision(
        self, thread_id: str, *, expected_revision: int
    ) -> WorkSessionRecord:
        def operation(connection: sqlite3.Connection) -> WorkSessionRecord:
            connection.execute("BEGIN IMMEDIATE")
            current = _get_session(connection, thread_id)
            _expect_revision(current, expected_revision)
            connection.execute(
                "UPDATE work_sessions SET revision = revision + 1 WHERE thread_id = ?",
                (thread_id,),
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

    async def reserve_operation(
        self,
        *,
        operation_id: str,
        kind: WorkOperationKind,
        session_id: str | None,
        intent_digest: str,
    ) -> tuple[WorkOperation, bool]:
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
                    or existing.intent_digest != intent_digest
                ):
                    connection.rollback()
                    raise WorkConflictError(
                        "Operation ID was already used for different Work input."
                    )
                connection.commit()
                return existing, False
            now = _now_ms()
            connection.execute(
                """
                INSERT INTO work_operations (
                    operation_id, kind, session_id, intent_digest, state,
                    result_thread_id, result_turn_id, error_code, error_message,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, 'reserved', NULL, NULL, NULL, NULL, ?, ?)
                """,
                (operation_id, kind.value, session_id, intent_digest, now, now),
            )
            connection.commit()
            created = connection.execute(
                "SELECT * FROM work_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if created is None:
                raise WorkUnavailableError("Could not reserve Work operation.")
            return _operation_from_row(created), True

        return await self._run(operation)

    async def update_operation(
        self,
        operation_id: str,
        *,
        state: WorkOperationState,
        result_thread_id: str | None = None,
        result_turn_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkOperation:
        def operation(connection: sqlite3.Connection) -> WorkOperation:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise WorkNotFoundError("Work operation was not found.")
            existing = _operation_from_row(row)
            if existing.state in {
                WorkOperationState.COMPLETED,
                WorkOperationState.INTERRUPTED,
                WorkOperationState.FAILED,
            }:
                connection.commit()
                return existing
            updated = connection.execute(
                """
                UPDATE work_operations
                SET state = ?, result_thread_id = ?, result_turn_id = ?,
                    error_code = ?, error_message = ?, updated_at_ms = ?
                WHERE operation_id = ?
                """,
                (
                    state.value,
                    result_thread_id,
                    result_turn_id,
                    error_code,
                    error_message,
                    _now_ms(),
                    operation_id,
                ),
            ).rowcount
            if updated != 1:
                connection.rollback()
                raise WorkUnavailableError("Could not update Work operation.")
            connection.commit()
            row = connection.execute(
                "SELECT * FROM work_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise WorkUnavailableError("Could not load Work operation.")
            return _operation_from_row(row)

        return await self._run(operation)

    async def prune_deleted_session_operations(
        self, thread_id: str, *, keep_operation_id: str
    ) -> None:
        await self._run(
            lambda connection: connection.execute(
                """
                DELETE FROM work_operations
                WHERE (session_id = ? OR result_thread_id = ?)
                    AND operation_id != ?
                """,
                (thread_id, thread_id, keep_operation_id),
            )
        )

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

    def _migrate_and_repair(self, connection: sqlite3.Connection) -> None:
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

        now = _now_ms()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE work_operations
            SET state = 'interrupted', error_code = 'server_restart',
                error_message = 'FCC restarted before the operation completed.',
                updated_at_ms = ?
            WHERE state IN ('reserved', 'submitted')
            """,
            (now,),
        )
        connection.commit()


def _validate_migrations() -> None:
    expected = tuple(range(1, len(MIGRATIONS) + 1))
    actual = tuple(migration.version for migration in MIGRATIONS)
    if actual != expected:
        raise RuntimeError("Work migrations must be ordered, unique, and contiguous.")


def _get_session(connection: sqlite3.Connection, thread_id: str) -> WorkSessionRecord:
    row = connection.execute(
        "SELECT * FROM work_sessions WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    if row is None:
        raise WorkNotFoundError("Work session was not found.")
    return _session_from_row(row)


def _session_from_row(row: sqlite3.Row) -> WorkSessionRecord:
    return WorkSessionRecord(
        thread_id=_row_str(row, "thread_id"),
        cwd=_row_str(row, "cwd"),
        cwd_key=_row_str(row, "cwd_key"),
        settings=WorkSessionSettings(
            model=_row_optional_str(row, "model"),
            reasoning_effort=_row_optional_str(row, "reasoning_effort"),
            collaboration_mode=_row_optional_str(row, "collaboration_mode"),
            permission_profile=_row_optional_str(row, "permission_profile"),
        ),
        revision=_row_int(row, "revision"),
        registered_at_ms=_row_int(row, "registered_at_ms"),
    )


def _operation_from_row(row: sqlite3.Row) -> WorkOperation:
    return WorkOperation(
        operation_id=_row_str(row, "operation_id"),
        kind=WorkOperationKind(_row_str(row, "kind")),
        session_id=_row_optional_str(row, "session_id"),
        intent_digest=_row_str(row, "intent_digest"),
        state=WorkOperationState(_row_str(row, "state")),
        result_thread_id=_row_optional_str(row, "result_thread_id"),
        result_turn_id=_row_optional_str(row, "result_turn_id"),
        error_code=_row_optional_str(row, "error_code"),
        error_message=_row_optional_str(row, "error_message"),
        created_at_ms=_row_int(row, "created_at_ms"),
        updated_at_ms=_row_int(row, "updated_at_ms"),
    )


def _expect_revision(session: WorkSessionRecord, expected: int) -> None:
    if session.revision != expected:
        raise WorkConflictError("Work session changed in another tab. Reload it.")


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


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _owner_only(path: Path, mode: int) -> None:
    if os.name != "nt":
        path.chmod(mode)

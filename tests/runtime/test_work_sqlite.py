import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from free_claude_code.application.work import (
    WorkConflictError,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkUnavailableError,
)
from free_claude_code.runtime.work_sqlite import SQLiteWorkStore


def _id() -> str:
    return str(uuid.uuid4())


def _store(tmp_path: Path) -> SQLiteWorkStore:
    return SQLiteWorkStore(tmp_path / "work.db", tmp_path / "work.lock")


def _record(
    thread_id: str, cwd: Path, *, registered_at_ms: int = 1
) -> WorkSessionRecord:
    resolved = str(cwd.resolve())
    return WorkSessionRecord(
        thread_id=thread_id,
        cwd=resolved,
        cwd_key=os.path.normcase(resolved),
        settings=WorkSessionSettings(
            model="model-1",
            reasoning_effort="medium",
            collaboration_mode="plan",
            permission_profile=":workspace",
        ),
        revision=1,
        registered_at_ms=registered_at_ms,
    )


@pytest.mark.asyncio
async def test_store_migrates_and_persists_sessions_with_optimistic_revisions(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    try:
        first = await store.create_session(
            _record("thread-1", tmp_path, registered_at_ms=1)
        )
        nested = tmp_path / "nested"
        nested.mkdir()
        second = await store.create_session(
            _record("thread-2", nested, registered_at_ms=2)
        )
        updated = await store.update_settings(
            first.thread_id,
            expected_revision=first.revision,
            settings=WorkSessionSettings(
                model="model-2",
                reasoning_effort="high",
                collaboration_mode=None,
                permission_profile=None,
            ),
        )

        assert updated.revision == 2
        assert updated.settings.model == "model-2"
        with pytest.raises(WorkConflictError, match="another tab"):
            await store.bump_revision(first.thread_id, expected_revision=1)
        assert [record.thread_id for record in await store.list_sessions()] == [
            second.thread_id,
            first.thread_id,
        ]
        assert await store.recent_projects(limit=2) == (
            second.cwd,
            first.cwd,
        )
        with closing(sqlite3.connect(tmp_path / "work.db")) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_operation_journal_is_idempotent_and_rejects_reused_input(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    operation_id = _id()
    try:
        operation, created = await store.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            intent_digest="same-intent",
        )
        repeated, repeated_created = await store.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            intent_digest="same-intent",
        )

        assert created is True
        assert repeated_created is False
        assert repeated == operation
        with pytest.raises(WorkConflictError, match="different Work input"):
            await store.reserve_operation(
                operation_id=operation_id,
                kind=WorkOperationKind.CREATE,
                session_id=None,
                intent_digest="different-intent",
            )

        completed = await store.update_operation(
            operation_id,
            state=WorkOperationState.COMPLETED,
            result_thread_id="thread-1",
        )
        late = await store.update_operation(
            operation_id,
            state=WorkOperationState.SUBMITTED,
            result_thread_id="thread-1",
            result_turn_id="late-turn",
        )
        assert late == completed
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_startup_marks_ambiguous_operations_interrupted_without_replay(
    tmp_path: Path,
) -> None:
    operation_id = _id()
    store = _store(tmp_path)
    await store.start()
    await store.reserve_operation(
        operation_id=operation_id,
        kind=WorkOperationKind.SEND,
        session_id="thread-1",
        intent_digest="send-intent",
    )
    await store.update_operation(
        operation_id,
        state=WorkOperationState.SUBMITTED,
        result_thread_id="thread-1",
        result_turn_id="turn-1",
    )
    await store.close()

    reopened = _store(tmp_path)
    await reopened.start()
    try:
        repaired, created = await reopened.reserve_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.SEND,
            session_id="thread-1",
            intent_digest="send-intent",
        )
        assert created is False
        assert repaired.state is WorkOperationState.INTERRUPTED
        assert repaired.error_code == "server_restart"
        assert repaired.result_thread_id == "thread-1"
        assert repaired.result_turn_id == "turn-1"
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_store_lock_prevents_two_local_owners(tmp_path: Path) -> None:
    first = _store(tmp_path)
    second = _store(tmp_path)
    await first.start()
    try:
        with pytest.raises(WorkUnavailableError, match="another FCC server"):
            await second.start()
    finally:
        await first.close()


@pytest.mark.asyncio
async def test_future_schema_is_rejected_without_rewriting_it(tmp_path: Path) -> None:
    database = tmp_path / "work.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA user_version = 99")
    store = _store(tmp_path)

    with pytest.raises(WorkUnavailableError, match="newer FCC version"):
        await store.start()

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (99,)

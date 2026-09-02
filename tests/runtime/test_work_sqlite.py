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
        settings=WorkSessionSettings(model="model-1", reasoning_effort="medium"),
        revision=1,
        registered_at_ms=registered_at_ms,
    )


async def _register(
    store: SQLiteWorkStore, record: WorkSessionRecord
) -> WorkSessionRecord:
    operation_id = _id()
    await store.admit_operation(
        operation_id=operation_id,
        kind=WorkOperationKind.CREATE,
        session_id=None,
        interaction_id=None,
        intent_digest="a" * 64,
        payload={"cwd": record.cwd, "cwd_key": record.cwd_key},
    )
    assert await store.claim_operation(operation_id) is not None
    await store.record_operation_evidence(
        operation_id,
        native_thread_id=record.thread_id,
        captured_model=record.settings.model,
        captured_reasoning_effort=record.settings.reasoning_effort,
    )
    _, persisted = await store.create_session_from_operation(operation_id, record)
    return persisted


@pytest.mark.asyncio
async def test_store_migrates_and_persists_sessions_with_optimistic_revisions(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    try:
        first = await _register(store, _record("thread-1", tmp_path))
        nested = tmp_path / "nested"
        nested.mkdir()
        second = await _register(store, _record("thread-2", nested, registered_at_ms=2))
        updated = await store.update_settings(
            first.thread_id,
            expected_revision=first.revision,
            settings=WorkSessionSettings(model="model-2", reasoning_effort="high"),
        )

        assert updated.revision == 2
        assert updated.settings == WorkSessionSettings("model-2", "high")
        with pytest.raises(WorkConflictError, match="another tab"):
            await store.update_settings(
                first.thread_id,
                expected_revision=1,
                settings=WorkSessionSettings("model-3", None),
            )
        assert [record.thread_id for record in await store.list_sessions()] == [
            second.thread_id,
            first.thread_id,
        ]
        assert await store.recent_projects(limit=2) == (second.cwd, first.cwd)
        with closing(sqlite3.connect(tmp_path / "work.db")) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_operation_admission_is_idempotent_and_rejects_changed_input(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    operation_id = _id()
    try:
        operation, created = await store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={"cwd": str(tmp_path), "cwd_key": str(tmp_path)},
        )
        repeated, repeated_created = await store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={"cwd": "ignored on replay", "cwd_key": "ignored"},
        )

        assert created is True
        assert repeated_created is False
        assert repeated == operation
        with pytest.raises(WorkConflictError, match="different Work input"):
            await store.admit_operation(
                operation_id=operation_id,
                kind=WorkOperationKind.CREATE,
                session_id=None,
                interaction_id=None,
                intent_digest="b" * 64,
                payload={"cwd": str(tmp_path), "cwd_key": str(tmp_path)},
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_terminal_transition_erases_payload_and_cannot_regress(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    operation_id = _id()
    try:
        await store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={"cwd": "secret", "cwd_key": "secret"},
        )
        executing = await store.claim_operation(operation_id)
        assert executing is not None
        assert executing.payload is not None
        failed = await store.transition_operation(
            operation_id,
            expected_states=(WorkOperationState.EXECUTING,),
            state=WorkOperationState.FAILED,
            error_code="rejected",
            error_message="Codex rejected the request.",
        )
        late = await store.transition_operation(
            operation_id,
            expected_states=(WorkOperationState.EXECUTING,),
            state=WorkOperationState.UNKNOWN,
        )

        assert failed.payload is None
        assert late == failed
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_same_effect_stop_and_delete_are_coalesced_across_operation_ids(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    try:
        record = await _register(store, _record("thread-1", tmp_path))
        first, created = await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.STOP,
            session_id=record.thread_id,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={},
        )
        coalesced, second_created = await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.STOP,
            session_id=record.thread_id,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={},
        )
        delete, delete_created = await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.DELETE,
            session_id=record.thread_id,
            interaction_id=None,
            intent_digest="b" * 64,
            payload={},
        )
        repeated_delete, repeated_delete_created = await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.DELETE,
            session_id=record.thread_id,
            interaction_id=None,
            intent_digest="b" * 64,
            payload={},
        )

        assert created is True
        assert second_created is False
        assert coalesced.operation_id == first.operation_id
        assert delete_created is True
        assert repeated_delete_created is False
        assert repeated_delete.operation_id == delete.operation_id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_active_send_and_interaction_response_have_one_owner(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    try:
        record = await _register(store, _record("thread-1", tmp_path))
        await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.SEND,
            session_id=record.thread_id,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={"text": "hello"},
            expected_revision=1,
        )
        with pytest.raises(WorkConflictError, match="active or uncertain"):
            await store.admit_operation(
                operation_id=_id(),
                kind=WorkOperationKind.SEND,
                session_id=record.thread_id,
                interaction_id=None,
                intent_digest="b" * 64,
                payload={"text": "again"},
                expected_revision=1,
            )

        await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.RESPOND,
            session_id=record.thread_id,
            interaction_id="interaction-1",
            intent_digest="c" * 64,
            payload={"kind": "user_input", "result": {}},
        )
        with pytest.raises(WorkConflictError, match="already answered"):
            await store.admit_operation(
                operation_id=_id(),
                kind=WorkOperationKind.RESPOND,
                session_id=record.thread_id,
                interaction_id="interaction-1",
                intent_digest="d" * 64,
                payload={"kind": "user_input", "result": {}},
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_active_stop_blocks_new_send_and_settings_change(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.start()
    try:
        record = await _register(store, _record("thread-1", tmp_path))
        await store.admit_operation(
            operation_id=_id(),
            kind=WorkOperationKind.STOP,
            session_id=record.thread_id,
            interaction_id=None,
            intent_digest="a" * 64,
            payload={},
        )

        with pytest.raises(WorkConflictError, match="Stop operation"):
            await store.admit_operation(
                operation_id=_id(),
                kind=WorkOperationKind.SEND,
                session_id=record.thread_id,
                interaction_id=None,
                intent_digest="b" * 64,
                payload={"text": "too early"},
                expected_revision=record.revision,
            )
        with pytest.raises(WorkConflictError, match="current Work operation"):
            await store.update_settings(
                record.thread_id,
                expected_revision=record.revision,
                settings=WorkSessionSettings("model-2", "high"),
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_restart_preserves_accepted_and_executing_for_coordinator_recovery(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.start()
    accepted_id = _id()
    executing_id = _id()
    for operation_id, digest in ((accepted_id, "a" * 64), (executing_id, "b" * 64)):
        await store.admit_operation(
            operation_id=operation_id,
            kind=WorkOperationKind.CREATE,
            session_id=None,
            interaction_id=None,
            intent_digest=digest,
            payload={"cwd": str(tmp_path), "cwd_key": str(tmp_path)},
        )
    assert await store.claim_operation(executing_id) is not None
    await store.close()

    reopened = _store(tmp_path)
    await reopened.start()
    try:
        assert (
            await reopened.get_operation(accepted_id)
        ).state is WorkOperationState.ACCEPTED
        executing = await reopened.get_operation(executing_id)
        assert executing.state is WorkOperationState.EXECUTING
        assert await reopened.claim_operation(executing_id) is None
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

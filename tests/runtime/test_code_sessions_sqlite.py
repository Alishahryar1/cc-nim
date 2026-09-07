import asyncio
import sqlite3
import threading
import uuid
from contextlib import closing

import pytest
import pytest_asyncio

from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeItem,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
)
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore


@pytest.mark.asyncio
async def test_cancelled_initialization_drains_thread_before_releasing_owner_lock(
    tmp_path,
):
    entered = threading.Event()
    release = threading.Event()
    initialized = threading.Event()

    class GatedStore(SQLiteCodeStore):
        def _initialize(self):
            entered.set()
            assert release.wait(5)
            super()._initialize()
            initialized.set()

    store = GatedStore(tmp_path / "code.db", tmp_path / "code.lock")
    starting = asyncio.create_task(store.start())
    await asyncio.to_thread(entered.wait, 3)
    starting.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await asyncio.to_thread(initialized.wait, 3)
    second = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    try:
        await second.start()
        session = await second.create(
            CodeSession(id=str(uuid.uuid4()), cwd=str(tmp_path), model="provider/model")
        )
        assert (await second.get_session(session.id)).id == session.id
    finally:
        await store.close()
        await second.close()


@pytest.mark.asyncio
async def test_store_has_one_process_owner_and_can_reopen_after_close(tmp_path):
    first = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    second = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    await first.start()
    try:
        with pytest.raises(CodeUnavailableError, match="another FCC"):
            await second.start()
    finally:
        await first.close()
    await second.start()
    await second.close()


@pytest.mark.asyncio
async def test_reused_item_id_cannot_overwrite_another_session(tmp_path):
    store = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    await store.start()
    try:
        first, second = [
            await store.create(
                CodeSession(
                    id=str(uuid.uuid4()), cwd=str(tmp_path), model="provider/model"
                )
            )
            for _ in range(2)
        ]
        shared_id = str(uuid.uuid4())
        first, first_run = await _admit(store, first, text="first")
        original = CodeItem(
            id=shared_id,
            session_id=first.id,
            sequence=2,
            run_id=first_run.id,
            kind="tool",
            text="original output",
            complete=True,
        )
        await store.save_progress(first, first.revision, items=(original,))
        await _admit(store, second, run_id=shared_id, text="second")
        assert (await store.items(first.id, None, None))[-1] == original
        assert (await store.items(second.id, None, None))[0].text == "second"
    finally:
        await store.close()


async def _admit(store, session, *, run_id=None, text="message", sequence=1):
    run_id = run_id or str(uuid.uuid4())
    run = CodeRun(id=run_id, session_id=session.id, text=text, model=session.model)
    item = CodeItem(
        id=run_id,
        session_id=session.id,
        run_id=run_id,
        sequence=sequence,
        kind="user",
        text=text,
        complete=True,
    )
    return await store.admit_run(
        session.model_copy(update={"revision": session.revision + 1}),
        run,
        item,
        session.revision,
    )


@pytest_asyncio.fixture
async def store(tmp_path):
    result = SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock")
    await result.start()
    yield result
    await result.close()


async def _session(store):
    return await store.create(
        CodeSession(id=str(uuid.uuid4()), cwd="/work", model="provider/model")
    )


@pytest.mark.asyncio
async def test_admission_is_atomic_and_idempotent(store):
    session = await _session(store)
    results = await asyncio.gather(
        _admit(store, session), _admit(store, session), return_exceptions=True
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    session, run = next(result for result in results if isinstance(result, tuple))
    assert (
        len(await store.runs(session.id))
        == len(await store.items(session.id, None, None))
        == 1
    )
    assert await _admit(store, session, run_id=run.id) == (session, run)
    await store.save_progress(
        session, session.revision, run=run.model_copy(update={"status": "completed"})
    )
    collision = CodeItem(
        id=str(uuid.uuid4()),
        session_id=session.id,
        run_id=run.id,
        sequence=2,
        kind="tool",
        text="keep me",
        complete=True,
    )
    await store.save_progress(session, session.revision, items=(collision,))
    with pytest.raises(CodeConflictError):
        await _admit(store, session, run_id=collision.id, sequence=3)
    assert (await store.get_session(session.id)).revision == session.revision
    assert len(await store.runs(session.id)) == 1
    assert (await store.items(session.id, None, None))[-1] == collision


@pytest.mark.asyncio
async def test_schema_rejects_second_active_run_and_cross_session_item_link(
    store, tmp_path
):
    session, run = await _admit(store, await _session(store))
    other = await _session(store)
    with closing(sqlite3.connect(tmp_path / "code.db")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(code_runs)")]
        expressions = [
            "'another'"
            if column == "id"
            else "ordinal + 1"
            if column == "ordinal"
            else column
            for column in columns
        ]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO code_runs SELECT {','.join(expressions)} FROM code_runs"
            )
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(code_items)")
        ]
        expressions = ["?" if column == "session_id" else column for column in columns]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO code_items SELECT {','.join(expressions)} FROM code_items",
                (other.id,),
            )
    assert await store.get_run(session.id, run.id) == run


@pytest.mark.asyncio
async def test_prompt_claim_and_settings_are_guarded_in_storage(store):
    session = await _session(store)
    prompts = tuple(
        CodePrompt(
            id=str(uuid.uuid4()),
            session_id=session.id,
            generation="g",
            request_id=request_id,
            kind="question",
            form={},
            raw={},
        )
        for request_id in (1, "1")
    )
    await store.save_progress(session, session.revision, prompts=prompts)
    results = await asyncio.gather(
        *(
            store.claim_prompt(session.id, prompts[0].id, str(uuid.uuid4()), "g")
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CodeConflictError) for result in results) == 1
    claimed = next(result for result in results if isinstance(result, CodePrompt))
    assert (
        await store.claim_prompt(session.id, claimed.id, claimed.response_id, "g")
        == claimed
    )
    with pytest.raises(CodeConflictError):
        await store.claim_prompt(session.id, prompts[1].id, claimed.response_id, "g")
    settings = session.model_copy(
        update={"model": "provider/other", "revision": session.revision + 1}
    )
    with pytest.raises(CodeConflictError):
        await store.update_settings(settings, session.revision)
    await store.save_progress(
        session,
        session.revision,
        prompts=(
            claimed.model_copy(update={"status": "resolved"}),
            prompts[1].model_copy(update={"status": "expired"}),
        ),
    )
    assert (
        await store.update_settings(settings, session.revision)
    ).model == "provider/other"
    with pytest.raises(CodeConflictError):
        await store.update_settings(
            session.model_copy(update={"title": "stale"}), session.revision
        )


@pytest.mark.asyncio
async def test_recovered_old_output_pages_with_its_original_run_outcome(store):
    session, old = await _admit(store, await _session(store))
    old = old.model_copy(update={"status": "failed", "error": "first failed"})
    await store.save_progress(session, session.revision, run=old)
    session, newer = await _admit(store, session, sequence=2)
    tail = CodeItem(
        id=str(uuid.uuid4()),
        session_id=session.id,
        run_id=old.id,
        sequence=3,
        kind="assistant",
        text="recovered",
        complete=True,
    )
    await store.save_progress(session, session.revision, items=(tail,))
    page = await store.item_page(session.id, None, 2)
    assert [item.run_id for item in page.items] == [old.id, newer.id]
    assert page.runs[0].error == "first failed"
    assert page.next_before == (old.ordinal, 3)
    older = await store.item_page(session.id, page.next_before, 2)
    assert [item.sequence for item in older.items] == [1]

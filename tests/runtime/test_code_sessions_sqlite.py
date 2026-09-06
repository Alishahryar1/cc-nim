import asyncio
import threading
import uuid

import pytest

from free_claude_code.application.code_sessions.models import (
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
            CodeSession(id=str(uuid.uuid4()), cwd=str(tmp_path))
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

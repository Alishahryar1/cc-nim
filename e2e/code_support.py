"""Explicit barriers for Code browser tests, on the application's own loop."""

import asyncio
from collections.abc import Coroutine
from unittest.mock import patch

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.code_sessions_support import FakeHarness


class CodeControl:
    def __init__(self, directory):
        self.harness = FakeHarness()
        self.service = CodeService(
            SQLiteCodeStore(directory / "code.db", directory / "code.lock"),
            self.harness,
        )
        self.loop: asyncio.AbstractEventLoop | None = None

    def run[T](self, work: Coroutine[object, object, T]) -> T:
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(work, self.loop).result(timeout=10)

    def connection(self):
        self.run(self.harness.started.wait())
        return self.harness.connections[-1]

    async def hold_send(self):
        self.admission = asyncio.Event()
        original = self.service._store.save

        async def save(session, **kwargs):
            run = kwargs.get("run")
            if run is not None and run.status == "preparing":
                await self.admission.wait()
            await original(session, **kwargs)

        self.admission_patch = patch.object(self.service._store, "save", save)
        self.admission_patch.start()

    async def release_send(self):
        self.admission.set()
        self.admission_patch.stop()

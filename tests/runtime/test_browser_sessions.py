import asyncio
import json
import logging
import signal
import sys
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from free_claude_code.application.browser_sessions import (
    BrowserSessionConflictError,
    BrowserSessionHarness,
    BrowserSessionState,
    BrowserSessionUnavailableError,
    HarnessAvailability,
)
from free_claude_code.runtime.browser_sessions.drivers import (
    ClaudeDriver,
    CodexDriver,
    HarnessDriver,
    HarnessDriverRegistry,
    PiDriver,
    _create_codex_thread,
    _delete_codex_thread,
    terminal_environment,
)
from free_claude_code.runtime.browser_sessions.manager import (
    OUTPUT_RING_LIMIT_BYTES,
    BrowserSessionManager,
)
from free_claude_code.runtime.browser_sessions.pty import (
    NativeTerminalProcess,
    TerminalProcessFactory,
)
from free_claude_code.runtime.browser_sessions.store import (
    BrowserSessionStore,
    SessionCatalog,
    SessionRecord,
    SessionStoreError,
)


class FakeDriver(HarnessDriver):
    def __init__(self, harness: BrowserSessionHarness) -> None:
        self.harness = harness
        self.wrapper_name = f"fcc-{harness.value}"
        self.client_name = harness.value
        self.commands: list[tuple[str, bool]] = []
        self.discarded: list[tuple[str, Path]] = []

    def availability(self) -> HarnessAvailability:
        return HarnessAvailability(self.harness, True)

    async def create_native_id(self, project_path: Path) -> str:
        return f"native-{self.harness.value}"

    async def discard_native_id(self, native_id: str, project_path: Path) -> None:
        self.discarded.append((native_id, project_path))

    def command(self, native_id: str, *, started_once: bool) -> list[str]:
        self.commands.append((native_id, started_once))
        return [f"fake-{self.harness.value}", native_id]


class FakeTerminalProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.writes: list[str] = []
        self.resizes: list[tuple[int, int]] = []
        self.terminated = False
        self.closed = False
        self.exit_code = 0
        self._chunks: Queue[str | Exception | None] = Queue()

    def read(self, size: int = 4096) -> str:
        chunk = self._chunks.get(timeout=5)
        if chunk is None:
            raise EOFError
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def write(self, data: str) -> None:
        self.writes.append(data)

    def resize(self, columns: int, rows: int) -> None:
        self.resizes.append((columns, rows))

    def wait(self) -> int:
        return self.exit_code

    def terminate_tree(self) -> None:
        self.terminated = True
        self._chunks.put(None)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._chunks.put(None)

    def output(self, value: str) -> None:
        self._chunks.put(value)

    def fail_read(self, error: Exception) -> None:
        self._chunks.put(error)

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self._chunks.put(None)


class FakeProcessFactory:
    def __init__(self) -> None:
        self.processes: list[FakeTerminalProcess] = []
        self.spawns: list[tuple[list[str], str, int, int]] = []

    def spawn(self, command, *, cwd, env, columns, rows):
        process = FakeTerminalProcess(1000 + len(self.processes))
        self.processes.append(process)
        self.spawns.append((list(command), cwd, columns, rows))
        return process


class BlockingProcessFactory(FakeProcessFactory):
    def __init__(self) -> None:
        super().__init__()
        self.spawn_started = threading.Event()
        self.release_spawn = threading.Event()

    def spawn(self, command, *, cwd, env, columns, rows):
        self.spawn_started.set()
        if not self.release_spawn.wait(timeout=5):
            raise TimeoutError("test did not release terminal spawn")
        return super().spawn(
            command,
            cwd=cwd,
            env=env,
            columns=columns,
            rows=rows,
        )


class BlockingDriver(FakeDriver):
    def __init__(self, harness: BrowserSessionHarness) -> None:
        super().__init__(harness)
        self.creation_started = asyncio.Event()
        self.release_creation = asyncio.Event()

    async def create_native_id(self, project_path: Path) -> str:
        self.creation_started.set()
        await self.release_creation.wait()
        return await super().create_native_id(project_path)


def fake_drivers():
    drivers = [FakeDriver(harness) for harness in BrowserSessionHarness]
    return HarnessDriverRegistry(drivers), {
        driver.harness: driver for driver in drivers
    }


def test_store_round_trips_ordered_metadata_and_rejects_corruption(tmp_path):
    path = tmp_path / "sessions.json"
    store = BrowserSessionStore(path)
    catalog = SessionCatalog(
        sessions=[
            SessionRecord(
                session_id="ses_one",
                path=str(tmp_path),
                name="Review",
                harness=BrowserSessionHarness.CODEX,
                native_id="thread-id",
                started_once=True,
            )
        ]
    )

    store.save(catalog)

    assert store.load() == catalog
    assert not list(tmp_path.glob("*.tmp"))

    path.write_text('{"version": 999, "sessions": []}', encoding="utf-8")
    with pytest.raises(SessionStoreError, match="original file was preserved"):
        store.load()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 999


@pytest.mark.asyncio
async def test_manager_detaches_without_stopping_and_resumes_native_session(tmp_path):
    registry, drivers = fake_drivers()
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=factory,
    )
    session = await manager.create_session(
        str(tmp_path),
        BrowserSessionHarness.CLAUDE,
        "First task",
    )
    process = factory.processes[0]

    assert session.state == BrowserSessionState.RUNNING
    assert drivers[BrowserSessionHarness.CLAUDE].commands == [("native-claude", False)]

    attachment = await manager.attach_terminal(session.session_id)
    ready = await asyncio.wait_for(attachment.receive(), timeout=1)
    assert ready.kind == "ready"
    process.output("hello from terminal")
    output = await asyncio.wait_for(attachment.receive(), timeout=1)
    assert output.data == b"hello from terminal"
    await attachment.write("answer\r")
    await attachment.resize(100, 30)
    await attachment.close()

    assert process.writes == ["answer\r"]
    assert process.resizes == [(100, 30)]
    assert (await manager.snapshot()).sessions[0].state == BrowserSessionState.RUNNING

    stopped = await manager.stop_session(session.session_id)
    assert stopped.state == BrowserSessionState.STOPPED
    assert process.terminated

    restarted = await manager.start_session(session.session_id)
    assert restarted.state == BrowserSessionState.RUNNING
    assert drivers[BrowserSessionHarness.CLAUDE].commands[-1] == (
        "native-claude",
        True,
    )

    await manager.close()
    assert factory.processes[1].terminated

    reloaded_factory = FakeProcessFactory()
    reloaded = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=reloaded_factory,
    )
    snapshot = await reloaded.snapshot()
    assert snapshot.sessions[0].state == BrowserSessionState.STOPPED
    await reloaded.start_session(session.session_id)
    assert drivers[BrowserSessionHarness.CLAUDE].commands[-1] == (
        "native-claude",
        True,
    )
    await reloaded.close()


@pytest.mark.asyncio
async def test_manager_replaces_terminal_controller_and_validates_names(tmp_path):
    registry, _ = fake_drivers()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=FakeProcessFactory(),
    )
    session = await manager.create_session(
        str(tmp_path),
        BrowserSessionHarness.PI,
        "Unique",
    )
    first = await manager.attach_terminal(session.session_id)
    await first.receive()
    second = await manager.attach_terminal(session.session_id)

    assert (await first.receive()).kind == "replaced"
    assert (await second.receive()).kind == "ready"

    with pytest.raises(BrowserSessionConflictError, match="unique"):
        await manager.create_session(
            str(tmp_path),
            BrowserSessionHarness.CODEX,
            "unique",
        )

    other = tmp_path / "other"
    other.mkdir()
    same_name_elsewhere = await manager.create_session(
        str(other),
        BrowserSessionHarness.CODEX,
        "unique",
    )
    assert same_name_elsewhere.project_name == "other"
    assert same_name_elsewhere.path == str(other.resolve())
    await manager.close()


@pytest.mark.asyncio
async def test_manager_defaults_names_and_returns_newest_sessions_first(tmp_path):
    registry, drivers = fake_drivers()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=FakeProcessFactory(),
    )

    first = await manager.create_session(str(tmp_path), BrowserSessionHarness.PI)
    second = await manager.create_session(str(tmp_path), BrowserSessionHarness.CODEX)

    assert first.name == "New session"
    assert second.name == "New session 2"
    assert [session.session_id for session in (await manager.snapshot()).sessions] == [
        second.session_id,
        first.session_id,
    ]
    await manager.delete_session(first.session_id)
    assert drivers[BrowserSessionHarness.PI].discarded == []
    await manager.close()


@pytest.mark.asyncio
async def test_metadata_commit_failure_discards_uncommitted_native_identity(
    monkeypatch, tmp_path
):
    registry, drivers = fake_drivers()
    store = BrowserSessionStore(tmp_path / "sessions.json")
    manager = BrowserSessionManager(
        store,
        drivers=registry,
        process_factory=FakeProcessFactory(),
    )

    def fail_save(catalog):
        raise SessionStoreError("Browser Sessions metadata could not be saved.")

    monkeypatch.setattr(store, "save", fail_save)

    with pytest.raises(BrowserSessionUnavailableError, match="could not be saved"):
        await manager.create_session(
            str(tmp_path),
            BrowserSessionHarness.CODEX,
            "Rollback",
        )

    driver = drivers[BrowserSessionHarness.CODEX]
    assert driver.discarded == [("native-codex", tmp_path.resolve())]
    assert (await manager.snapshot()).sessions == ()
    await manager.close()


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_metadata_failure(
    monkeypatch, tmp_path, caplog
):
    class FailingDiscardDriver(FakeDriver):
        async def discard_native_id(self, native_id: str, project_path: Path) -> None:
            await super().discard_native_id(native_id, project_path)
            raise RuntimeError("cleanup details must stay secondary")

    driver = FailingDiscardDriver(BrowserSessionHarness.CODEX)
    store = BrowserSessionStore(tmp_path / "sessions.json")
    manager = BrowserSessionManager(
        store,
        drivers=HarnessDriverRegistry([driver]),
        process_factory=FakeProcessFactory(),
    )

    def fail_save(_catalog):
        raise SessionStoreError("Browser Sessions metadata could not be saved.")

    monkeypatch.setattr(store, "save", fail_save)

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(BrowserSessionUnavailableError, match="could not be saved"),
    ):
        await manager.create_session(
            str(tmp_path),
            BrowserSessionHarness.CODEX,
            "Rollback",
        )

    assert driver.discarded == [("native-codex", tmp_path.resolve())]
    warning = " | ".join(record.getMessage() for record in caplog.records)
    assert "uncommitted codex browser session native-codex" in warning
    assert "cleanup details must stay secondary" not in warning
    await manager.close()


@pytest.mark.asyncio
async def test_reader_failure_terminates_the_owned_process_tree(tmp_path):
    registry, _ = fake_drivers()
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=factory,
    )
    session = await manager.create_session(
        str(tmp_path),
        BrowserSessionHarness.PI,
        "Reader failure",
    )
    attachment = await manager.attach_terminal(session.session_id)
    await attachment.receive()

    factory.processes[0].fail_read(OSError("terminal channel failed"))
    event = await asyncio.wait_for(attachment.receive(), timeout=1)

    assert event.kind == "exit"
    assert event.state == BrowserSessionState.FAILED
    assert event.detail == "Terminal ended unexpectedly (OSError)."
    assert factory.processes[0].terminated
    assert factory.processes[0].closed
    await manager.close()


@pytest.mark.asyncio
async def test_natural_nonzero_exit_is_reported_without_exposing_a_shell(tmp_path):
    registry, _ = fake_drivers()
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=factory,
    )
    session = await manager.create_session(
        str(tmp_path),
        BrowserSessionHarness.CLAUDE,
        "Natural exit",
    )
    attachment = await manager.attach_terminal(session.session_id)
    await attachment.receive()

    factory.processes[0].finish(exit_code=7)
    event = await asyncio.wait_for(attachment.receive(), timeout=1)

    assert event.kind == "exit"
    assert event.state == BrowserSessionState.FAILED
    assert event.detail == "Harness exited with code 7."
    assert factory.processes[0].closed
    persisted = (await manager.snapshot()).sessions[0]
    assert persisted.state == BrowserSessionState.FAILED
    await manager.close()


@pytest.mark.asyncio
async def test_terminal_reconnect_replays_only_the_bounded_output_tail(tmp_path):
    registry, _ = fake_drivers()
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=factory,
    )
    session = await manager.create_session(
        str(tmp_path),
        BrowserSessionHarness.PI,
        "Bounded replay",
    )
    first = await manager.attach_terminal(session.session_id)
    await first.receive()
    oversized = "prefix" + "x" * OUTPUT_RING_LIMIT_BYTES
    factory.processes[0].output(oversized)
    assert (
        await asyncio.wait_for(first.receive(), timeout=1)
    ).data == oversized.encode()
    await first.close()

    second = await manager.attach_terminal(session.session_id)
    assert (await second.receive()).kind == "ready"
    replay = await second.receive()
    assert replay.kind == "output"
    assert replay.data == oversized.encode()[-OUTPUT_RING_LIMIT_BYTES:]
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_terminal_spawn_cleans_up_the_late_process(tmp_path):
    registry, _ = fake_drivers()
    store = BrowserSessionStore(tmp_path / "sessions.json")
    store.save(
        SessionCatalog(
            sessions=[
                SessionRecord(
                    session_id="ses_one",
                    path=str(tmp_path),
                    name="Cancellation",
                    harness=BrowserSessionHarness.CLAUDE,
                    native_id="native-claude",
                )
            ]
        )
    )
    factory = BlockingProcessFactory()
    manager = BrowserSessionManager(
        store,
        drivers=registry,
        process_factory=factory,
    )

    start = asyncio.create_task(manager.start_session("ses_one"))
    assert await asyncio.to_thread(factory.spawn_started.wait, 1)
    start.cancel()
    factory.release_spawn.set()

    with pytest.raises(asyncio.CancelledError):
        await start
    assert factory.processes[0].terminated
    assert factory.processes[0].closed
    session = (await manager.snapshot()).sessions[0]
    assert session.state == BrowserSessionState.FAILED
    assert session.detail == "Session start was cancelled."
    await manager.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_identity_creation_and_rejects_late_commit(
    tmp_path,
):
    driver = BlockingDriver(BrowserSessionHarness.CODEX)
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=HarnessDriverRegistry([driver]),
        process_factory=FakeProcessFactory(),
    )
    creation = asyncio.create_task(
        manager.create_session(
            str(tmp_path),
            BrowserSessionHarness.CODEX,
            "During shutdown",
        )
    )
    await asyncio.wait_for(driver.creation_started.wait(), timeout=1)

    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    assert not closing.done()
    driver.release_creation.set()
    await closing

    with pytest.raises(BrowserSessionUnavailableError, match="shutting down"):
        await creation
    persisted = BrowserSessionStore(tmp_path / "sessions.json").load()
    assert persisted.sessions == []
    assert driver.discarded == [("native-codex", tmp_path.resolve())]


def test_harness_drivers_build_documented_native_resume_commands(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers._resolve_command",
        lambda name: f"/bin/{name}",
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.shutil.which",
        lambda name: f"/bin/{name}",
    )

    assert ClaudeDriver().command("uuid", started_once=False) == [
        "/bin/fcc-claude",
        "--session-id",
        "uuid",
    ]
    assert ClaudeDriver().command("uuid", started_once=True) == [
        "/bin/fcc-claude",
        "--resume",
        "uuid",
    ]
    assert PiDriver().command("uuid", started_once=True) == [
        "/bin/fcc-pi",
        "--session-id",
        "uuid",
    ]
    assert CodexDriver().command("thread", started_once=False) == [
        "/bin/fcc-codex",
        "resume",
        "thread",
    ]
    assert terminal_environment(
        {"TERM": "dumb", "COLORTERM": "false", "KEEP": "yes"}
    ) == {
        "TERM": "xterm-256color",
        "COLORTERM": "truecolor",
        "KEEP": "yes",
    }


def test_harness_availability_checks_both_fcc_wrapper_and_native_client(monkeypatch):
    driver = ClaudeDriver()
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers._resolve_command",
        lambda name: None,
    )

    missing_wrapper = driver.availability()

    assert missing_wrapper.available is False
    assert "Update Free Claude Code" in (missing_wrapper.message or "")

    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers._resolve_command",
        lambda name: f"/bin/{name}",
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.shutil.which",
        lambda name: None,
    )

    missing_client = driver.availability()

    assert missing_client.available is False
    assert "Install" in (missing_client.message or "")


class FakeNativePty:
    def __init__(self, *, wait_result: int | None = 0, exitstatus: int | None = None):
        self.pid = 4242
        self.wait_result = wait_result
        self.exitstatus = exitstatus
        self.writes: list[str] = []
        self.dimensions: list[tuple[int, int]] = []
        self.close_calls: list[bool] = []

    def read(self, size: int) -> str:
        return f"read-{size}"

    def write(self, data: str) -> None:
        self.writes.append(data)

    def setwinsize(self, rows: int, columns: int) -> None:
        self.dimensions.append((rows, columns))

    def wait(self) -> int | None:
        return self.wait_result

    def close(self, *, force: bool) -> None:
        self.close_calls.append(force)


def test_native_terminal_process_normalizes_io_cleanup_and_exit_status(monkeypatch):
    registered: list[int] = []
    unregistered: list[int] = []
    terminated: list[int] = []
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.register_pid",
        registered.append,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.unregister_pid",
        unregistered.append,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.kill_pid_tree_best_effort",
        terminated.append,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.IS_WINDOWS",
        True,
    )
    native = FakeNativePty(wait_result=None, exitstatus=9)
    process = NativeTerminalProcess(native)

    assert process.pid == 4242
    assert process.read(12) == "read-12"
    process.write("hello")
    process.resize(100, 30)
    assert process.wait() == 9
    process.terminate_tree()
    process.close()
    process.close()

    assert registered == [4242]
    assert native.writes == ["hello"]
    assert native.dimensions == [(30, 100)]
    assert terminated == [4242]
    assert native.close_calls == [True]
    assert unregistered == [4242]


def test_native_terminal_process_terminates_posix_process_group(monkeypatch):
    terminated: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.IS_WINDOWS",
        False,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.register_pid", lambda pid: None
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.unregister_pid", lambda pid: None
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.os.killpg",
        lambda pid, sig: terminated.append((pid, sig)),
        raising=False,
    )
    process = NativeTerminalProcess(FakeNativePty())

    process.terminate_tree()
    process.close()

    assert terminated == [(4242, signal.SIGTERM)]


@pytest.mark.parametrize(
    ("platform_name", "class_name"),
    [("nt", "PtyProcess"), ("posix", "PtyProcessUnicode")],
)
def test_terminal_factory_selects_platform_adapter(
    monkeypatch, platform_name, class_name
):
    spawned: list[tuple[list[str], dict[str, object]]] = []

    class FakePtyClass:
        @staticmethod
        def spawn(command, **kwargs):
            spawned.append((command, kwargs))
            return FakeNativePty()

    module = SimpleNamespace(**{class_name: FakePtyClass})
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.IS_WINDOWS",
        platform_name == "nt",
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.register_pid", lambda pid: None
    )
    module_name = "winpty" if platform_name == "nt" else "ptyprocess"
    monkeypatch.setitem(sys.modules, module_name, module)

    process = TerminalProcessFactory().spawn(
        ["fcc-codex", "resume", "thread"],
        cwd="project",
        env={"TERM": "xterm-256color"},
        columns=120,
        rows=32,
    )

    assert process.pid == 4242
    assert spawned == [
        (
            ["fcc-codex", "resume", "thread"],
            {
                "cwd": "project",
                "env": {"TERM": "xterm-256color"},
                "dimensions": (32, 120),
            },
        )
    ]


class FakeWriter:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.payloads.append(json.loads(data))

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeCodexProcess:
    def __init__(self, exit_code: int = 0) -> None:
        self.pid = 8080
        self.returncode: int | None = None
        self.exit_code = exit_code
        self.stdin = FakeWriter()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(b'{"id":0,"result":{"userAgent":"codex"}}\n')
        self.stdout.feed_data(b'{"method":"thread/started","params":{}}\n')
        self.stdout.feed_data(
            b'{"id":1,"result":{"thread":{"id":"assigned-thread-id"}}}\n'
        )
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    async def wait(self) -> int:
        self.returncode = self.exit_code
        return self.exit_code


@pytest.mark.asyncio
async def test_codex_thread_start_records_the_id_assigned_by_codex(
    monkeypatch, tmp_path
):
    process = FakeCodexProcess()

    async def create_process(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.register_pid",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.unregister_pid",
        lambda pid: None,
    )

    thread_id = await _create_codex_thread("fcc-codex", tmp_path)

    assert thread_id == "assigned-thread-id"
    assert process.stdin.payloads[0]["method"] == "initialize"
    assert process.stdin.payloads[1] == {"method": "initialized", "params": {}}
    assert process.stdin.payloads[2]["method"] == "thread/start"
    params = process.stdin.payloads[2]["params"]
    assert params == {"cwd": str(tmp_path), "serviceName": "free_claude_code"}
    assert process.stdin.closed is True


@pytest.mark.asyncio
async def test_codex_thread_delete_uses_the_stable_app_server_method(
    monkeypatch, tmp_path
):
    process = FakeCodexProcess()

    async def create_process(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.register_pid",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.unregister_pid",
        lambda pid: None,
    )

    await _delete_codex_thread("fcc-codex", tmp_path, "assigned-thread-id")

    assert process.stdin.payloads[2] == {
        "method": "thread/delete",
        "id": 1,
        "params": {"threadId": "assigned-thread-id"},
    }
    assert process.stdin.closed is True


@pytest.mark.asyncio
async def test_codex_assigned_id_is_rejected_when_app_server_cannot_save_it(
    monkeypatch, tmp_path
):
    process = FakeCodexProcess(exit_code=1)

    async def create_process(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.register_pid",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.drivers.unregister_pid",
        lambda pid: None,
    )

    with pytest.raises(BrowserSessionUnavailableError, match="could not save"):
        await _create_codex_thread("fcc-codex", tmp_path)

    assert process.stdin.closed is True

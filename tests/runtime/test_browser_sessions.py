import asyncio
import json
import signal
import sys
import threading
from queue import Queue
from types import SimpleNamespace
from uuid import UUID

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
    HarnessDriver,
    HarnessDriverRegistry,
    PiDriver,
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

SESSION_ID = "019f0000-0000-4000-8000-000000000001"
SECOND_SESSION_ID = "019f0000-0000-4000-8000-000000000002"


class FakeDriver(HarnessDriver):
    def __init__(self, harness: BrowserSessionHarness) -> None:
        self.harness = harness
        self.wrapper_name = f"fcc-{harness.value}"
        self.client_name = harness.value
        self.commands: list[tuple[str, bool]] = []

    def availability(self) -> HarnessAvailability:
        return HarnessAvailability(self.harness, True)

    def command(
        self,
        session_id: str,
        *,
        started_once: bool,
    ) -> list[str]:
        self.commands.append((session_id, started_once))
        return [f"fake-{self.harness.value}", session_id]


class FakeTerminalProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.writes: list[str] = []
        self.resizes: list[tuple[int, int]] = []
        self.terminated = False
        self.closed = False
        self.terminate_calls = 0
        self.close_calls = 0
        self.terminate_error: Exception | None = None
        self.close_error: Exception | None = None
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
        self.terminate_calls += 1
        if self.terminate_error is not None:
            raise self.terminate_error
        self.terminated = True
        self._chunks.put(None)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
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
        self.next_terminate_error: Exception | None = None
        self.next_close_error: Exception | None = None

    def spawn(self, command, *, cwd, env, columns, rows):
        process = FakeTerminalProcess(1000 + len(self.processes))
        process.terminate_error = self.next_terminate_error
        process.close_error = self.next_close_error
        self.next_terminate_error = None
        self.next_close_error = None
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
                session_id=SESSION_ID,
                path=str(tmp_path),
                name="Review",
                harness=BrowserSessionHarness.CLAUDE,
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


def test_store_rejects_duplicate_session_ids(tmp_path):
    path = tmp_path / "sessions.json"
    record = {
        "id": SESSION_ID,
        "path": str(tmp_path),
        "name": "Duplicate",
        "harness": "pi",
        "started_once": False,
    }
    path.write_text(
        json.dumps({"version": 1, "sessions": [record, record]}),
        encoding="utf-8",
    )

    with pytest.raises(SessionStoreError, match="original file was preserved"):
        BrowserSessionStore(path).load()


@pytest.mark.parametrize(
    "record",
    [
        SessionRecord(
            session_id=SESSION_ID,
            path="project",
            name="Claude",
            harness=BrowserSessionHarness.CLAUDE,
        ),
        SessionRecord(
            session_id=SECOND_SESSION_ID,
            path="project",
            name="Pi",
            harness=BrowserSessionHarness.PI,
            started_once=True,
        ),
    ],
)
def test_store_round_trips_supported_harnesses(tmp_path, record):
    store = BrowserSessionStore(tmp_path / "sessions.json")
    store.save(SessionCatalog(sessions=[record]))

    assert store.load().sessions == [record]


@pytest.mark.parametrize(
    ("session_id", "harness", "started_once"),
    [
        (SESSION_ID, "codex", False),
        ("ses_one", "claude", False),
        (SESSION_ID.upper(), "pi", False),
        (SESSION_ID, "claude", "false"),
    ],
)
def test_store_rejects_invalid_session_records(
    tmp_path, session_id, harness, started_once
):
    path = tmp_path / "sessions.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [
                    {
                        "id": session_id,
                        "path": str(tmp_path),
                        "name": "Invalid",
                        "harness": harness,
                        "started_once": started_once,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SessionStoreError, match="original file was preserved"):
        BrowserSessionStore(path).load()


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
    assert str(UUID(session.session_id)) == session.session_id
    assert [item.harness for item in (await manager.snapshot()).harnesses] == [
        BrowserSessionHarness.CLAUDE,
        BrowserSessionHarness.PI,
    ]
    assert drivers[BrowserSessionHarness.CLAUDE].commands == [
        (session.session_id, False)
    ]

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
        session.session_id,
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
        session.session_id,
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
            BrowserSessionHarness.CLAUDE,
            "unique",
        )

    other = tmp_path / "other"
    other.mkdir()
    same_name_elsewhere = await manager.create_session(
        str(other),
        BrowserSessionHarness.CLAUDE,
        "unique",
    )
    assert same_name_elsewhere.project_name == "other"
    assert same_name_elsewhere.path == str(other.resolve())
    await manager.close()


@pytest.mark.asyncio
async def test_manager_defaults_names_and_returns_newest_sessions_first(tmp_path):
    registry, _drivers = fake_drivers()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=FakeProcessFactory(),
    )

    first = await manager.create_session(str(tmp_path), BrowserSessionHarness.PI)
    second = await manager.create_session(str(tmp_path), BrowserSessionHarness.CLAUDE)

    assert first.name == "New session"
    assert second.name == "New session 2"
    assert [session.session_id for session in (await manager.snapshot()).sessions] == [
        second.session_id,
        first.session_id,
    ]
    await manager.delete_session(first.session_id)
    await manager.close()


@pytest.mark.asyncio
async def test_metadata_commit_failure_creates_no_native_resource(
    monkeypatch, tmp_path
):
    registry, _drivers = fake_drivers()
    store = BrowserSessionStore(tmp_path / "sessions.json")
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        store,
        drivers=registry,
        process_factory=factory,
    )

    def fail_save(catalog):
        raise SessionStoreError("Browser Sessions metadata could not be saved.")

    monkeypatch.setattr(store, "save", fail_save)

    with pytest.raises(BrowserSessionUnavailableError, match="could not be saved"):
        await manager.create_session(
            str(tmp_path),
            BrowserSessionHarness.CLAUDE,
            "Rollback",
        )

    assert (await manager.snapshot()).sessions == ()
    assert factory.processes == []
    await manager.close()


@pytest.mark.asyncio
async def test_first_start_commit_failure_terminates_the_uncommitted_process(
    monkeypatch, tmp_path
):
    registry, _drivers = fake_drivers()
    factory = FakeProcessFactory()
    store = BrowserSessionStore(tmp_path / "sessions.json")
    original_save = store.save
    save_count = 0

    def fail_second_save(catalog):
        nonlocal save_count
        save_count += 1
        if save_count == 2:
            raise SessionStoreError("Browser Sessions metadata could not be saved.")
        original_save(catalog)

    monkeypatch.setattr(store, "save", fail_second_save)
    manager = BrowserSessionManager(
        store,
        drivers=registry,
        process_factory=factory,
    )

    session = await manager.create_session(
        str(tmp_path), BrowserSessionHarness.PI, "Uncommitted start"
    )

    assert session.state is BrowserSessionState.FAILED
    assert session.detail == "Session startup could not be committed."
    assert factory.processes[0].terminated
    assert store.load().sessions[0].started_once is False
    await manager.close()


@pytest.mark.asyncio
async def test_start_rollback_closes_and_reconciles_after_termination_failure(
    monkeypatch, tmp_path
):
    registry, _drivers = fake_drivers()
    factory = FakeProcessFactory()
    factory.next_terminate_error = OSError("tree termination failed")
    store = BrowserSessionStore(tmp_path / "sessions.json")
    original_save = store.save
    save_count = 0

    def fail_second_save(catalog):
        nonlocal save_count
        save_count += 1
        if save_count == 2:
            raise SessionStoreError("Browser Sessions metadata could not be saved.")
        original_save(catalog)

    monkeypatch.setattr(store, "save", fail_second_save)
    manager = BrowserSessionManager(
        store,
        drivers=registry,
        process_factory=factory,
    )

    session = await manager.create_session(
        str(tmp_path), BrowserSessionHarness.CLAUDE, "Termination failure"
    )
    process = factory.processes[0]

    assert session.state is BrowserSessionState.FAILED
    assert session.detail == "Session startup could not be committed."
    assert process.terminate_calls == 1
    assert process.close_calls == 1
    assert process.closed is True

    restarted = await manager.start_session(session.session_id)
    assert restarted.state is BrowserSessionState.RUNNING
    await manager.close()


@pytest.mark.asyncio
async def test_start_rollback_retains_an_unclosed_process_for_stop_retry(
    monkeypatch, tmp_path
):
    registry, _drivers = fake_drivers()
    factory = FakeProcessFactory()
    factory.next_close_error = OSError("pty close failed")
    store = BrowserSessionStore(tmp_path / "sessions.json")
    original_save = store.save
    save_count = 0

    def fail_second_save(catalog):
        nonlocal save_count
        save_count += 1
        if save_count == 2:
            raise SessionStoreError("Browser Sessions metadata could not be saved.")
        original_save(catalog)

    monkeypatch.setattr(store, "save", fail_second_save)
    manager = BrowserSessionManager(
        store,
        drivers=registry,
        process_factory=factory,
    )

    session = await manager.create_session(
        str(tmp_path), BrowserSessionHarness.PI, "Close failure"
    )
    process = factory.processes[0]

    assert session.state is BrowserSessionState.FAILED
    assert session.detail == "Terminal cleanup is incomplete. Use Stop to retry."
    assert process.closed is False
    with pytest.raises(BrowserSessionConflictError, match="cleanup is incomplete"):
        await manager.start_session(session.session_id)

    process.close_error = None
    stopped = await manager.stop_session(session.session_id)
    assert stopped.state is BrowserSessionState.STOPPED
    assert process.closed is True
    await manager.close()


@pytest.mark.asyncio
async def test_manager_close_retries_incomplete_terminal_cleanup(tmp_path):
    registry, _drivers = fake_drivers()
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=factory,
    )
    await manager.create_session(
        str(tmp_path), BrowserSessionHarness.CLAUDE, "Shutdown retry"
    )
    process = factory.processes[0]
    process.close_error = OSError("pty close failed")

    with pytest.raises(BrowserSessionUnavailableError, match="shutdown is incomplete"):
        await manager.close()

    session = (await manager.snapshot()).sessions[0]
    assert session.state is BrowserSessionState.FAILED
    assert session.detail == "Terminal cleanup is incomplete. Use Stop to retry."
    assert process.closed is False

    process.close_error = None
    await manager.close()
    assert process.closed is True


@pytest.mark.asyncio
async def test_native_exit_leaves_no_shell_and_keeps_harness_lock(tmp_path):
    registry, drivers = fake_drivers()
    factory = FakeProcessFactory()
    manager = BrowserSessionManager(
        BrowserSessionStore(tmp_path / "sessions.json"),
        drivers=registry,
        process_factory=factory,
    )
    session = await manager.create_session(
        str(tmp_path), BrowserSessionHarness.PI, "Locked to Pi"
    )
    attachment = await manager.attach_terminal(session.session_id)
    await attachment.receive()
    process = factory.processes[0]

    await attachment.write("\x03")
    await attachment.write("\x03")
    process.finish()
    event = await asyncio.wait_for(attachment.receive(), timeout=1)

    assert process.writes == ["\x03", "\x03"]
    assert event.kind == "exit"
    assert event.state is BrowserSessionState.EXITED
    await manager.start_session(session.session_id)
    assert drivers[BrowserSessionHarness.PI].commands[-1] == (
        session.session_id,
        True,
    )
    assert all(spawn[0][0] == "fake-pi" for spawn in factory.spawns)
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
                    session_id=SESSION_ID,
                    path=str(tmp_path),
                    name="Cancellation",
                    harness=BrowserSessionHarness.CLAUDE,
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

    start = asyncio.create_task(manager.start_session(SESSION_ID))
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
        self.close_error: Exception | None = None

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
        if self.close_error is not None:
            raise self.close_error


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
    process.terminate_tree()

    assert registered == [4242]
    assert native.writes == ["hello"]
    assert native.dimensions == [(30, 100)]
    assert terminated == [4242]
    assert native.close_calls == [True]
    assert unregistered == [4242]


def test_native_terminal_process_retries_close_before_releasing_registration(
    monkeypatch,
):
    registered: list[int] = []
    unregistered: list[int] = []
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.register_pid",
        registered.append,
    )
    monkeypatch.setattr(
        "free_claude_code.runtime.browser_sessions.pty.unregister_pid",
        unregistered.append,
    )
    native = FakeNativePty()
    native.close_error = OSError("pty close failed")
    process = NativeTerminalProcess(native)

    with pytest.raises(OSError, match="pty close failed"):
        process.close()

    assert native.close_calls == [True]
    assert unregistered == []

    native.close_error = None
    process.close()
    process.close()

    assert registered == [4242]
    assert native.close_calls == [True, True]
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
        ["fcc-pi", "--session-id", SESSION_ID],
        cwd="project",
        env={"TERM": "xterm-256color"},
        columns=120,
        rows=32,
    )

    assert process.pid == 4242
    assert spawned == [
        (
            ["fcc-pi", "--session-id", SESSION_ID],
            {
                "cwd": "project",
                "env": {"TERM": "xterm-256color"},
                "dimensions": (32, 120),
            },
        )
    ]

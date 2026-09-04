import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from free_claude_code.application.terminal import (
    TerminalClientRole,
    TerminalConflictError,
    TerminalDeletedEvent,
    TerminalEngineSnapshot,
    TerminalNotFoundError,
    TerminalOutputEvent,
    TerminalResetEvent,
    TerminalService,
    TerminalStateEvent,
    TerminalStatus,
    TerminalUnavailableError,
    TerminalValidationError,
)


class FakeTerminalClient:
    def __init__(self, role: TerminalClientRole) -> None:
        self.role = role
        self._reads: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.writes: list[str] = []
        self.resizes: list[tuple[int, int]] = []
        self.closed = False

    async def read(self) -> bytes:
        return (await self._reads.get()) or b""

    async def write(self, data: str) -> None:
        self.writes.append(data)

    async def resize(self, rows: int, columns: int) -> None:
        self.resizes.append((rows, columns))

    async def wait(self) -> int | None:
        return 0

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._reads.put_nowait(None)

    def emit(self, data: bytes) -> None:
        self._reads.put_nowait(data)


class FakeTerminalEngineSession:
    def __init__(self) -> None:
        self.clients: list[FakeTerminalClient] = []
        self.opened: list[tuple[TerminalClientRole, int, int]] = []
        self.resizes: list[tuple[int, int]] = []
        self.snapshot_value = TerminalEngineSnapshot(b"history\r\n", b"screen\r\n")
        self.snapshot_failure: Exception | None = None
        self.open_failure: Exception | None = None
        self.resize_failure: Exception | None = None
        self.terminate_failure: Exception | None = None
        self.close_failure: Exception | None = None
        self.terminate_started = asyncio.Event()
        self.release_terminate = asyncio.Event()
        self.release_terminate.set()
        self._root_exit = asyncio.Event()
        self._exit_code: int | None = None
        self.terminate_calls = 0
        self.close_calls = 0

    async def open_client(
        self,
        role: TerminalClientRole,
        *,
        rows: int,
        columns: int,
    ) -> FakeTerminalClient:
        if self.open_failure is not None:
            raise self.open_failure
        client = FakeTerminalClient(role)
        self.clients.append(client)
        self.opened.append((role, rows, columns))
        return client

    async def snapshot(self) -> TerminalEngineSnapshot:
        if self.snapshot_failure is not None:
            raise self.snapshot_failure
        return self.snapshot_value

    async def resize(self, rows: int, columns: int) -> None:
        if self.resize_failure is not None:
            raise self.resize_failure
        self.resizes.append((rows, columns))

    async def wait_root(self) -> int | None:
        await self._root_exit.wait()
        return self._exit_code

    async def terminate_tree(self) -> None:
        self.terminate_calls += 1
        self.terminate_started.set()
        await self.release_terminate.wait()
        if self.terminate_failure is not None:
            raise self.terminate_failure
        self.exit(143)

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure

    def exit(self, code: int | None) -> None:
        if self._root_exit.is_set():
            return
        self._exit_code = code
        self._root_exit.set()


class FakeTerminalEngineHost:
    def __init__(self) -> None:
        self.start_failure: Exception | None = None
        self.create_failure: Exception | None = None
        self.close_failure: Exception | None = None
        self.engines: list[FakeTerminalEngineSession] = []
        self.creates: list[
            tuple[str, tuple[str, ...], Path, Mapping[str, str], int, int]
        ] = []
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1
        if self.start_failure is not None:
            raise self.start_failure

    async def create_session(
        self,
        *,
        session_name: str,
        command: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> FakeTerminalEngineSession:
        if self.create_failure is not None:
            raise self.create_failure
        engine = FakeTerminalEngineSession()
        self.engines.append(engine)
        self.creates.append(
            (session_name, tuple(command), cwd, dict(env), rows, columns)
        )
        return engine

    async def close(self) -> None:
        self.closed += 1
        if self.close_failure is not None:
            raise self.close_failure


async def _service(
    *,
    ids: Sequence[str] = ("session-one",),
) -> tuple[TerminalService, FakeTerminalEngineHost]:
    identifiers = iter(ids)
    host = FakeTerminalEngineHost()
    service = TerminalService(
        host,
        home=Path("/home/example"),
        env={"EXAMPLE": "yes"},
        clock=lambda: 1234,
        id_factory=lambda: next(identifiers),
        shell_factory=lambda _env: ("shell", "--interactive"),
    )
    await service.start()
    return service, host


async def _created_service() -> tuple[
    TerminalService, FakeTerminalEngineHost, FakeTerminalEngineSession, str
]:
    service, host = await _service()
    session = await service.create_session()
    return service, host, host.engines[0], session.id


async def _next_state(events) -> TerminalStateEvent:
    while True:
        event = await asyncio.wait_for(anext(events), timeout=1)
        if isinstance(event, TerminalStateEvent):
            return event


async def _next_reset(events) -> TerminalResetEvent:
    while True:
        event = await asyncio.wait_for(anext(events), timeout=1)
        if isinstance(event, TerminalResetEvent):
            return event


@pytest.mark.asyncio
async def test_create_uses_home_environment_shell_and_monotonic_names() -> None:
    service, host = await _service(ids=("one", "two"))

    first = await service.create_session()
    second = await service.create_session()

    assert (first.id, first.name, first.status) == (
        "one",
        "Terminal 1",
        TerminalStatus.RUNNING,
    )
    assert second.name == "Terminal 2"
    assert host.creates == [
        (
            "fcc-one",
            ("shell", "--interactive"),
            Path("/home/example"),
            {"EXAMPLE": "yes"},
            24,
            80,
        ),
        (
            "fcc-two",
            ("shell", "--interactive"),
            Path("/home/example"),
            {"EXAMPLE": "yes"},
            24,
            80,
        ),
    ]
    await service.close()


@pytest.mark.asyncio
async def test_engine_unavailability_is_scoped_to_terminal_sessions() -> None:
    host = FakeTerminalEngineHost()
    host.start_failure = FileNotFoundError("zellij missing")
    service = TerminalService(host)

    await service.start()

    assert service.availability_error == (
        "Terminal Sessions is unavailable. Rerun the FCC installer and restart FCC."
    )
    assert await service.list_sessions() == ()
    with pytest.raises(TerminalUnavailableError, match="Rerun the FCC installer"):
        await service.create_session()
    await service.close()
    assert host.closed == 0


@pytest.mark.asyncio
async def test_failed_engine_creation_does_not_publish_or_consume_name() -> None:
    service, host = await _service(ids=("one", "two"))
    host.create_failure = OSError("create failed")

    with pytest.raises(TerminalUnavailableError, match="system shell"):
        await service.create_session()
    assert await service.list_sessions() == ()

    host.create_failure = None
    session = await service.create_session()
    assert (session.id, session.name) == ("two", "Terminal 1")
    await service.close()


@pytest.mark.asyncio
async def test_each_attachment_owns_an_independent_engine_client() -> None:
    service, _, engine, session_id = await _created_service()
    first = await service.attach(session_id, rows=30, columns=100)
    second = await service.attach(session_id, rows=40, columns=120)
    first_events = first.__aiter__()
    second_events = second.__aiter__()

    assert first.initial.role is TerminalClientRole.CONTROLLER
    assert second.initial.role is TerminalClientRole.OBSERVER
    assert first.initial.output == second.initial.output == b"history\r\n"
    assert engine.opened == [
        (TerminalClientRole.CONTROLLER, 30, 100),
        (TerminalClientRole.OBSERVER, 30, 100),
    ]
    assert engine.resizes == [(30, 100)]

    engine.clients[0].emit(b"controller output")
    engine.clients[1].emit(b"observer output")
    assert await asyncio.wait_for(anext(first_events), timeout=1) == (
        TerminalOutputEvent(b"controller output")
    )
    assert await asyncio.wait_for(anext(second_events), timeout=1) == (
        TerminalOutputEvent(b"observer output")
    )

    await first.aclose()
    await second.aclose()
    assert engine.terminate_calls == 0
    await service.close()


@pytest.mark.asyncio
async def test_claim_rebuilds_only_the_controller_and_claimant_clients() -> None:
    service, _, engine, session_id = await _created_service()
    first = await service.attach(session_id, rows=30, columns=100)
    second = await service.attach(session_id, rows=40, columns=120)
    first_events = first.__aiter__()
    second_events = second.__aiter__()
    old_controller, old_observer = engine.clients

    await second.claim()

    assert old_controller.closed
    assert old_observer.closed
    assert engine.opened[-2:] == [
        (TerminalClientRole.CONTROLLER, 40, 120),
        (TerminalClientRole.OBSERVER, 40, 120),
    ]
    assert engine.resizes == [(30, 100), (40, 120)]
    assert await anext(second_events) == TerminalResetEvent(
        output=b"history\r\n", role=TerminalClientRole.CONTROLLER
    )
    assert await anext(first_events) == TerminalResetEvent(
        output=b"history\r\n", role=TerminalClientRole.OBSERVER
    )

    await second.write("whoami\r")
    assert engine.clients[-2].writes == ["whoami\r"]
    assert engine.clients[-1].writes == []
    await first.aclose()
    await second.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_failed_claim_keeps_existing_roles_and_dimensions() -> None:
    service, _, engine, session_id = await _created_service()
    controller = await service.attach(session_id, rows=30, columns=100)
    observer = await service.attach(session_id, rows=40, columns=120)
    old_controller, old_observer = engine.clients
    engine.open_failure = OSError("attach failed")

    with pytest.raises(TerminalUnavailableError, match="transfer terminal control"):
        await observer.claim()

    session = await service.get_session(session_id)
    assert (session.rows, session.columns) == (30, 100)
    assert engine.resizes == [(30, 100), (40, 120), (30, 100)]
    assert not old_controller.closed
    assert not old_observer.closed

    await controller.write("still-controller")
    assert old_controller.writes == ["still-controller"]
    await controller.aclose()
    await observer.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_observer_input_and_resize_stay_on_its_read_only_view() -> None:
    service, _, engine, session_id = await _created_service()
    controller = await service.attach(session_id, rows=30, columns=100)
    observer = await service.attach(session_id, rows=40, columns=120)

    await observer.write("terminal-query-reply")
    await observer.resize(rows=50, columns=140)

    assert engine.clients[1].writes == ["terminal-query-reply"]
    assert engine.resizes == [(30, 100)]
    assert (await service.get_session(session_id)).rows == 30

    await controller.resize(rows=35, columns=110)
    assert engine.resizes == [(30, 100), (35, 110)]
    assert (await service.get_session(session_id)).rows == 35
    await controller.aclose()
    await observer.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_controller_disconnect_promotes_most_recent_interactor() -> None:
    service, _, engine, session_id = await _created_service()
    first = await service.attach(session_id, rows=24, columns=80)
    second = await service.attach(session_id, rows=30, columns=90)
    third = await service.attach(session_id, rows=40, columns=120)
    third_events = third.__aiter__()

    await third.claim()
    await _next_reset(third_events)
    await first.claim()
    assert (await _next_reset(third_events)).role is TerminalClientRole.OBSERVER
    await first.aclose()

    reset = await _next_reset(third_events)
    assert reset.role is TerminalClientRole.CONTROLLER
    assert engine.opened[-1][0] is TerminalClientRole.CONTROLLER
    await second.aclose()
    await third.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_rename_input_and_dimension_validation() -> None:
    service, _, engine, session_id = await _created_service()
    attachment = await service.attach(session_id, rows=24, columns=80)
    events = attachment.__aiter__()

    renamed = await service.rename_session(session_id, "  Build shell  ")
    assert renamed.name == "Build shell"
    assert (await _next_state(events)).session.name == "Build shell"

    await attachment.write("echo hello\r")
    assert engine.clients[0].writes == ["echo hello\r"]
    with pytest.raises(TerminalValidationError, match="cannot be empty"):
        await service.rename_session(session_id, " ")
    with pytest.raises(TerminalValidationError, match="dimensions"):
        await attachment.resize(rows=0, columns=80)
    with pytest.raises(TerminalValidationError, match="65536"):
        await attachment.write("é" * (32 * 1024 + 1))

    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_natural_exit_captures_final_screen_and_stop_is_idempotent() -> None:
    service, _, engine, session_id = await _created_service()
    attachment = await service.attach(session_id, rows=24, columns=80)
    events = attachment.__aiter__()
    engine.snapshot_value = TerminalEngineSnapshot(b"final history\r\n", b"prompt")

    engine.exit(7)

    stopping = await _next_state(events)
    exited = await _next_state(events)
    assert stopping.session.status is TerminalStatus.STOPPING
    assert (exited.session.status, exited.session.exit_code) == (
        TerminalStatus.EXITED,
        7,
    )
    assert (engine.terminate_calls, engine.close_calls) == (1, 1)
    assert (await service.stop_session(session_id)).status is TerminalStatus.EXITED
    assert engine.terminate_calls == 1

    retained = await service.attach(session_id, rows=50, columns=120)
    assert retained.initial.output == b"final history\r\nprompt"
    assert retained.initial.role is TerminalClientRole.OBSERVER
    with pytest.raises(TerminalConflictError):
        await retained.write("too late")
    await retained.aclose()
    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_prevent_cleanup() -> None:
    service, _, engine, session_id = await _created_service()
    engine.snapshot_failure = ValueError("bad snapshot")

    stopped = await service.stop_session(session_id)

    assert stopped.status is TerminalStatus.EXITED
    assert stopped.error == "Final terminal output is unavailable (ValueError)."
    assert (engine.terminate_calls, engine.close_calls) == (1, 1)
    await service.close()


@pytest.mark.asyncio
async def test_cleanup_failure_stays_stopping_and_retries() -> None:
    service, _, engine, session_id = await _created_service()
    engine.terminate_failure = OSError("busy")

    with pytest.raises(TerminalUnavailableError, match="process tree"):
        await service.stop_session(session_id)
    assert (await service.get_session(session_id)).status is TerminalStatus.STOPPING

    engine.terminate_failure = None
    stopped = await service.stop_session(session_id)
    assert stopped.status is TerminalStatus.EXITED
    assert engine.terminate_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_delete_notifies_views_then_removes_the_session() -> None:
    service, _, engine, session_id = await _created_service()
    attachment = await service.attach(session_id, rows=24, columns=80)
    events = attachment.__aiter__()

    await service.delete_session(session_id)

    seen_deleted = False
    async for event in events:
        if isinstance(event, TerminalDeletedEvent):
            seen_deleted = True
    assert seen_deleted
    assert engine.terminate_calls == 1
    with pytest.raises(TerminalNotFoundError):
        await service.get_session(session_id)
    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_stop_and_delete_race_terminalizes_only_once() -> None:
    service, _, engine, session_id = await _created_service()

    results = await asyncio.gather(
        service.stop_session(session_id),
        service.delete_session(session_id),
        return_exceptions=True,
    )

    assert not any(isinstance(result, Exception) for result in results)
    assert engine.terminate_calls == 1
    assert await service.list_sessions() == ()
    await service.close()


@pytest.mark.asyncio
async def test_shutdown_attempts_all_session_cleanup_concurrently() -> None:
    service, host = await _service(ids=("one", "two"))
    await service.create_session()
    await service.create_session()
    first, second = host.engines
    first.release_terminate.clear()
    second.release_terminate.clear()

    closing = asyncio.create_task(service.close())
    await asyncio.wait_for(
        asyncio.gather(first.terminate_started.wait(), second.terminate_started.wait()),
        timeout=1,
    )
    first.release_terminate.set()
    second.release_terminate.set()
    await closing

    assert await service.list_sessions() == ()
    assert host.closed == 1


@pytest.mark.asyncio
async def test_engine_host_close_failure_is_retried_without_sessions() -> None:
    service, host = await _service()
    host.close_failure = OSError("busy")

    with pytest.raises(TerminalUnavailableError, match="terminal engine"):
        await service.close()
    host.close_failure = None
    await service.close()

    assert host.closed == 2

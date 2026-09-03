import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from free_claude_code.application.terminal import (
    TerminalAttachmentOverflowError,
    TerminalConflictError,
    TerminalDeletedEvent,
    TerminalNotFoundError,
    TerminalOutputEvent,
    TerminalProcessPort,
    TerminalService,
    TerminalStateEvent,
    TerminalStatus,
    TerminalUnavailableError,
    TerminalValidationError,
)


class FakeTerminalProcess(TerminalProcessPort):
    def __init__(self) -> None:
        self._reads: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._exit_code: int | None = None
        self._exited = asyncio.Event()
        self.writes: list[bytes] = []
        self.resizes: list[tuple[int, int]] = []
        self.terminated = False
        self.closed = False
        self.block_writes = False
        self.write_started = asyncio.Event()
        self.release_writes = asyncio.Event()
        self.active_writes = 0
        self.max_active_writes = 0
        self.terminate_failure: Exception | None = None
        self.close_failure: Exception | None = None

    @property
    def pid(self) -> int:
        return 123

    @property
    def alive(self) -> bool:
        return not self._exited.is_set()

    async def read(self) -> bytes:
        return (await self._reads.get()) or b""

    async def write(self, data: bytes) -> None:
        self.active_writes += 1
        self.max_active_writes = max(self.max_active_writes, self.active_writes)
        self.write_started.set()
        try:
            if self.block_writes:
                await self.release_writes.wait()
            self.writes.append(data)
        finally:
            self.active_writes -= 1

    async def resize(self, rows: int, columns: int) -> None:
        self.resizes.append((rows, columns))

    async def wait(self) -> int | None:
        await self._exited.wait()
        return self._exit_code

    async def terminate_tree(self) -> None:
        if self.terminate_failure is not None:
            raise self.terminate_failure
        self.terminated = True
        self.exit(143)

    async def close(self) -> None:
        if self.close_failure is not None:
            raise self.close_failure
        self.closed = True

    def emit(self, data: bytes) -> None:
        self._reads.put_nowait(data)

    def exit(self, code: int) -> None:
        if self._exited.is_set():
            return
        self._exit_code = code
        self._exited.set()
        self._reads.put_nowait(None)


class FakeTerminalFactory:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.processes: list[FakeTerminalProcess] = []
        self.spawns: list[tuple[Path, Mapping[str, str], int, int]] = []

    async def spawn(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> FakeTerminalProcess:
        if self.failure is not None:
            raise self.failure
        process = FakeTerminalProcess()
        self.processes.append(process)
        self.spawns.append((cwd, env, rows, columns))
        return process


async def _service(
    *,
    output_limit: int = 64,
    queue_size: int = 8,
) -> tuple[TerminalService, FakeTerminalFactory, str]:
    factory = FakeTerminalFactory()
    service = TerminalService(
        factory,
        home=Path("/home/example"),
        env={"EXAMPLE": "yes"},
        output_limit=output_limit,
        attachment_queue_size=queue_size,
        clock=lambda: 1234,
        id_factory=lambda: "session-one",
    )
    await service.start()
    session = await service.create_session()
    return service, factory, session.id


@pytest.mark.asyncio
async def test_create_uses_home_environment_and_monotonic_names() -> None:
    ids = iter(("one", "two"))
    factory = FakeTerminalFactory()
    service = TerminalService(
        factory,
        home=Path("/home/example"),
        env={"EXAMPLE": "yes"},
        clock=lambda: 1234,
        id_factory=lambda: next(ids),
    )
    await service.start()

    first = await service.create_session()
    second = await service.create_session()

    assert (first.id, first.name, first.status) == (
        "one",
        "Terminal 1",
        TerminalStatus.RUNNING,
    )
    assert second.name == "Terminal 2"
    assert factory.spawns == [
        (Path("/home/example"), {"EXAMPLE": "yes"}, 24, 80),
        (Path("/home/example"), {"EXAMPLE": "yes"}, 24, 80),
    ]
    await service.close()


@pytest.mark.asyncio
async def test_failed_spawn_does_not_publish_a_session() -> None:
    service = TerminalService(FakeTerminalFactory(failure=OSError("spawn failed")))
    await service.start()

    with pytest.raises(TerminalUnavailableError):
        await service.create_session()

    assert await service.list_sessions() == ()
    await service.close()


@pytest.mark.asyncio
async def test_two_attachments_receive_output_and_detach_does_not_stop() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    first = await service.attach(session_id)
    second = await service.attach(session_id)
    first_events = first.__aiter__()
    second_events = second.__aiter__()

    process.emit(b"hello")
    first_event = await asyncio.wait_for(anext(first_events), timeout=1)
    second_event = await asyncio.wait_for(anext(second_events), timeout=1)

    assert first_event == second_event == TerminalOutputEvent(b"hello")
    await first.aclose()
    assert process.alive
    assert not process.terminated

    process.emit(b" again")
    assert await asyncio.wait_for(anext(second_events), timeout=1) == (
        TerminalOutputEvent(b" again")
    )
    await second.aclose()
    assert process.alive
    await service.close()


@pytest.mark.asyncio
async def test_attach_replays_retained_output_then_follows_live_output_once() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    observer = await service.attach(session_id)
    observer_events = observer.__aiter__()
    process.emit(b"before attach")
    assert await asyncio.wait_for(anext(observer_events), timeout=1) == (
        TerminalOutputEvent(b"before attach")
    )

    attached = await service.attach(session_id)
    events = attached.__aiter__()
    assert attached.initial.output == b"before attach"
    process.emit(b" after attach")
    assert await asyncio.wait_for(anext(events), timeout=1) == (
        TerminalOutputEvent(b" after attach")
    )

    await observer.aclose()
    await attached.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_sessions_keep_input_and_output_isolated() -> None:
    ids = iter(("one", "two"))
    factory = FakeTerminalFactory()
    service = TerminalService(factory, id_factory=lambda: next(ids))
    await service.start()
    first = await service.create_session()
    second = await service.create_session()
    first_attachment = await service.attach(first.id)
    second_attachment = await service.attach(second.id)
    first_events = first_attachment.__aiter__()
    second_events = second_attachment.__aiter__()

    factory.processes[0].emit(b"first output")
    factory.processes[1].emit(b"second output")
    assert await anext(first_events) == TerminalOutputEvent(b"first output")
    assert await anext(second_events) == TerminalOutputEvent(b"second output")

    await service.write(first.id, b"first input")
    await service.write(second.id, b"second input")
    assert factory.processes[0].writes == [b"first input"]
    assert factory.processes[1].writes == [b"second input"]

    await first_attachment.aclose()
    await second_attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_reattach_receives_bounded_retained_output() -> None:
    service, factory, session_id = await _service(output_limit=5)
    process = factory.processes[0]
    process.emit(b"abc")
    process.emit(b"def")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    attachment = await service.attach(session_id)

    assert attachment.initial.output == b"bcdef"
    assert attachment.initial.session.history_truncated
    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_exact_output_limit_is_not_reported_as_truncated() -> None:
    service, factory, session_id = await _service(output_limit=5)
    factory.processes[0].emit(b"abcde")
    await asyncio.sleep(0)

    attachment = await service.attach(session_id)

    assert attachment.initial.output == b"abcde"
    assert not attachment.initial.session.history_truncated
    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_slow_attachment_overflows_without_affecting_healthy_one() -> None:
    service, factory, session_id = await _service(queue_size=1)
    process = factory.processes[0]
    slow = await service.attach(session_id)
    healthy = await service.attach(session_id)
    slow_events = slow.__aiter__()
    healthy_events = healthy.__aiter__()

    process.emit(b"one")
    assert await asyncio.wait_for(anext(healthy_events), timeout=1) == (
        TerminalOutputEvent(b"one")
    )
    process.emit(b"two")
    assert await asyncio.wait_for(anext(healthy_events), timeout=1) == (
        TerminalOutputEvent(b"two")
    )

    with pytest.raises(TerminalAttachmentOverflowError):
        await asyncio.wait_for(anext(slow_events), timeout=1)
    assert process.alive
    await slow.aclose()
    await healthy.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_write_resize_rename_and_validation() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    attachment = await service.attach(session_id)
    events = attachment.__aiter__()

    await service.write(session_id, b"echo hello\r")
    resized = await service.resize(session_id, rows=40, columns=120)
    renamed = await service.rename_session(session_id, "  Build shell  ")

    assert process.writes == [b"echo hello\r"]
    assert process.resizes == [(40, 120)]
    assert (resized.rows, resized.columns) == (40, 120)
    assert renamed.name == "Build shell"
    assert isinstance(await anext(events), TerminalStateEvent)
    assert isinstance(await anext(events), TerminalStateEvent)

    with pytest.raises(TerminalValidationError):
        await service.rename_session(session_id, " ")
    with pytest.raises(TerminalValidationError):
        await service.resize(session_id, rows=0, columns=80)
    with pytest.raises(TerminalValidationError):
        await service.write(session_id, b"x" * (64 * 1024 + 1))

    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_concurrent_input_messages_are_serialized() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    process.block_writes = True

    first = asyncio.create_task(service.write(session_id, b"first"))
    await process.write_started.wait()
    second = asyncio.create_task(service.write(session_id, b"second"))
    await asyncio.sleep(0)

    assert process.max_active_writes == 1
    process.release_writes.set()
    await asyncio.gather(first, second)
    assert process.writes == [b"first", b"second"]
    assert process.max_active_writes == 1
    await service.close()


@pytest.mark.asyncio
async def test_normal_exit_drains_output_and_stop_is_idempotent() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    attachment = await service.attach(session_id)
    events = attachment.__aiter__()
    process.emit(b"last output")
    process.exit(7)

    assert await asyncio.wait_for(anext(events), timeout=1) == (
        TerminalOutputEvent(b"last output")
    )
    state = await asyncio.wait_for(anext(events), timeout=1)
    assert isinstance(state, TerminalStateEvent)
    assert (state.session.status, state.session.exit_code) == (
        TerminalStatus.EXITED,
        7,
    )
    assert process.closed

    stopped = await service.stop_session(session_id)
    assert stopped.status is TerminalStatus.EXITED
    assert not process.terminated
    with pytest.raises(TerminalConflictError):
        await service.write(session_id, b"too late")
    with pytest.raises(TerminalConflictError):
        await service.resize(session_id, rows=24, columns=80)
    await attachment.aclose()
    await service.close()


@pytest.mark.asyncio
async def test_delete_stops_process_and_removes_session() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    first = await service.attach(session_id)
    second = await service.attach(session_id)

    await service.delete_session(session_id)

    assert process.terminated
    assert process.closed
    for attachment in (first, second):
        attachment_events = attachment.__aiter__()
        events = [await anext(attachment_events) for _ in range(3)]
        assert isinstance(events[-1], TerminalDeletedEvent)
        await attachment.aclose()
    with pytest.raises(TerminalNotFoundError):
        await service.get_session(session_id)
    await service.close()


@pytest.mark.asyncio
async def test_close_terminates_every_process() -> None:
    ids = iter(("one", "two"))
    factory = FakeTerminalFactory()
    service = TerminalService(factory, id_factory=lambda: next(ids))
    await service.start()
    await service.create_session()
    await service.create_session()

    await service.close()

    assert all(process.terminated and process.closed for process in factory.processes)
    assert await service.list_sessions() == ()


@pytest.mark.asyncio
async def test_failed_close_retains_session_for_cleanup_retry() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    process.terminate_failure = OSError("busy")

    with pytest.raises(TerminalUnavailableError):
        await service.close()

    assert (await service.get_session(session_id)).status is TerminalStatus.STOPPING
    process.terminate_failure = None
    await service.close()
    assert await service.list_sessions() == ()


@pytest.mark.asyncio
async def test_failed_process_handle_cleanup_is_retained_for_retry() -> None:
    service, factory, session_id = await _service()
    process = factory.processes[0]
    process.close_failure = OSError("busy")
    attachment = await service.attach(session_id)
    events = attachment.__aiter__()
    process.exit(0)

    state = await asyncio.wait_for(anext(events), timeout=1)
    assert isinstance(state, TerminalStateEvent)
    assert state.session.status is TerminalStatus.EXITED
    assert state.session.error == "Terminal process cleanup failed (OSError)."

    with pytest.raises(TerminalUnavailableError):
        await service.close()
    assert (await service.get_session(session_id)).status is TerminalStatus.EXITED

    process.close_failure = None
    await service.close()
    assert process.closed
    assert await service.list_sessions() == ()
    await attachment.aclose()

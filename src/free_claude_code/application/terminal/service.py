"""Server-owned attach/detach lifecycle for interactive Terminal Sessions."""

import asyncio
import os
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from uuid import uuid4

from .models import (
    TerminalAttachmentEvent,
    TerminalAttachmentOverflowError,
    TerminalAttachmentSnapshot,
    TerminalConflictError,
    TerminalDeletedEvent,
    TerminalNotFoundError,
    TerminalOutputEvent,
    TerminalSession,
    TerminalStateEvent,
    TerminalStatus,
    TerminalUnavailableError,
    TerminalValidationError,
)
from .ports import (
    TerminalAttachmentPort,
    TerminalProcessFactoryPort,
    TerminalProcessPort,
)

DEFAULT_ROWS = 24
DEFAULT_COLUMNS = 80
MAX_ROWS = 500
MAX_COLUMNS = 1_000
MAX_INPUT_BYTES = 64 * 1024
DEFAULT_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_ATTACHMENT_QUEUE_SIZE = 256
MAX_NAME_LENGTH = 100
_OUTPUT_CHUNK_BYTES = 64 * 1024


class _Closed:
    pass


class _Overflow:
    pass


_CLOSED = _Closed()
_OVERFLOW = _Overflow()
type _AttachmentQueueItem = TerminalAttachmentEvent | _Closed | _Overflow


class _ByteRingBuffer:
    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Terminal output limit must be positive.")
        self._limit = limit
        self._chunks: deque[bytearray] = deque()
        self._size = 0
        self.truncated = False

    def append(self, data: bytes) -> None:
        if not data:
            return
        if len(data) >= self._limit:
            discarded = self._size > 0 or len(data) > self._limit
            self._chunks.clear()
            self._chunks.append(bytearray(data[-self._limit :]))
            self._size = self._limit
            self.truncated = self.truncated or discarded
            return

        remaining = memoryview(data)
        if self._chunks and len(self._chunks[-1]) < _OUTPUT_CHUNK_BYTES:
            take = min(_OUTPUT_CHUNK_BYTES - len(self._chunks[-1]), len(remaining))
            self._chunks[-1].extend(remaining[:take])
            self._size += take
            remaining = remaining[take:]
        while remaining:
            take = min(_OUTPUT_CHUNK_BYTES, len(remaining))
            self._chunks.append(bytearray(remaining[:take]))
            self._size += take
            remaining = remaining[take:]
        while self._size > self._limit:
            excess = self._size - self._limit
            first = self._chunks[0]
            if len(first) <= excess:
                self._chunks.popleft()
                self._size -= len(first)
            else:
                del first[:excess]
                self._size -= excess
            self.truncated = True

    def snapshot(self) -> bytes:
        return b"".join(self._chunks)


class _TerminalAttachment(TerminalAttachmentPort):
    def __init__(
        self,
        owner: _TerminalSession,
        *,
        initial: TerminalAttachmentSnapshot,
        queue_size: int,
    ) -> None:
        self._owner = owner
        self._initial = initial
        self._queue: asyncio.Queue[_AttachmentQueueItem] = asyncio.Queue(
            maxsize=queue_size
        )
        self._closed = False

    @property
    def initial(self) -> TerminalAttachmentSnapshot:
        return self._initial

    def __aiter__(self) -> AsyncIterator[TerminalAttachmentEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[TerminalAttachmentEvent]:
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                return
            if item is _OVERFLOW:
                raise TerminalAttachmentOverflowError(
                    "Terminal attachment fell behind retained output."
                )
            if isinstance(
                item,
                (TerminalOutputEvent, TerminalStateEvent, TerminalDeletedEvent),
            ):
                yield item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._owner.detach(self)
        self._signal(_CLOSED)

    def _signal(self, item: _AttachmentQueueItem) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(item)


class _TerminalSession:
    def __init__(
        self,
        *,
        session_id: str,
        name: str,
        created_at: int,
        process: TerminalProcessPort,
        output_limit: int,
        attachment_queue_size: int,
    ) -> None:
        self._id = session_id
        self._name = name
        self._created_at = created_at
        self._process = process
        self._status = TerminalStatus.RUNNING
        self._rows = DEFAULT_ROWS
        self._columns = DEFAULT_COLUMNS
        self._exit_code: int | None = None
        self._error: str | None = None
        self._cleanup_error: str | None = None
        self._process_closed = False
        self._deleted = False
        self._buffer = _ByteRingBuffer(output_limit)
        self._attachment_queue_size = attachment_queue_size
        self._attachments: set[_TerminalAttachment] = set()
        self._state_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._settled = asyncio.Event()

    def start(self) -> None:
        if self._reader_task is not None:
            raise RuntimeError("Terminal reader already started.")
        self._reader_task = asyncio.create_task(
            self._read_output(), name=f"terminal-reader-{self._id}"
        )

    async def snapshot(self) -> TerminalSession:
        async with self._state_lock:
            return self._snapshot_locked()

    async def rename(self, name: str) -> TerminalSession:
        async with self._state_lock:
            self._require_present_locked()
            self._name = name
            snapshot = self._snapshot_locked()
            self._publish_locked(TerminalStateEvent(snapshot))
            return snapshot

    async def attach(self) -> _TerminalAttachment:
        async with self._state_lock:
            self._require_present_locked()
            attachment = _TerminalAttachment(
                self,
                initial=TerminalAttachmentSnapshot(
                    session=self._snapshot_locked(),
                    output=self._buffer.snapshot(),
                ),
                queue_size=self._attachment_queue_size,
            )
            self._attachments.add(attachment)
            return attachment

    async def detach(self, attachment: _TerminalAttachment) -> None:
        async with self._state_lock:
            self._attachments.discard(attachment)

    async def write(self, data: bytes) -> None:
        async with self._control_lock:
            async with self._state_lock:
                self._require_running_locked()
            try:
                await self._process.write(data)
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not write to the terminal process."
                ) from exc

    async def resize(self, *, rows: int, columns: int) -> TerminalSession:
        async with self._control_lock:
            async with self._state_lock:
                self._require_running_locked()
            try:
                await self._process.resize(rows, columns)
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not resize the terminal process."
                ) from exc
            async with self._state_lock:
                self._require_present_locked()
                if self._status is TerminalStatus.RUNNING:
                    self._rows = rows
                    self._columns = columns
                snapshot = self._snapshot_locked()
                self._publish_locked(TerminalStateEvent(snapshot))
                return snapshot

    async def stop(self) -> TerminalSession:
        async with self._stop_lock:
            async with self._state_lock:
                self._require_present_locked()
                if self._status is TerminalStatus.EXITED:
                    return self._snapshot_locked()
                self._status = TerminalStatus.STOPPING
                self._publish_locked(TerminalStateEvent(self._snapshot_locked()))

            try:
                async with self._control_lock:
                    await self._process.terminate_tree()
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not stop the terminal process tree."
                ) from exc

            await self._settled.wait()
            return await self.snapshot()

    async def dispose(self) -> None:
        await self.stop()
        close_failure = await self._close_process()
        async with self._state_lock:
            self._cleanup_error = _cleanup_error_message(close_failure)
        if close_failure is not None:
            raise TerminalUnavailableError(
                "Could not close the terminal process handles."
            ) from close_failure
        async with self._state_lock:
            if self._deleted:
                return
            self._deleted = True
            attachments = tuple(self._attachments)
            self._attachments.clear()
            for attachment in attachments:
                try:
                    attachment._queue.put_nowait(TerminalDeletedEvent())
                except asyncio.QueueFull:
                    attachment._signal(TerminalDeletedEvent())

    async def _read_output(self) -> None:
        exit_code: int | None = None
        error: str | None = None
        try:
            while True:
                data = await self._process.read()
                if not data:
                    break
                async with self._state_lock:
                    self._buffer.append(data)
                    self._publish_locked(TerminalOutputEvent(data))
            exit_code = await self._process.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"Terminal process ended unexpectedly ({type(exc).__name__})."
        finally:
            close_failure = await self._close_process()
            async with self._state_lock:
                self._status = TerminalStatus.EXITED
                self._exit_code = exit_code
                self._error = error
                self._cleanup_error = _cleanup_error_message(close_failure)
                self._publish_locked(TerminalStateEvent(self._snapshot_locked()))
                self._settled.set()

    async def _close_process(self) -> Exception | None:
        async with self._close_lock:
            if self._process_closed:
                return None
            try:
                await self._process.close()
            except Exception as exc:
                return exc
            self._process_closed = True
            return None

    def _snapshot_locked(self) -> TerminalSession:
        return TerminalSession(
            id=self._id,
            name=self._name,
            status=self._status,
            created_at=self._created_at,
            rows=self._rows,
            columns=self._columns,
            exit_code=self._exit_code,
            error=self._error or self._cleanup_error,
            history_truncated=self._buffer.truncated,
        )

    def _publish_locked(self, event: TerminalAttachmentEvent) -> None:
        for attachment in tuple(self._attachments):
            try:
                attachment._queue.put_nowait(event)
            except asyncio.QueueFull:
                self._attachments.discard(attachment)
                attachment._signal(_OVERFLOW)

    def _require_present_locked(self) -> None:
        if self._deleted:
            raise TerminalNotFoundError("Terminal Session not found.")

    def _require_running_locked(self) -> None:
        self._require_present_locked()
        if self._status is not TerminalStatus.RUNNING:
            raise TerminalConflictError("Terminal Session is not running.")


class TerminalService:
    """Own interactive shell PTYs independently from browser attachments."""

    def __init__(
        self,
        process_factory: TerminalProcessFactoryPort,
        *,
        home: Path | None = None,
        env: Mapping[str, str] | None = None,
        output_limit: int = DEFAULT_OUTPUT_BYTES,
        attachment_queue_size: int = DEFAULT_ATTACHMENT_QUEUE_SIZE,
        clock: Callable[[], int] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if output_limit <= 0:
            raise ValueError("Terminal output limit must be positive.")
        if attachment_queue_size <= 0:
            raise ValueError("Terminal attachment queue size must be positive.")
        self._process_factory = process_factory
        self._home = home or Path.home()
        self._env = dict(env) if env is not None else dict(os.environ)
        self._output_limit = output_limit
        self._attachment_queue_size = attachment_queue_size
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._sessions: dict[str, _TerminalSession] = {}
        self._next_name = 1
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._closing = False

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            if self._closing:
                raise TerminalUnavailableError("Terminal Sessions is shutting down.")
            self._started = True

    async def create_session(self) -> TerminalSession:
        async with self._lifecycle_lock:
            self._require_available_locked()
            try:
                process = await self._process_factory.spawn(
                    cwd=self._home,
                    env=self._env,
                    rows=DEFAULT_ROWS,
                    columns=DEFAULT_COLUMNS,
                )
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not start the system shell."
                ) from exc
            session_id = self._id_factory()
            session = _TerminalSession(
                session_id=session_id,
                name=f"Terminal {self._next_name}",
                created_at=self._clock(),
                process=process,
                output_limit=self._output_limit,
                attachment_queue_size=self._attachment_queue_size,
            )
            self._next_name += 1
            self._sessions[session_id] = session
            session.start()
        return await session.snapshot()

    async def list_sessions(self) -> tuple[TerminalSession, ...]:
        async with self._lifecycle_lock:
            sessions = tuple(self._sessions.values())
        snapshots = await asyncio.gather(*(session.snapshot() for session in sessions))
        return tuple(sorted(snapshots, key=lambda item: item.created_at, reverse=True))

    async def get_session(self, session_id: str) -> TerminalSession:
        return await (await self._session(session_id)).snapshot()

    async def rename_session(self, session_id: str, name: str) -> TerminalSession:
        return await (await self._session(session_id)).rename(_valid_name(name))

    async def attach(self, session_id: str) -> TerminalAttachmentPort:
        return await (await self._session(session_id)).attach()

    async def write(self, session_id: str, data: bytes) -> None:
        if len(data) > MAX_INPUT_BYTES:
            raise TerminalValidationError(
                f"Terminal input cannot exceed {MAX_INPUT_BYTES} bytes."
            )
        await (await self._session(session_id)).write(data)

    async def resize(
        self, session_id: str, *, rows: int, columns: int
    ) -> TerminalSession:
        if not 1 <= rows <= MAX_ROWS or not 1 <= columns <= MAX_COLUMNS:
            raise TerminalValidationError("Terminal dimensions are out of range.")
        return await (await self._session(session_id)).resize(
            rows=rows, columns=columns
        )

    async def stop_session(self, session_id: str) -> TerminalSession:
        return await (await self._session(session_id)).stop()

    async def delete_session(self, session_id: str) -> None:
        async with self._lifecycle_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise TerminalNotFoundError("Terminal Session not found.")
        await session.dispose()
        async with self._lifecycle_lock:
            if self._sessions.get(session_id) is session:
                del self._sessions[session_id]

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closing and not self._sessions:
                return
            self._closing = True
            sessions = tuple(self._sessions.items())

        failures: list[Exception] = []
        for session_id, session in sessions:
            try:
                await session.dispose()
            except Exception as exc:
                failures.append(exc)
                continue
            async with self._lifecycle_lock:
                if self._sessions.get(session_id) is session:
                    del self._sessions[session_id]

        if failures:
            raise TerminalUnavailableError(
                f"Could not close {len(failures)} Terminal Session(s)."
            ) from failures[0]
        async with self._lifecycle_lock:
            self._started = False

    async def _session(self, session_id: str) -> _TerminalSession:
        async with self._lifecycle_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise TerminalNotFoundError("Terminal Session not found.")
        return session

    def _require_available_locked(self) -> None:
        if not self._started or self._closing:
            raise TerminalUnavailableError("Terminal Sessions is unavailable.")


def _valid_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise TerminalValidationError("Terminal name cannot be empty.")
    if len(name) > MAX_NAME_LENGTH:
        raise TerminalValidationError(
            f"Terminal name cannot exceed {MAX_NAME_LENGTH} characters."
        )
    return name


def _cleanup_error_message(failure: Exception | None) -> str | None:
    if failure is None:
        return None
    return f"Terminal process cleanup failed ({type(failure).__name__})."

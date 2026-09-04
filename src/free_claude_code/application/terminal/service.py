"""Server-owned lifecycle for terminal-engine sessions and browser views."""

import asyncio
import os
import shutil
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from .models import (
    TerminalAttachmentEvent,
    TerminalAttachmentSnapshot,
    TerminalClientRole,
    TerminalConflictError,
    TerminalDeletedEvent,
    TerminalNotFoundError,
    TerminalOutputEvent,
    TerminalResetEvent,
    TerminalSession,
    TerminalStateEvent,
    TerminalStatus,
    TerminalUnavailableError,
    TerminalValidationError,
)
from .ports import (
    TerminalAttachmentPort,
    TerminalClientPort,
    TerminalEngineHostPort,
    TerminalEngineSessionPort,
)

DEFAULT_ROWS = 24
DEFAULT_COLUMNS = 80
MAX_ROWS = 500
MAX_COLUMNS = 1_000
MAX_INPUT_BYTES = 64 * 1024
MAX_NAME_LENGTH = 100
_ATTACHMENT_QUEUE_SIZE = 256


class _Closed:
    pass


_CLOSED = _Closed()
type _AttachmentQueueItem = TerminalAttachmentEvent | _Closed


class _TerminalAttachment(TerminalAttachmentPort):
    def __init__(
        self,
        owner: _TerminalSession,
        *,
        attachment_id: int,
        initial: TerminalAttachmentSnapshot,
        client: TerminalClientPort | None,
        attach_order: int,
        rows: int,
        columns: int,
    ) -> None:
        self._owner = owner
        self._id = attachment_id
        self._initial = initial
        self._client = client
        self._role = initial.role
        self._attach_order = attach_order
        self._interaction_order = 0
        self._rows = rows
        self._columns = columns
        self._queue: asyncio.Queue[_AttachmentQueueItem] = asyncio.Queue(
            maxsize=_ATTACHMENT_QUEUE_SIZE
        )
        self._reader_task: asyncio.Task[None] | None = None
        self._client_generation = 0
        self._overflowed = False
        self._closed = False

    @property
    def initial(self) -> TerminalAttachmentSnapshot:
        return self._initial

    @property
    def id(self) -> int:
        return self._id

    @property
    def role(self) -> TerminalClientRole:
        return self._role

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def attach_order(self) -> int:
        return self._attach_order

    @property
    def interaction_order(self) -> int:
        return self._interaction_order

    @property
    def client(self) -> TerminalClientPort | None:
        return self._client

    def __aiter__(self) -> AsyncIterator[TerminalAttachmentEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[TerminalAttachmentEvent]:
        while True:
            item = await self._queue.get()
            if isinstance(item, _Closed):
                return
            yield item

    async def claim(self) -> None:
        await self._owner.claim(self)

    async def write(self, data: str) -> None:
        await self._owner.write(self, data)

    async def resize(self, *, rows: int, columns: int) -> None:
        await self._owner.resize(self, rows=rows, columns=columns)

    async def aclose(self) -> None:
        if self._closed:
            return
        await self._owner.detach(self)

    def start_reader(self) -> None:
        client = self._client
        if client is None or self._closed:
            return
        self._client_generation += 1
        generation = self._client_generation
        self._reader_task = asyncio.create_task(
            self._read_client(client, generation),
            name=f"terminal-client-reader-{self._owner.id}-{self._id}",
        )

    async def replace_client(
        self,
        client: TerminalClientPort,
        *,
        role: TerminalClientRole,
        output: bytes,
    ) -> None:
        try:
            await self._close_client()
        except Exception:
            with suppress(Exception):
                await client.close()
            raise
        if self._closed:
            await client.close()
            return
        self._client = client
        self._role = role
        await self.emit(TerminalResetEvent(output=output, role=role))
        self.start_reader()

    async def close_client(self) -> None:
        await self._close_client()

    async def close_from_owner(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._close_client()
        finally:
            self._signal_closed()

    async def fail_from_owner(self) -> None:
        await self.close_from_owner()

    def set_interaction_order(self, value: int) -> None:
        self._interaction_order = value

    def set_dimensions(self, *, rows: int, columns: int) -> None:
        self._rows = rows
        self._columns = columns

    async def emit(self, event: TerminalAttachmentEvent) -> None:
        if self._closed or self._overflowed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._overflowed = True
            asyncio.create_task(
                self._owner.client_failed(self),
                name=f"terminal-overflow-close-{self._owner.id}-{self._id}",
            )

    async def _read_client(self, client: TerminalClientPort, generation: int) -> None:
        try:
            while True:
                data = await client.read()
                if not data:
                    break
                await self.emit(TerminalOutputEvent(data))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            if (
                not self._closed
                and generation == self._client_generation
                and client is self._client
            ):
                await self._owner.client_failed(self)

    async def _close_client(self) -> None:
        self._client_generation += 1
        task = self._reader_task
        self._reader_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    def _signal_closed(self) -> None:
        try:
            self._queue.put_nowait(_CLOSED)
            return
        except asyncio.QueueFull:
            pass
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(_CLOSED)


class _TerminalSession:
    def __init__(
        self,
        *,
        session_id: str,
        name: str,
        created_at: int,
        engine: TerminalEngineSessionPort,
    ) -> None:
        self._id = session_id
        self._name = name
        self._created_at = created_at
        self._engine = engine
        self._status = TerminalStatus.RUNNING
        self._rows = DEFAULT_ROWS
        self._columns = DEFAULT_COLUMNS
        self._exit_code: int | None = None
        self._error: str | None = None
        self._final_output = b""
        self._deleted = False
        self._attachments: dict[int, _TerminalAttachment] = {}
        self._controller_id: int | None = None
        self._next_attachment_id = 1
        self._next_attach_order = 1
        self._next_interaction_order = 1
        self._control_lock = asyncio.Lock()
        self._finalize_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None

    @property
    def id(self) -> str:
        return self._id

    def start(self) -> None:
        if self._monitor_task is not None:
            raise RuntimeError("Terminal session monitor already started.")
        self._monitor_task = asyncio.create_task(
            self._monitor_root(), name=f"terminal-root-monitor-{self._id}"
        )

    async def snapshot(self) -> TerminalSession:
        async with self._control_lock:
            return self._snapshot_locked()

    async def rename(self, name: str) -> TerminalSession:
        async with self._control_lock:
            self._require_present_locked()
            self._name = name
            snapshot = self._snapshot_locked()
            attachments = tuple(self._attachments.values())
        await _emit_all(attachments, TerminalStateEvent(snapshot))
        return snapshot

    async def attach(self, *, rows: int, columns: int) -> _TerminalAttachment:
        async with self._control_lock:
            self._require_present_locked()
            role = (
                TerminalClientRole.CONTROLLER
                if self._status is TerminalStatus.RUNNING
                and self._controller_id is None
                else TerminalClientRole.OBSERVER
            )
            client: TerminalClientPort | None = None
            output = self._final_output
            if self._status is TerminalStatus.RUNNING:
                if role is TerminalClientRole.CONTROLLER:
                    self._rows = rows
                    self._columns = columns
                    await self._engine.resize(rows, columns)
                output = (await self._engine.snapshot()).scrollback
                client = await self._engine.open_client(
                    role,
                    rows=self._rows,
                    columns=self._columns,
                )
            attachment_id = self._next_attachment_id
            self._next_attachment_id += 1
            attach_order = self._next_attach_order
            self._next_attach_order += 1
            attachment = _TerminalAttachment(
                self,
                attachment_id=attachment_id,
                initial=TerminalAttachmentSnapshot(
                    session=self._snapshot_locked(), output=output, role=role
                ),
                client=client,
                attach_order=attach_order,
                rows=rows,
                columns=columns,
            )
            self._attachments[attachment_id] = attachment
            if role is TerminalClientRole.CONTROLLER:
                self._controller_id = attachment_id
            attachment.start_reader()
            return attachment

    async def detach(self, attachment: _TerminalAttachment) -> None:
        promotion: _TerminalAttachment | None = None
        async with self._control_lock:
            if self._attachments.pop(attachment.id, None) is not attachment:
                await attachment.close_from_owner()
                return
            was_controller = self._controller_id == attachment.id
            if was_controller:
                self._controller_id = None
                promotion = self._promotion_candidate_locked()
            await attachment.close_from_owner()
            if promotion is not None and self._status is TerminalStatus.RUNNING:
                await self._promote_after_disconnect_locked(promotion)

    async def claim(self, attachment: _TerminalAttachment) -> None:
        async with self._control_lock:
            self._require_running_locked()
            self._require_attachment_locked(attachment)
            attachment.set_interaction_order(self._next_interaction_order)
            self._next_interaction_order += 1
            if self._controller_id == attachment.id:
                return
            await self._promote_locked(attachment)

    async def write(self, attachment: _TerminalAttachment, data: str) -> None:
        encoded_size = len(data.encode("utf-8"))
        if encoded_size > MAX_INPUT_BYTES:
            raise TerminalValidationError(
                f"Terminal input cannot exceed {MAX_INPUT_BYTES} bytes."
            )
        async with self._control_lock:
            self._require_running_locked()
            self._require_attachment_locked(attachment)
            client = attachment.client
            if client is None:
                raise TerminalUnavailableError("Terminal view is disconnected.")
            try:
                await client.write(data)
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not write to the terminal view."
                ) from exc

    async def resize(
        self,
        attachment: _TerminalAttachment,
        *,
        rows: int,
        columns: int,
    ) -> None:
        _validate_dimensions(rows, columns)
        async with self._control_lock:
            self._require_running_locked()
            self._require_attachment_locked(attachment)
            attachment.set_dimensions(rows=rows, columns=columns)
            if self._controller_id != attachment.id:
                return
            await self._resize_locked(rows=rows, columns=columns)
            snapshot = self._snapshot_locked()
            attachments = tuple(self._attachments.values())
        await _emit_all(attachments, TerminalStateEvent(snapshot))

    async def client_failed(self, attachment: _TerminalAttachment) -> None:
        async with self._control_lock:
            if self._attachments.pop(attachment.id, None) is not attachment:
                return
            was_controller = self._controller_id == attachment.id
            if was_controller:
                self._controller_id = None
            await attachment.fail_from_owner()
            if was_controller and self._status is TerminalStatus.RUNNING:
                candidate = self._promotion_candidate_locked()
                if candidate is not None:
                    await self._promote_after_disconnect_locked(candidate)

    async def stop(self) -> TerminalSession:
        await self._finalize(exit_code=None)
        return await self.snapshot()

    async def dispose(self) -> None:
        await self._finalize(exit_code=None)
        async with self._control_lock:
            if self._deleted:
                return
            self._deleted = True
            attachments = tuple(self._attachments.values())
            self._attachments.clear()
            self._controller_id = None
        await _emit_all(attachments, TerminalDeletedEvent())
        await asyncio.gather(
            *(attachment.close_from_owner() for attachment in attachments),
            return_exceptions=True,
        )

    async def _promote_locked(self, claimant: _TerminalAttachment) -> None:
        self._require_attachment_locked(claimant)
        previous = self._attachments.get(self._controller_id or -1)
        previous_rows = self._rows
        previous_columns = self._columns
        next_controller: TerminalClientPort | None = None
        next_observer: TerminalClientPort | None = None
        resized = False
        try:
            output = (await self._engine.snapshot()).scrollback
            await self._engine.resize(claimant.rows, claimant.columns)
            resized = True
            next_controller = await self._engine.open_client(
                TerminalClientRole.CONTROLLER,
                rows=claimant.rows,
                columns=claimant.columns,
            )
            if previous is not None:
                next_observer = await self._engine.open_client(
                    TerminalClientRole.OBSERVER,
                    rows=claimant.rows,
                    columns=claimant.columns,
                )
        except Exception as exc:
            await _close_clients(next_controller, next_observer)
            if resized:
                with suppress(Exception):
                    await self._engine.resize(previous_rows, previous_columns)
            raise TerminalUnavailableError(
                "Could not transfer terminal control."
            ) from exc

        try:
            await claimant.replace_client(
                next_controller,
                role=TerminalClientRole.CONTROLLER,
                output=output,
            )
            next_controller = None
            if previous is not None and next_observer is not None:
                await previous.replace_client(
                    next_observer,
                    role=TerminalClientRole.OBSERVER,
                    output=output,
                )
                next_observer = None
        except Exception as exc:
            await _close_clients(next_controller, next_observer)
            with suppress(Exception):
                await self._engine.resize(previous_rows, previous_columns)
            affected = tuple(
                attachment
                for attachment in (claimant, previous)
                if attachment is not None
            )
            for attachment in affected:
                self._attachments.pop(attachment.id, None)
            self._controller_id = None
            await asyncio.gather(
                *(attachment.close_from_owner() for attachment in affected),
                return_exceptions=True,
            )
            raise TerminalUnavailableError(
                "Could not transfer terminal control."
            ) from exc
        self._controller_id = claimant.id
        self._rows = claimant.rows
        self._columns = claimant.columns
        snapshot = self._snapshot_locked()
        await _emit_all(tuple(self._attachments.values()), TerminalStateEvent(snapshot))

    async def _promote_after_disconnect_locked(
        self, candidate: _TerminalAttachment
    ) -> None:
        try:
            await self._promote_locked(candidate)
        except TerminalUnavailableError:
            if self._attachments.pop(candidate.id, None) is candidate:
                await candidate.fail_from_owner()

    async def _resize_locked(self, *, rows: int, columns: int) -> None:
        await self._engine.resize(rows, columns)
        self._rows = rows
        self._columns = columns

    async def _monitor_root(self) -> None:
        try:
            exit_code = await self._engine.wait_root()
            await self._finalize(exit_code=exit_code)
        except asyncio.CancelledError:
            raise
        except TerminalUnavailableError:
            pass
        except Exception as exc:
            async with self._control_lock:
                if self._status is TerminalStatus.RUNNING:
                    self._error = (
                        f"Terminal process ended unexpectedly ({type(exc).__name__})."
                    )
            with suppress(TerminalUnavailableError):
                await self._finalize(exit_code=None)

    async def _finalize(self, *, exit_code: int | None) -> None:
        async with self._finalize_lock:
            async with self._control_lock:
                if self._deleted:
                    return
                if self._status is TerminalStatus.EXITED:
                    return
                self._status = TerminalStatus.STOPPING
                stopping = self._snapshot_locked()
                attachments = tuple(self._attachments.values())
            await _emit_all(attachments, TerminalStateEvent(stopping))

            output = self._final_output
            retention_error: str | None = None
            try:
                output = (await self._engine.snapshot()).rendered
            except Exception as exc:
                retention_error = (
                    f"Final terminal output is unavailable ({type(exc).__name__})."
                )

            await asyncio.gather(
                *(attachment.close_client() for attachment in attachments),
                return_exceptions=True,
            )
            try:
                await self._engine.terminate_tree()
                await self._engine.close()
            except Exception as exc:
                async with self._control_lock:
                    self._error = (
                        f"Terminal process cleanup failed ({type(exc).__name__})."
                    )
                raise TerminalUnavailableError(
                    "Could not stop the terminal process tree."
                ) from exc

            async with self._control_lock:
                self._status = TerminalStatus.EXITED
                self._exit_code = exit_code
                self._error = retention_error or self._error
                self._final_output = output
                self._controller_id = None
                exited = self._snapshot_locked()
                attachments = tuple(self._attachments.values())
            await _emit_all(attachments, TerminalStateEvent(exited))

    def _snapshot_locked(self) -> TerminalSession:
        return TerminalSession(
            id=self._id,
            name=self._name,
            status=self._status,
            created_at=self._created_at,
            rows=self._rows,
            columns=self._columns,
            exit_code=self._exit_code,
            error=self._error,
        )

    def _promotion_candidate_locked(self) -> _TerminalAttachment | None:
        if not self._attachments:
            return None
        return max(
            self._attachments.values(),
            key=lambda attachment: (
                attachment.interaction_order > 0,
                attachment.interaction_order,
                -attachment.attach_order,
            ),
        )

    def _require_present_locked(self) -> None:
        if self._deleted:
            raise TerminalNotFoundError("Terminal Session not found.")

    def _require_running_locked(self) -> None:
        self._require_present_locked()
        if self._status is not TerminalStatus.RUNNING:
            raise TerminalConflictError("Terminal Session is not running.")

    def _require_attachment_locked(self, attachment: _TerminalAttachment) -> None:
        if self._attachments.get(attachment.id) is not attachment:
            raise TerminalConflictError("Terminal view is no longer attached.")


class TerminalService:
    """Own Terminal Sessions while a runtime-provided engine owns terminal state."""

    def __init__(
        self,
        engine_host: TerminalEngineHostPort,
        *,
        home: Path | None = None,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], int] | None = None,
        id_factory: Callable[[], str] | None = None,
        shell_factory: Callable[[Mapping[str, str]], Sequence[str]] | None = None,
    ) -> None:
        self._engine_host = engine_host
        self._home = home or Path.home()
        self._env = dict(env) if env is not None else dict(os.environ)
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._shell_factory = shell_factory or _system_shell_argv
        self._sessions: dict[str, _TerminalSession] = {}
        self._next_name = 1
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._engine_started = False
        self._closing = False
        self._availability_error: str | None = None

    @property
    def availability_error(self) -> str | None:
        return self._availability_error

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            if self._closing:
                raise TerminalUnavailableError("Terminal Sessions is shutting down.")
            try:
                await self._engine_host.start()
            except Exception:
                self._availability_error = (
                    "Terminal Sessions is unavailable. Rerun the FCC installer and "
                    "restart FCC."
                )
            else:
                self._engine_started = True
                self._availability_error = None
            self._started = True

    async def create_session(self) -> TerminalSession:
        async with self._lifecycle_lock:
            self._require_available_locked()
            session_id = self._id_factory()
            command = self._shell_factory(self._env)
            try:
                engine = await self._engine_host.create_session(
                    session_name=f"fcc-{session_id}",
                    command=command,
                    cwd=self._home,
                    env=self._env,
                    rows=DEFAULT_ROWS,
                    columns=DEFAULT_COLUMNS,
                )
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not start the system shell."
                ) from exc
            session = _TerminalSession(
                session_id=session_id,
                name=f"Terminal {self._next_name}",
                created_at=self._clock(),
                engine=engine,
            )
            self._sessions[session_id] = session
            self._next_name += 1
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

    async def attach(
        self, session_id: str, *, rows: int, columns: int
    ) -> TerminalAttachmentPort:
        _validate_dimensions(rows, columns)
        return await (await self._session(session_id)).attach(
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
            if self._closing and not self._sessions and not self._engine_started:
                return
            self._closing = True
            sessions = tuple(self._sessions.items())

        results = await asyncio.gather(
            *(session.dispose() for _, session in sessions),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        async with self._lifecycle_lock:
            for (session_id, session), result in zip(sessions, results, strict=True):
                if isinstance(result, BaseException):
                    failures.append(result)
                elif self._sessions.get(session_id) is session:
                    del self._sessions[session_id]

        if failures:
            raise TerminalUnavailableError(
                f"Could not close {len(failures)} Terminal Session(s)."
            ) from failures[0]
        if self._engine_started:
            try:
                await self._engine_host.close()
            except Exception as exc:
                raise TerminalUnavailableError(
                    "Could not close the terminal engine."
                ) from exc
            self._engine_started = False
        async with self._lifecycle_lock:
            self._started = False

    async def _session(self, session_id: str) -> _TerminalSession:
        async with self._lifecycle_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise TerminalNotFoundError("Terminal Session not found.")
        return session

    def _require_available_locked(self) -> None:
        if (
            not self._started
            or self._closing
            or not self._engine_started
            or self._availability_error is not None
        ):
            raise TerminalUnavailableError(
                self._availability_error or "Terminal Sessions is unavailable."
            )


async def _emit_all(
    attachments: tuple[_TerminalAttachment, ...], event: TerminalAttachmentEvent
) -> None:
    await asyncio.gather(
        *(attachment.emit(event) for attachment in attachments),
        return_exceptions=True,
    )


async def _close_clients(*clients: TerminalClientPort | None) -> None:
    await asyncio.gather(
        *(client.close() for client in clients if client is not None),
        return_exceptions=True,
    )


def _validate_dimensions(rows: int, columns: int) -> None:
    if not 1 <= rows <= MAX_ROWS or not 1 <= columns <= MAX_COLUMNS:
        raise TerminalValidationError("Terminal dimensions are out of range.")


def _valid_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise TerminalValidationError("Terminal name cannot be empty.")
    if len(name) > MAX_NAME_LENGTH:
        raise TerminalValidationError(
            f"Terminal name cannot exceed {MAX_NAME_LENGTH} characters."
        )
    return name


def _system_shell_argv(env: Mapping[str, str]) -> Sequence[str]:
    path = env.get("PATH")
    if os.name == "nt":
        for command in ("pwsh", "powershell"):
            executable = shutil.which(command, path=path)
            if executable is not None:
                return (executable, "-NoLogo")
        comspec = env.get("COMSPEC")
        if comspec and Path(comspec).is_file():
            return (comspec,)
        raise FileNotFoundError("No supported Windows system shell was found.")

    configured = env.get("SHELL")
    if configured:
        shell = Path(configured).expanduser()
        if shell.is_file() and os.access(shell, os.X_OK):
            return (str(shell),)
    fallback = Path("/bin/sh")
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return (str(fallback),)
    raise FileNotFoundError("No supported POSIX system shell was found.")

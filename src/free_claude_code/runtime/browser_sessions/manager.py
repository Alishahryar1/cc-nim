"""Runtime lifecycle owner for browser-based native harness terminals."""

import asyncio
import copy
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from free_claude_code.application.browser_sessions import (
    BrowserSessionConflictError,
    BrowserSessionError,
    BrowserSessionHarness,
    BrowserSessionNotFoundError,
    BrowserSessionSnapshot,
    BrowserSessionState,
    BrowserSessionUnavailableError,
    BrowserSessionValidationError,
    BrowserSessionView,
    BrowserTerminalPort,
    TerminalEvent,
)

from .drivers import HarnessDriverRegistry, terminal_environment
from .pty import TerminalProcess, TerminalProcessFactory, TerminalSpawner
from .store import (
    BrowserSessionStore,
    SessionCatalog,
    SessionRecord,
    SessionStoreError,
)

OUTPUT_RING_LIMIT_BYTES = 512 * 1024
ATTACHMENT_QUEUE_LIMIT = 256
DEFAULT_COLUMNS = 120
DEFAULT_ROWS = 32
PROCESS_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class _SessionRuntime:
    state: BrowserSessionState = BrowserSessionState.STOPPED
    detail: str | None = None
    process: TerminalProcess | None = None
    reader_task: asyncio.Task[None] | None = None
    output: bytearray = field(default_factory=bytearray)
    attachment: _TerminalAttachment | None = None
    stop_requested: bool = False


class _TerminalAttachment(BrowserTerminalPort):
    def __init__(
        self,
        session_id: str,
        token: str,
        *,
        write_callback: Callable[[str, str, str], Awaitable[None]],
        resize_callback: Callable[[str, str, int, int], Awaitable[None]],
        close_callback: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self.session_id = session_id
        self.token = token
        self._write_callback = write_callback
        self._resize_callback = resize_callback
        self._close_callback = close_callback
        self._queue: asyncio.Queue[TerminalEvent] = asyncio.Queue(
            maxsize=ATTACHMENT_QUEUE_LIMIT
        )
        self._closed = False

    async def receive(self) -> TerminalEvent:
        return await self._queue.get()

    async def write(self, data: str) -> None:
        if self._closed:
            raise BrowserSessionConflictError("Terminal attachment is closed.")
        await self._write_callback(self.session_id, self.token, data)

    async def resize(self, columns: int, rows: int) -> None:
        if self._closed:
            raise BrowserSessionConflictError("Terminal attachment is closed.")
        await self._resize_callback(self.session_id, self.token, columns, rows)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._close_callback(self.session_id, self.token)

    def publish(self, event: TerminalEvent) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.finish(
                TerminalEvent(kind="error", detail="Terminal connection was too slow.")
            )
            return False
        return True

    def finish(self, event: TerminalEvent) -> None:
        if self._closed:
            return
        self._closed = True
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(event)


class BrowserSessionManager:
    """Persist session identities and own all live harness process resources."""

    def __init__(
        self,
        store: BrowserSessionStore,
        *,
        drivers: HarnessDriverRegistry | None = None,
        process_factory: TerminalSpawner | None = None,
    ) -> None:
        self._store = store
        self._drivers = drivers or HarnessDriverRegistry()
        self._process_factory = process_factory or TerminalProcessFactory()
        self._lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._close_complete = False
        self._deleting_sessions: set[str] = set()
        self._start_tasks: dict[str, asyncio.Task[object]] = {}
        self._unavailable_message: str | None = None
        try:
            self._catalog = store.load()
        except SessionStoreError as exc:
            self._catalog = SessionCatalog()
            self._unavailable_message = str(exc)
        self._runtime = {
            session.session_id: _SessionRuntime() for session in self._catalog.sessions
        }

    async def snapshot(self) -> BrowserSessionSnapshot:
        async with self._lock:
            return BrowserSessionSnapshot(
                available=self._unavailable_message is None and not self._closed,
                message=self._unavailable_message,
                harnesses=self._drivers.availability(),
                sessions=tuple(
                    self._session_view(session)
                    for session in reversed(self._catalog.sessions)
                ),
            )

    async def create_session(
        self,
        path: str,
        harness: BrowserSessionHarness,
        name: str | None = None,
    ) -> BrowserSessionView:
        canonical = _canonical_directory(path)
        driver = self._drivers.driver(harness)
        async with self._lock:
            self._ensure_available()
            clean_name = (
                _default_session_name(self._catalog, canonical)
                if name is None
                else _session_name(name)
            )
            _ensure_unique_name(self._catalog, canonical, clean_name)
            availability = driver.availability()
            if not availability.available:
                raise BrowserSessionUnavailableError(
                    availability.message or "That harness is unavailable."
                )
            session_id = str(uuid4())
            updated = copy.deepcopy(self._catalog)
            updated.sessions.append(
                SessionRecord(
                    session_id=session_id,
                    path=str(canonical),
                    name=clean_name,
                    harness=harness,
                )
            )
            self._commit(updated)
            self._runtime[session_id] = _SessionRuntime()
        return await self.start_session(session_id)

    async def rename_session(self, session_id: str, name: str) -> BrowserSessionView:
        clean_name = _session_name(name)
        async with self._lock:
            self._ensure_available()
            session = self._session(session_id)
            _ensure_unique_name(
                self._catalog,
                Path(session.path),
                clean_name,
                except_id=session_id,
            )
            updated = copy.deepcopy(self._catalog)
            updated_session = _session_in(updated, session_id)
            updated_session.name = clean_name
            self._commit(updated)
            session.name = clean_name
            return self._session_view(session)

    async def start_session(self, session_id: str) -> BrowserSessionView:
        async with self._lock:
            self._ensure_available()
            session = self._session(session_id)
            if session_id in self._deleting_sessions:
                raise BrowserSessionConflictError("That session is being deleted.")
            runtime = self._runtime[session_id]
            if runtime.state not in {
                BrowserSessionState.STOPPED,
                BrowserSessionState.EXITED,
                BrowserSessionState.FAILED,
            }:
                raise BrowserSessionConflictError(
                    f"Session is already {runtime.state.value}."
                )
            project_path = _existing_session_path(session.path)
            driver = self._drivers.driver(session.harness)
            try:
                command = driver.command(
                    session.session_id,
                    started_once=session.started_once,
                )
            except BrowserSessionUnavailableError as exc:
                runtime.state = BrowserSessionState.FAILED
                runtime.detail = str(exc)
                self._publish_state(runtime)
                return self._session_view(session)
            runtime.state = BrowserSessionState.STARTING
            runtime.detail = None
            runtime.output.clear()
            runtime.stop_requested = False
            start_task = asyncio.current_task()
            if start_task is not None:
                self._start_tasks[session_id] = start_task
                start_task.add_done_callback(
                    lambda completed, target=session_id: self._forget_start_task(
                        target, completed
                    )
                )
            self._publish_state(runtime)

        try:
            process = await self._spawn_terminal(
                command,
                cwd=str(project_path),
                env=terminal_environment(),
                columns=DEFAULT_COLUMNS,
                rows=DEFAULT_ROWS,
            )
        except asyncio.CancelledError:
            await self._mark_cancelled_start(session_id)
            raise
        except Exception as exc:
            async with self._lock:
                runtime = self._runtime[session_id]
                runtime.state = BrowserSessionState.FAILED
                runtime.detail = (
                    f"Could not start {session.harness.value} ({type(exc).__name__})."
                )
                self._publish_state(runtime)
                return self._session_view(session)

        try:
            async with self._lock:
                if self._closed or session_id in self._deleting_sessions:
                    should_terminate = True
                else:
                    should_terminate = False
                    runtime = self._runtime[session_id]
                    runtime.process = process
                    if not session.started_once:
                        updated = copy.deepcopy(self._catalog)
                        updated_session = _session_in(updated, session_id)
                        updated_session.started_once = True
                        try:
                            self._commit(updated)
                        except BrowserSessionUnavailableError:
                            should_terminate = True
                        else:
                            session.started_once = True
                    if not should_terminate:
                        runtime.state = BrowserSessionState.RUNNING
                        runtime.reader_task = asyncio.create_task(
                            self._read_terminal(session_id, process)
                        )
                        self._publish_state(runtime)
                        return self._session_view(session)

            await self._terminate_process(process)
        except asyncio.CancelledError:
            await asyncio.shield(self._terminate_process(process))
            await self._mark_cancelled_start(session_id, process)
            raise
        async with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is not None:
                runtime.process = None
                runtime.state = BrowserSessionState.FAILED
                runtime.detail = "Session startup could not be committed."
                self._publish_state(runtime)
            if self._closed:
                raise BrowserSessionUnavailableError(
                    "Browser Sessions is shutting down."
                )
            current = self._session(session_id)
            return self._session_view(current)

    async def stop_session(self, session_id: str) -> BrowserSessionView:
        async with self._lock:
            self._ensure_available(allow_closed=True)
            session = self._session(session_id)
            runtime = self._runtime[session_id]
            if runtime.state in {
                BrowserSessionState.STOPPED,
                BrowserSessionState.EXITED,
                BrowserSessionState.FAILED,
            }:
                runtime.state = BrowserSessionState.STOPPED
                runtime.detail = None
                self._publish_state(runtime)
                return self._session_view(session)
            if runtime.state == BrowserSessionState.STARTING:
                start_task = self._start_tasks.get(session_id)
            else:
                start_task = None
            if start_task is not None:
                process = None
                task = None
            else:
                process = runtime.process
                task = runtime.reader_task
                runtime.stop_requested = True
                runtime.state = BrowserSessionState.STOPPING
                self._publish_state(runtime)

        if start_task is not None:
            await asyncio.shield(start_task)
            return await self.stop_session(session_id)

        if process is not None:
            await self._terminate_process(process)
        if task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=PROCESS_STOP_TIMEOUT_SECONDS
                )
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        async with self._lock:
            session = self._session(session_id)
            runtime = self._runtime[session_id]
            runtime.process = None
            runtime.reader_task = None
            runtime.stop_requested = False
            runtime.state = BrowserSessionState.STOPPED
            runtime.detail = None
            self._publish_state(runtime)
            return self._session_view(session)

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._ensure_available()
            self._session(session_id)
            self._deleting_sessions.add(session_id)
        try:
            await self.stop_session(session_id)
            async with self._lock:
                updated = copy.deepcopy(self._catalog)
                updated.sessions = [
                    session
                    for session in updated.sessions
                    if session.session_id != session_id
                ]
                self._commit(updated)
                runtime = self._runtime.pop(session_id)
                if runtime.attachment is not None:
                    runtime.attachment.finish(TerminalEvent(kind="deleted"))
        finally:
            async with self._lock:
                self._deleting_sessions.discard(session_id)

    async def attach_terminal(self, session_id: str) -> BrowserTerminalPort:
        async with self._lock:
            self._ensure_available()
            self._session(session_id)
            runtime = self._runtime[session_id]
            if previous := runtime.attachment:
                previous.finish(TerminalEvent(kind="replaced"))
            attachment = _TerminalAttachment(
                session_id,
                uuid4().hex,
                write_callback=self._write_terminal,
                resize_callback=self._resize_terminal,
                close_callback=self._detach_terminal,
            )
            runtime.attachment = attachment
            attachment.publish(
                TerminalEvent(
                    kind="ready",
                    state=runtime.state,
                    detail=runtime.detail,
                )
            )
            if runtime.output:
                attachment.publish(
                    TerminalEvent(kind="output", data=bytes(runtime.output))
                )
            return attachment

    async def close(self) -> None:
        async with self._close_lock:
            async with self._lock:
                if self._close_complete:
                    return
                self._closed = True
            async with self._lock:
                session_ids = list(self._runtime)
            for session_id in session_ids:
                try:
                    await self.stop_session(session_id)
                except BrowserSessionError:
                    continue
            async with self._lock:
                for runtime in self._runtime.values():
                    if runtime.attachment is not None:
                        runtime.attachment.finish(
                            TerminalEvent(
                                kind="error", detail="FCC server is shutting down."
                            )
                        )
                        runtime.attachment = None
                self._close_complete = True

    async def _read_terminal(self, session_id: str, process: TerminalProcess) -> None:
        read_error: Exception | None = None
        try:
            while True:
                try:
                    chunk = await asyncio.to_thread(process.read, 4096)
                except EOFError:
                    break
                if not chunk:
                    break
                encoded = chunk.encode("utf-8", errors="replace")
                async with self._lock:
                    runtime = self._runtime.get(session_id)
                    if runtime is None or runtime.process is not process:
                        return
                    runtime.output.extend(encoded)
                    if len(runtime.output) > OUTPUT_RING_LIMIT_BYTES:
                        del runtime.output[:-OUTPUT_RING_LIMIT_BYTES]
                    if (
                        runtime.attachment is not None
                        and not runtime.attachment.publish(
                            TerminalEvent(kind="output", data=encoded)
                        )
                    ):
                        runtime.attachment = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            read_error = exc
        finally:
            if read_error is not None:
                await asyncio.to_thread(process.terminate_tree)
            try:
                exit_code = await asyncio.to_thread(process.wait)
            except Exception:
                exit_code = -1
            await asyncio.to_thread(process.close)
            async with self._lock:
                runtime = self._runtime.get(session_id)
                if runtime is not None and runtime.process is process:
                    runtime.process = None
                    runtime.reader_task = None
                    if runtime.stop_requested:
                        runtime.state = BrowserSessionState.STOPPED
                        runtime.detail = None
                    elif read_error is not None:
                        runtime.state = BrowserSessionState.FAILED
                        runtime.detail = (
                            "Terminal ended unexpectedly "
                            f"({type(read_error).__name__})."
                        )
                    elif exit_code == 0:
                        runtime.state = BrowserSessionState.EXITED
                        runtime.detail = "Harness exited."
                    else:
                        runtime.state = BrowserSessionState.FAILED
                        runtime.detail = f"Harness exited with code {exit_code}."
                    if runtime.attachment is not None:
                        runtime.attachment.publish(
                            TerminalEvent(
                                kind="exit",
                                state=runtime.state,
                                detail=runtime.detail,
                            )
                        )

    async def _terminate_process(self, process: TerminalProcess) -> None:
        await asyncio.to_thread(process.terminate_tree)
        await asyncio.to_thread(process.close)

    async def _spawn_terminal(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        columns: int,
        rows: int,
    ) -> TerminalProcess:
        spawn_task = asyncio.create_task(
            asyncio.to_thread(
                self._process_factory.spawn,
                command,
                cwd=cwd,
                env=env,
                columns=columns,
                rows=rows,
            )
        )
        try:
            return await asyncio.shield(spawn_task)
        except asyncio.CancelledError:
            process: TerminalProcess | None = None
            with suppress(Exception):
                process = await asyncio.shield(spawn_task)
            if process is not None:
                await asyncio.shield(self._terminate_process(process))
            raise

    async def _mark_cancelled_start(
        self, session_id: str, process: TerminalProcess | None = None
    ) -> None:
        async with self._lock:
            runtime = self._runtime.get(session_id)
            if runtime is not None and runtime.state == BrowserSessionState.STARTING:
                if process is not None and runtime.process is process:
                    runtime.process = None
                runtime.state = BrowserSessionState.FAILED
                runtime.detail = "Session start was cancelled."
                self._publish_state(runtime)

    async def _write_terminal(self, session_id: str, token: str, data: str) -> None:
        async with self._lock:
            runtime = self._active_attachment(session_id, token)
            if runtime.state != BrowserSessionState.RUNNING or runtime.process is None:
                raise BrowserSessionConflictError("Session is not running.")
            process = runtime.process
        await asyncio.to_thread(process.write, data)

    async def _resize_terminal(
        self, session_id: str, token: str, columns: int, rows: int
    ) -> None:
        async with self._lock:
            runtime = self._active_attachment(session_id, token)
            process = runtime.process
        if process is not None:
            await asyncio.to_thread(process.resize, columns, rows)

    async def _detach_terminal(self, session_id: str, token: str) -> None:
        async with self._lock:
            runtime = self._runtime.get(session_id)
            if (
                runtime is not None
                and runtime.attachment is not None
                and runtime.attachment.token == token
            ):
                runtime.attachment = None

    def _active_attachment(self, session_id: str, token: str) -> _SessionRuntime:
        runtime = self._runtime.get(session_id)
        if (
            runtime is None
            or runtime.attachment is None
            or runtime.attachment.token != token
        ):
            raise BrowserSessionConflictError("Terminal attachment was replaced.")
        return runtime

    def _forget_start_task(
        self, session_id: str, completed: asyncio.Task[object]
    ) -> None:
        if self._start_tasks.get(session_id) is completed:
            self._start_tasks.pop(session_id, None)

    def _publish_state(self, runtime: _SessionRuntime) -> None:
        if runtime.attachment is not None:
            runtime.attachment.publish(
                TerminalEvent(
                    kind="state",
                    state=runtime.state,
                    detail=runtime.detail,
                )
            )

    def _commit(self, catalog: SessionCatalog) -> None:
        try:
            self._store.save(catalog)
        except SessionStoreError as exc:
            raise BrowserSessionUnavailableError(str(exc)) from exc
        self._catalog = catalog

    def _ensure_available(self, *, allow_closed: bool = False) -> None:
        if self._unavailable_message is not None:
            raise BrowserSessionUnavailableError(self._unavailable_message)
        if self._closed and not allow_closed:
            raise BrowserSessionUnavailableError("Browser Sessions is shutting down.")

    def _session(self, session_id: str) -> SessionRecord:
        try:
            return _session_in(self._catalog, session_id)
        except KeyError as exc:
            raise BrowserSessionNotFoundError("Session not found.") from exc

    def _session_view(self, session: SessionRecord) -> BrowserSessionView:
        runtime = self._runtime[session.session_id]
        path = Path(session.path)
        return BrowserSessionView(
            session_id=session.session_id,
            name=session.name,
            harness=session.harness,
            state=runtime.state,
            project_name=path.name or path.anchor or str(path),
            path=session.path,
            detail=runtime.detail,
        )


def _canonical_directory(value: str) -> Path:
    if not value.strip():
        raise BrowserSessionValidationError("Folder path is required.")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BrowserSessionValidationError(
            "Folder path must be an existing local directory."
        ) from exc
    if not path.is_dir():
        raise BrowserSessionValidationError(
            "Folder path must be an existing local directory."
        )
    return path


def _existing_session_path(value: str) -> Path:
    try:
        return _canonical_directory(value)
    except BrowserSessionValidationError as exc:
        raise BrowserSessionConflictError(
            "The session folder no longer exists."
        ) from exc


def _same_directory(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(
            str(right.resolve())
        )


def _session_name(value: str) -> str:
    name = value.strip()
    if not 1 <= len(name) <= 80:
        raise BrowserSessionValidationError(
            "Session name must contain between 1 and 80 characters."
        )
    return name


def _ensure_unique_name(
    catalog: SessionCatalog,
    path: Path,
    name: str,
    *,
    except_id: str | None = None,
) -> None:
    if any(
        session.session_id != except_id
        and _same_directory(Path(session.path), path)
        and session.name.casefold() == name.casefold()
        for session in catalog.sessions
    ):
        raise BrowserSessionConflictError(
            "Session names must be unique within a folder."
        )


def _default_session_name(catalog: SessionCatalog, path: Path) -> str:
    existing = {
        session.name.casefold()
        for session in catalog.sessions
        if _same_directory(Path(session.path), path)
    }
    if "new session" not in existing:
        return "New session"
    suffix = 2
    while f"new session {suffix}" in existing:
        suffix += 1
    return f"New session {suffix}"


def _session_in(catalog: SessionCatalog, session_id: str) -> SessionRecord:
    return next(
        session for session in catalog.sessions if session.session_id == session_id
    )

"""Concrete stdio client for Codex Direct mode."""

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from loguru import logger

from free_claude_code.application.work import (
    CodexAppServerEvent,
    CodexAvailability,
    CodexCompatibilityError,
    CodexConnectionError,
    CodexConnectionLost,
    CodexControlCatalog,
    CodexDelivery,
    CodexInitialization,
    CodexInteractionKind,
    CodexInteractionRequest,
    CodexInteractionResponse,
    CodexNotification,
    CodexObjectPage,
    CodexProtocolError,
    CodexRequestError,
    CodexRequestId,
    CodexThreadHandle,
    CodexThreadSettings,
    CodexThreadSnapshot,
    CodexTurnHandle,
    CodexTurnSettings,
    CodexUnavailableError,
    CodexUnsupportedInteraction,
)
from free_claude_code.cli.launchers.codex import (
    build_codex_launcher_plan,
    codex_binary_name,
    codex_model_catalog_plan,
)
from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.version import package_version

_PROTOCOL_LINE_LIMIT = 16 * 1024 * 1024
_EVENT_QUEUE_LIMIT = 1024
_REQUEST_TIMEOUT_SECONDS = 30.0
_GRACEFUL_CLOSE_SECONDS = 5.0
_TERMINATE_SECONDS = 2.0
_VERSION_TIMEOUT_SECONDS = 5.0
_STDERR_CHUNK_BYTES = 8192
_METHOD_NOT_FOUND = -32601
_SERVER_REQUEST_RESOLVED = "serverRequest/resolved"
_IS_WINDOWS = os.name == "nt"

_INTERACTION_KINDS = {
    "item/commandExecution/requestApproval": CodexInteractionKind.COMMAND_APPROVAL,
    "item/fileChange/requestApproval": CodexInteractionKind.FILE_CHANGE_APPROVAL,
    "item/permissions/requestApproval": CodexInteractionKind.PERMISSION_APPROVAL,
    "item/tool/requestUserInput": CodexInteractionKind.USER_INPUT,
}
_INTERACTIVE_SERVER_METHODS = frozenset(_INTERACTION_KINDS)


@dataclass(frozen=True, slots=True)
class CodexAppServerProcessPlan:
    """Resolved child invocation; injectable so tests need no Codex install."""

    command: tuple[str, ...]
    env: dict[str, str]
    binary_path: str
    version: str | None


type CodexProcessPlanFactory = Callable[[], Awaitable[CodexAppServerProcessPlan]]
type CodexAvailabilityFactory = Callable[[], Awaitable[CodexAvailability]]


@dataclass(slots=True)
class _PendingCall:
    method: str
    future: asyncio.Future[JsonValue]


class _ConnectionState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


class _ServerRequestState(StrEnum):
    AWAITING_RESPONSE = "awaiting_response"
    RESPONSE_COMMITTED = "response_committed"


class _ServerRequestObservation(StrEnum):
    NEW = "new"
    PENDING_REPLAY = "pending_replay"
    COMMITTED_DUPLICATE = "committed_duplicate"
    CONFLICT = "conflict"


@dataclass(slots=True)
class _ServerRequestRecord:
    method: str
    params: JsonValue
    state: _ServerRequestState = _ServerRequestState.AWAITING_RESPONSE


@dataclass(frozen=True, slots=True)
class _StartSucceeded:
    connection: _Connection


@dataclass(frozen=True, slots=True)
class _StartFailed:
    error: Exception


type _StartOutcome = _StartSucceeded | _StartFailed


@dataclass(frozen=True, slots=True)
class _CleanupOutcome:
    error: CodexConnectionError | None


@dataclass(slots=True)
class _Connection:
    id: str
    state: _ConnectionState = _ConnectionState.STARTING
    initialize_response_received: bool = False
    ready_for_events: asyncio.Event = field(default_factory=asyncio.Event)
    process: asyncio.subprocess.Process | None = None
    version: str | None = None
    pending: dict[CodexRequestId, _PendingCall] = field(default_factory=dict)
    interactive_server_requests: dict[CodexRequestId, _ServerRequestRecord] = field(
        default_factory=dict
    )
    initialization: CodexInitialization | None = None
    startup_task: asyncio.Task[_StartOutcome] | None = None
    reader_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    terminal_error: Exception | None = None
    cleanup_task: asyncio.Task[_CleanupOutcome] | None = None
    cleanup_outcome: _CleanupOutcome | None = None

    def observe_server_request(
        self,
        *,
        request_id: CodexRequestId,
        method: str,
        params: JsonValue,
    ) -> _ServerRequestObservation:
        """Record a new request or classify a replay of an existing request."""

        record = self.interactive_server_requests.get(request_id)
        if record is None:
            self.interactive_server_requests[request_id] = _ServerRequestRecord(
                method=method,
                params=deepcopy(params),
            )
            return _ServerRequestObservation.NEW
        if record.method != method or record.params != params:
            return _ServerRequestObservation.CONFLICT
        if record.state is _ServerRequestState.AWAITING_RESPONSE:
            return _ServerRequestObservation.PENDING_REPLAY
        return _ServerRequestObservation.COMMITTED_DUPLICATE

    def require_answerable_server_request(
        self, request_id: CodexRequestId
    ) -> _ServerRequestRecord:
        """Return the request only while a response may still be written."""

        record = self.interactive_server_requests.get(request_id)
        if record is None or record.state is not _ServerRequestState.AWAITING_RESPONSE:
            raise CodexConnectionError(
                "The Codex request is no longer awaiting a response."
            )
        return record

    def commit_server_response(self, request_id: CodexRequestId) -> None:
        """Mark a response committed immediately after its successful write."""

        record = self.require_answerable_server_request(request_id)
        record.state = _ServerRequestState.RESPONSE_COMMITTED

    def resolve_server_request(self, request_id: CodexRequestId) -> None:
        """Retire a request that Codex has already resolved or cleared."""

        self.interactive_server_requests.pop(request_id, None)

    def clear_server_requests(self) -> None:
        """Retire every request owned by this connection generation."""

        self.interactive_server_requests.clear()


@dataclass(frozen=True, slots=True)
class _EventStreamClosed:
    pass


type _QueuedEvent = CodexAppServerEvent | _EventStreamClosed


class CodexAppServerClient:
    """Own one lazy Codex app-server and its bidirectional JSONL connection."""

    def __init__(
        self,
        process_plan_factory: CodexProcessPlanFactory,
        *,
        availability_factory: CodexAvailabilityFactory | None = None,
        client_version: str | None = None,
    ) -> None:
        self._process_plan_factory = process_plan_factory
        self._availability_factory = availability_factory
        self._client_version = client_version or package_version()
        self._plan: CodexAppServerProcessPlan | None = None
        self._connection: _Connection | None = None
        self._plan_lock = asyncio.Lock()
        self._connection_lock = asyncio.Lock()
        self._writer_lock = asyncio.Lock()
        self._events: asyncio.Queue[_QueuedEvent] = asyncio.Queue(
            maxsize=_EVENT_QUEUE_LIMIT
        )
        self._next_request_id = 0
        self._closed = False

    async def availability(self) -> CodexAvailability:
        """Resolve the binary and version without starting app-server."""

        if self._closed:
            return CodexAvailability(
                available=False,
                binary_path=None,
                version=None,
                reason="Codex Direct mode is closed.",
            )
        try:
            if self._availability_factory is not None:
                return await self._availability_factory()
            plan = await self._get_plan()
        except CodexUnavailableError as exc:
            return CodexAvailability(
                available=False,
                binary_path=None,
                version=None,
                reason=str(exc),
            )
        return CodexAvailability(
            available=True,
            binary_path=plan.binary_path,
            version=plan.version,
            reason=None,
        )

    async def initialize(self) -> CodexInitialization:
        """Start and initialize app-server on first use."""

        connection = await self._ensure_connection()
        initialization = connection.initialization
        if initialization is None:
            raise CodexProtocolError("Codex app-server did not initialize.")
        return initialization

    async def controls(self, *, cwd: str) -> CodexControlCatalog:
        """Read native Codex controls, following every paginated catalog."""

        connection = await self._ensure_connection()
        models, config = await asyncio.gather(
            self._optional_paged_objects(connection, "model/list", {}),
            self._optional_config(
                connection,
                cwd=cwd,
            ),
        )
        return CodexControlCatalog(
            models=models,
            config=config,
        )

    async def start_thread(self, settings: CodexThreadSettings) -> CodexThreadHandle:
        """Create a provisional native thread and expose its identity."""

        connection = await self._ensure_connection()
        params = _thread_params(settings)
        params["historyMode"] = "paginated"
        response = await self._request_object(
            connection,
            "thread/start",
            params,
        )
        return _thread_handle(connection.id, response)

    async def materialize_thread(self, thread_id: str) -> None:
        """Make one provisional app-server thread durable and discoverable."""

        connection = await self._ensure_connection()
        await self._request_object(
            connection,
            "thread/section/move",
            {
                "threadId": thread_id,
                "sectionId": None,
                "beforeThreadId": None,
            },
        )

    async def resume_thread(
        self, thread_id: str, settings: CodexThreadSettings
    ) -> CodexThreadHandle:
        """Load a durable native Codex thread into this connection."""

        connection = await self._ensure_connection()
        params = _thread_params(settings)
        params["threadId"] = thread_id
        params["excludeTurns"] = True
        response = await self._request_object(
            connection,
            "thread/resume",
            params,
        )
        return _thread_handle(connection.id, response)

    async def read_thread(self, thread_id: str) -> CodexThreadSnapshot:
        """Read native thread metadata without deprecated history hydration."""

        connection = await self._ensure_connection()
        response = await self._request_object(
            connection,
            "thread/read",
            {"threadId": thread_id, "includeTurns": False},
        )
        thread = _required_object(response, "thread", "thread/read")
        returned_id = _required_string(thread, "id", "thread/read thread")
        if returned_id != thread_id:
            raise CodexProtocolError(
                "Codex thread/read returned a different thread ID."
            )
        return CodexThreadSnapshot(thread_id=thread_id, thread=thread)

    async def list_threads_page(
        self, *, cursor: str | None, limit: int
    ) -> CodexObjectPage:
        """Read one page of app-server native thread metadata."""

        connection = await self._ensure_connection()
        params: JsonObject = {
            "sourceKinds": ["appServer"],
            "sortKey": "recency_at",
            "sortDirection": "desc",
            "limit": _page_limit(limit),
        }
        _add_optional(params, "cursor", cursor)
        response = await self._request_object(connection, "thread/list", params)
        return _object_page(response, "thread/list")

    async def list_turns_page(
        self,
        *,
        thread_id: str,
        cursor: str | None,
        limit: int,
    ) -> CodexObjectPage:
        """Read one newest-first page of persisted native turns and full items."""

        connection = await self._ensure_connection()
        params: JsonObject = {
            "threadId": thread_id,
            "sortDirection": "desc",
            "itemsView": "full",
            "limit": _page_limit(limit),
        }
        _add_optional(params, "cursor", cursor)
        response = await self._request_object(connection, "thread/turns/list", params)
        return _object_page(response, "thread/turns/list")

    async def delete_thread(self, thread_id: str) -> None:
        """Hard-delete one native Codex thread and its descendants."""

        connection = await self._ensure_connection()
        await self._delete_thread(connection, thread_id)

    async def _delete_thread(self, connection: _Connection, thread_id: str) -> None:
        await self._request_object(
            connection,
            "thread/delete",
            {"threadId": thread_id},
        )

    async def start_turn(
        self,
        *,
        thread_id: str,
        text: str,
        settings: CodexTurnSettings,
        client_user_message_id: str | None = None,
    ) -> CodexTurnHandle:
        """Submit one native turn exactly once; failures are never replayed."""

        connection = await self._ensure_connection()
        params: JsonObject = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
        }
        params["model"] = settings.model
        _add_optional(params, "effort", settings.effort)
        _add_optional(params, "clientUserMessageId", client_user_message_id)
        response = await self._request_object(
            connection,
            "turn/start",
            params,
            timeout=None,
        )
        turn = _required_object(response, "turn", "turn/start")
        turn_id = _required_string(turn, "id", "turn/start turn")
        return CodexTurnHandle(
            connection_id=connection.id,
            thread_id=thread_id,
            turn_id=turn_id,
            response=response,
        )

    async def interrupt_turn(self, *, thread_id: str, turn_id: str) -> None:
        """Request interruption; the later turn/completed event is authoritative."""

        connection = await self._ensure_connection()
        await self._request_object(
            connection,
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
        )

    async def respond(
        self,
        *,
        connection_id: str,
        request_id: CodexRequestId,
        response: CodexInteractionResponse,
    ) -> None:
        """Answer one server request only on the generation that emitted it."""

        connection = self._connection
        if (
            connection is None
            or connection.state is not _ConnectionState.READY
            or connection.id != connection_id
        ):
            raise CodexConnectionError(
                "The Codex request belongs to a closed app-server connection.",
                delivery=CodexDelivery.DEFINITELY_NOT_WRITTEN,
            )
        record = connection.require_answerable_server_request(request_id)
        expected_kind = _INTERACTION_KINDS.get(record.method)
        if expected_kind is not response.kind:
            raise CodexProtocolError(
                "Codex interaction response does not match the native request."
            )
        await self._write_server_response(
            connection,
            request_id=request_id,
            result=response.result,
        )

    async def events(self) -> AsyncIterator[CodexAppServerEvent]:
        """Yield the single-consumer native event stream."""

        while True:
            event = await self._events.get()
            if isinstance(event, _EventStreamClosed):
                return
            yield event

    async def close(self) -> None:
        """Close the owned child once, escalating only when it does not exit."""

        connection: _Connection | None = None
        startup_task: asyncio.Task[_StartOutcome] | None = None
        cleanup_task: asyncio.Task[_CleanupOutcome] | None = None
        async with self._connection_lock:
            first_close = not self._closed
            if first_close:
                self._closed = True
            connection = self._connection
            if connection is not None:
                if connection.state is _ConnectionState.STARTING:
                    startup_task = connection.startup_task
                elif connection.state is _ConnectionState.READY:
                    cleanup_task = self._begin_connection_shutdown(
                        connection,
                        error=CodexConnectionError("Codex Direct mode closed."),
                        emit_event=False,
                    )
                else:
                    cleanup_task = connection.cleanup_task
            if first_close:
                self._finish_event_stream()
        if startup_task is not None:
            await asyncio.shield(startup_task)
            if connection is not None:
                cleanup_task = connection.cleanup_task
        if cleanup_task is not None:
            outcome = await asyncio.shield(cleanup_task)
            if outcome.error is not None:
                raise outcome.error

    async def _get_plan(self) -> CodexAppServerProcessPlan:
        plan = self._plan
        if plan is not None:
            return plan
        async with self._plan_lock:
            if self._plan is None:
                self._plan = await self._process_plan_factory()
            return self._plan

    async def _ensure_connection(self) -> _Connection:
        while True:
            startup_task: asyncio.Task[_StartOutcome] | None = None
            cleanup_task: asyncio.Task[_CleanupOutcome] | None = None
            async with self._connection_lock:
                if self._closed:
                    raise CodexUnavailableError("Codex Direct mode is closed.")
                connection = self._connection
                if connection is None:
                    connection = _Connection(id=str(uuid.uuid4()))
                    startup_task = asyncio.create_task(
                        self._run_connection_start(connection),
                        name=f"fcc-codex-app-server-start-{connection.id}",
                    )
                    connection.startup_task = startup_task
                    self._connection = connection
                elif connection.state is _ConnectionState.READY:
                    startup_task = connection.startup_task
                    if startup_task is None or startup_task.done():
                        return connection
                elif connection.state is _ConnectionState.STARTING:
                    startup_task = connection.startup_task
                elif connection.state is _ConnectionState.CLOSING:
                    cleanup_task = connection.cleanup_task
                else:
                    outcome = connection.cleanup_outcome
                    if outcome is not None and outcome.error is not None:
                        raise CodexUnavailableError(
                            str(outcome.error)
                        ) from outcome.error
                    if self._connection is connection:
                        self._connection = None
                    continue

            if startup_task is not None:
                outcome = await asyncio.shield(startup_task)
                if isinstance(outcome, _StartSucceeded):
                    return outcome.connection
                raise outcome.error
            if cleanup_task is None:
                raise RuntimeError("Codex connection lifecycle lost its owner task.")
            outcome = await asyncio.shield(cleanup_task)
            if outcome.error is not None:
                raise CodexUnavailableError(str(outcome.error)) from outcome.error

    async def _run_connection_start(self, connection: _Connection) -> _StartOutcome:
        try:
            plan = await self._get_plan()
            connection.version = plan.version
            if self._closed or self._connection is not connection:
                raise CodexConnectionError(
                    "Codex Direct mode closed during initialization."
                )
            try:
                process = await asyncio.create_subprocess_exec(
                    *plan.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=plan.env,
                    limit=_PROTOCOL_LINE_LIMIT + 1,
                )
            except OSError as exc:
                raise CodexUnavailableError(
                    f"Could not start Codex app-server: {exc}"
                ) from exc
            connection.process = process
            if process.pid is not None:
                register_pid(process.pid)
            if (
                process.stdin is None
                or process.stdout is None
                or process.stderr is None
            ):
                raise CodexUnavailableError(
                    "Codex app-server did not expose its stdio pipes."
                )
            connection.reader_task = asyncio.create_task(
                self._reader_loop(connection),
                name=f"fcc-codex-app-server-reader-{connection.id}",
            )
            connection.stderr_task = asyncio.create_task(
                self._stderr_loop(connection),
                name=f"fcc-codex-app-server-stderr-{connection.id}",
            )
            response = await self._request_object(
                connection,
                "initialize",
                {
                    "clientInfo": {
                        "name": "free_claude_code",
                        "title": "Free Claude Code",
                        "version": self._client_version,
                    },
                    "capabilities": {"experimentalApi": True},
                },
                during_startup=True,
            )
            connection.initialization = CodexInitialization(
                connection_id=connection.id,
                user_agent=_required_string(response, "userAgent", "initialize"),
                codex_home=_required_string(response, "codexHome", "initialize"),
                platform_family=_required_string(
                    response, "platformFamily", "initialize"
                ),
                platform_os=_required_string(response, "platformOs", "initialize"),
            )
            await self._write_initialized(connection)
            async with self._connection_lock:
                if (
                    not self._closed
                    and self._connection is connection
                    and connection.state is _ConnectionState.READY
                ):
                    logger.info(
                        "Codex app-server initialized: connection_id={} version={}",
                        connection.id,
                        connection.version or "unknown",
                    )
                    return _StartSucceeded(connection)
            error = connection.terminal_error or CodexConnectionError(
                "Codex Direct mode closed during initialization."
            )
        except asyncio.CancelledError:
            self._begin_connection_shutdown(
                connection,
                error=CodexConnectionError(
                    "Codex app-server initialization was cancelled."
                ),
                emit_event=False,
            )
            raise
        except Exception as exc:
            error = _connection_error(exc)

        cleanup = await self._shutdown_connection(
            connection,
            error=error,
            emit_event=False,
        )
        return _StartFailed(cleanup.error or error)

    async def _request_object(
        self,
        connection: _Connection,
        method: str,
        params: JsonObject,
        *,
        timeout: float | None = _REQUEST_TIMEOUT_SECONDS,
        during_startup: bool = False,
    ) -> JsonObject:
        result = await self._request(
            connection,
            method,
            params,
            timeout=timeout,
            during_startup=during_startup,
        )
        if not isinstance(result, dict):
            raise CodexProtocolError(f"Codex {method} returned a non-object result.")
        return result

    async def _request(
        self,
        connection: _Connection,
        method: str,
        params: JsonObject,
        *,
        timeout: float | None,
        during_startup: bool = False,
    ) -> JsonValue:
        _require_requestable_connection(
            self,
            connection,
            during_startup=during_startup,
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[JsonValue] = asyncio.get_running_loop().create_future()
        connection.pending[request_id] = _PendingCall(method=method, future=future)
        written = False
        try:
            await self._write_message(
                connection,
                {"id": request_id, "method": method, "params": params},
                during_startup=during_startup,
            )
            written = True
            if timeout is None:
                return await future
            async with asyncio.timeout(timeout):
                return await future
        except TimeoutError as exc:
            raise CodexConnectionError(
                f"Codex {method} did not respond within {timeout:g} seconds.",
                delivery=CodexDelivery.POSSIBLY_WRITTEN,
            ) from exc
        except CodexConnectionError as exc:
            if not written or exc.delivery is CodexDelivery.POSSIBLY_WRITTEN:
                raise
            raise CodexConnectionError(
                str(exc),
                delivery=CodexDelivery.POSSIBLY_WRITTEN,
            ) from exc
        finally:
            pending = connection.pending.get(request_id)
            if pending is not None and pending.future is future:
                connection.pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                with suppress(Exception):
                    future.exception()

    async def _write_initialized(self, connection: _Connection) -> None:
        """Write the handshake barrier before admitting normal requests."""

        process = _connection_process(connection)
        stdin = process.stdin
        if stdin is None:
            raise CodexConnectionError("Codex app-server stdin is unavailable.")
        encoded = b'{"method":"initialized"}\n'
        try:
            async with self._writer_lock:
                _require_requestable_connection(
                    self,
                    connection,
                    during_startup=True,
                )
                if self._closed:
                    raise CodexConnectionError(
                        "Codex Direct mode closed during initialization."
                    )
                stdin.write(encoded)
                await stdin.drain()
                if (
                    self._closed
                    or self._connection is not connection
                    or connection.state is not _ConnectionState.STARTING
                ):
                    raise CodexConnectionError(
                        "Codex Direct mode closed during initialization."
                    )
                _transition_connection(
                    connection,
                    expected=_ConnectionState.STARTING,
                    target=_ConnectionState.READY,
                )
                connection.ready_for_events.set()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            error = CodexConnectionError(
                "Could not finish Codex app-server initialization."
            )
            self._begin_connection_shutdown(
                connection,
                error=error,
                emit_event=False,
            )
            raise error from exc

    async def _write_message(
        self,
        connection: _Connection,
        message: JsonObject,
        *,
        during_startup: bool = False,
    ) -> None:
        _require_requestable_connection(
            self,
            connection,
            during_startup=during_startup,
        )
        process = _connection_process(connection)
        stdin = process.stdin
        if stdin is None:
            raise CodexConnectionError("Codex app-server stdin is unavailable.")
        encoded = (
            json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        try:
            async with self._writer_lock:
                _require_requestable_connection(
                    self,
                    connection,
                    during_startup=during_startup,
                )
                stdin.write(encoded)
                await stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            error = self._write_failure(connection)
            raise error from exc

    async def _write_server_response(
        self,
        connection: _Connection,
        *,
        request_id: CodexRequestId,
        result: JsonValue,
    ) -> None:
        """Write one interactive response and commit its admission atomically."""

        _require_requestable_connection(self, connection, during_startup=False)
        process = _connection_process(connection)
        stdin = process.stdin
        if stdin is None:
            raise CodexConnectionError("Codex app-server stdin is unavailable.")
        encoded = (
            json.dumps(
                {"id": request_id, "result": result},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            async with self._writer_lock:
                _require_requestable_connection(self, connection, during_startup=False)
                connection.require_answerable_server_request(request_id)
                stdin.write(encoded)
                connection.commit_server_response(request_id)
                await stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            error = self._write_failure(connection)
            raise error from exc

    def _write_failure(self, connection: _Connection) -> CodexConnectionError:
        error = CodexConnectionError(
            "Could not write to Codex app-server.",
            delivery=CodexDelivery.POSSIBLY_WRITTEN,
        )
        self._begin_connection_shutdown(
            connection,
            error=error,
            emit_event=True,
        )
        return error

    async def _reader_loop(self, connection: _Connection) -> None:
        process = _connection_process(connection)
        stdout = process.stdout
        if stdout is None:
            return
        try:
            while True:
                try:
                    line = await stdout.readline()
                except ValueError as exc:
                    raise CodexProtocolError(
                        "Codex app-server emitted a protocol line larger than 16 MiB."
                    ) from exc
                if not line:
                    return_code = process.returncode
                    suffix = (
                        f"exit code {return_code}"
                        if return_code is not None
                        else "before process exit"
                    )
                    self._begin_connection_shutdown(
                        connection,
                        error=CodexConnectionError(
                            f"Codex app-server closed its connection ({suffix})."
                        ),
                        emit_event=True,
                    )
                    return
                if connection.state in {
                    _ConnectionState.CLOSING,
                    _ConnectionState.CLOSED,
                }:
                    return
                if len(line) > _PROTOCOL_LINE_LIMIT:
                    raise CodexProtocolError(
                        "Codex app-server emitted a protocol line larger than 16 MiB."
                    )
                message = _decode_message(line)
                if not await self._handle_message(connection, message):
                    return
        except asyncio.CancelledError:
            raise
        except CodexProtocolError as exc:
            self._begin_connection_shutdown(
                connection,
                error=exc,
                emit_event=True,
            )
            return
        except Exception as exc:
            self._begin_connection_shutdown(
                connection,
                error=CodexConnectionError(
                    f"Codex app-server reader failed: {type(exc).__name__}."
                ),
                emit_event=True,
            )
            return

    async def _stderr_loop(self, connection: _Connection) -> None:
        stderr = _connection_process(connection).stderr
        if stderr is None:
            return
        chunks = 0
        total_bytes = 0
        try:
            while chunk := await stderr.read(_STDERR_CHUNK_BYTES):
                chunks += 1
                total_bytes += len(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Codex app-server stderr drain ended: connection_id={} exc_type={}",
                connection.id,
                type(exc).__name__,
            )
        finally:
            if total_bytes:
                logger.debug(
                    "Codex app-server stderr drained: "
                    "connection_id={} chunks={} bytes={}",
                    connection.id,
                    chunks,
                    total_bytes,
                )

    async def _handle_message(
        self, connection: _Connection, message: JsonObject
    ) -> bool:
        if connection.state in {
            _ConnectionState.CLOSING,
            _ConnectionState.CLOSED,
        }:
            return False
        method = message.get("method")
        request_id_value = message.get("id")
        has_id = "id" in message
        if isinstance(method, str):
            if connection.state is _ConnectionState.STARTING:
                if not connection.initialize_response_received:
                    raise CodexProtocolError(
                        "Codex app-server emitted a request or notification "
                        "before its initialize response."
                    )
                await connection.ready_for_events.wait()
                if (
                    self._closed
                    or self._connection is not connection
                    or connection.state is not _ConnectionState.READY
                ):
                    return False
            elif connection.state is not _ConnectionState.READY:
                return False
            params = message.get("params", {})
            if has_id:
                request_id = _request_id(request_id_value)
                return await self._handle_server_request(
                    connection,
                    request_id=request_id,
                    method=method,
                    params=params,
                )
            return await self._handle_notification(
                connection,
                method=method,
                params=params,
            )
        if has_id:
            request_id = _request_id(request_id_value)
            pending = connection.pending.get(request_id)
            if pending is None:
                return True
            if (
                connection.state is _ConnectionState.STARTING
                and pending.method != "initialize"
            ):
                raise RuntimeError(
                    "Codex connection admitted a non-initialize startup request."
                )
            if "result" in message and "error" not in message:
                if (
                    connection.state is _ConnectionState.STARTING
                    and pending.method == "initialize"
                ):
                    connection.initialize_response_received = True
                if not pending.future.done():
                    pending.future.set_result(message["result"])
                return True
            error_value = message.get("error")
            if not isinstance(error_value, dict):
                raise CodexProtocolError(
                    "Codex app-server returned an invalid response envelope."
                )
            code_value = error_value.get("code")
            text_value = error_value.get("message")
            if (
                isinstance(code_value, bool)
                or not isinstance(code_value, int)
                or not isinstance(text_value, str)
            ):
                raise CodexProtocolError(
                    "Codex app-server returned an invalid error envelope."
                )
            text = _bounded_text(text_value)
            if code_value == _METHOD_NOT_FOUND:
                version = connection.version or "unknown"
                failure: Exception = CodexCompatibilityError(
                    f"Codex {version} does not support method "
                    f"{pending.method}; update Codex and try again."
                )
            else:
                failure = CodexRequestError(
                    method=pending.method,
                    code=code_value,
                    message=text,
                )
            if not pending.future.done():
                pending.future.set_exception(failure)
            return True
        raise CodexProtocolError(
            "Codex app-server emitted an unrecognized protocol message."
        )

    async def _handle_notification(
        self,
        connection: _Connection,
        *,
        method: str,
        params: JsonValue,
    ) -> bool:
        if method == _SERVER_REQUEST_RESOLVED:
            if not isinstance(params, dict):
                raise CodexProtocolError(
                    "Codex serverRequest/resolved omitted its params object."
                )
            if not isinstance(params.get("threadId"), str):
                raise CodexProtocolError(
                    "Codex serverRequest/resolved omitted its threadId string."
                )
            connection.resolve_server_request(_request_id(params.get("requestId")))
        return await self._emit(
            connection,
            CodexNotification(
                connection_id=connection.id,
                method=method,
                params=params,
            ),
        )

    async def _handle_server_request(
        self,
        connection: _Connection,
        *,
        request_id: CodexRequestId,
        method: str,
        params: JsonValue,
    ) -> bool:
        if method == "currentTime/read":
            await self._write_message(
                connection,
                {
                    "id": request_id,
                    "result": {"currentTimeAt": int(time.time())},
                },
            )
            return True
        if method in _INTERACTIVE_SERVER_METHODS:
            if not isinstance(params, dict):
                raise CodexProtocolError(f"Codex {method} omitted its params object.")
            thread_id = _required_string(params, "threadId", method)
            turn_id_value = params.get("turnId")
            if turn_id_value is not None and (
                not isinstance(turn_id_value, str) or not turn_id_value
            ):
                raise CodexProtocolError(f"Codex {method} returned an invalid turnId.")
            observation = connection.observe_server_request(
                request_id=request_id,
                method=method,
                params=params,
            )
            if observation is _ServerRequestObservation.CONFLICT:
                raise CodexProtocolError(
                    "Codex app-server reused a server request ID with different "
                    "request content."
                )
            if observation is _ServerRequestObservation.COMMITTED_DUPLICATE:
                return True
            return await self._emit(
                connection,
                CodexInteractionRequest(
                    connection_id=connection.id,
                    request_id=request_id,
                    method=method,
                    thread_id=thread_id,
                    turn_id=turn_id_value,
                    kind=_INTERACTION_KINDS[method],
                    params=dict(params),
                ),
            )
        await self._write_message(
            connection,
            {
                "id": request_id,
                "error": {
                    "code": _METHOD_NOT_FOUND,
                    "message": f"Unsupported server request: {method}",
                },
            },
        )
        return await self._emit(
            connection,
            CodexUnsupportedInteraction(
                connection_id=connection.id,
                method=_bounded_text(method, limit=200),
            ),
        )

    async def _emit(self, connection: _Connection, event: CodexAppServerEvent) -> bool:
        if (
            connection.state is not _ConnectionState.READY
            or self._connection is not connection
        ):
            return False
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            self._begin_connection_shutdown(
                connection,
                error=CodexConnectionError("Codex app-server event queue overflowed."),
                emit_event=True,
            )
            return False
        return True

    async def _paged_objects(
        self,
        connection: _Connection,
        method: str,
        base_params: JsonObject,
    ) -> tuple[JsonObject, ...]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        records: list[JsonObject] = []
        while True:
            params = dict(base_params)
            params["limit"] = 100
            if cursor is not None:
                params["cursor"] = cursor
            response = await self._request_object(connection, method, params)
            records.extend(_object_sequence(response.get("data"), method))
            next_cursor = response.get("nextCursor")
            if next_cursor is None:
                return tuple(records)
            if not isinstance(next_cursor, str) or not next_cursor:
                raise CodexProtocolError(
                    f"Codex {method} returned an invalid next cursor."
                )
            if next_cursor in seen_cursors:
                raise CodexProtocolError(
                    f"Codex {method} repeated a pagination cursor."
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    async def _optional_paged_objects(
        self,
        connection: _Connection,
        method: str,
        base_params: JsonObject,
    ) -> tuple[JsonObject, ...] | None:
        try:
            return await self._paged_objects(connection, method, base_params)
        except CodexCompatibilityError:
            return None

    async def _catalog_objects(
        self,
        connection: _Connection,
        method: str,
        params: JsonObject,
    ) -> tuple[JsonObject, ...]:
        response = await self._request_object(connection, method, params)
        return _object_sequence(response.get("data"), method)

    async def _optional_catalog_objects(
        self,
        connection: _Connection,
        method: str,
        params: JsonObject,
    ) -> tuple[JsonObject, ...] | None:
        try:
            return await self._catalog_objects(connection, method, params)
        except CodexCompatibilityError:
            return None

    async def _optional_config(
        self,
        connection: _Connection,
        *,
        cwd: str,
    ) -> JsonObject | None:
        try:
            response = await self._request_object(
                connection,
                "config/read",
                {"cwd": cwd, "includeLayers": False},
            )
        except CodexCompatibilityError:
            return None
        config = response.get("config")
        if not isinstance(config, dict):
            raise CodexProtocolError("Codex config/read omitted its config object.")
        return config

    async def _shutdown_connection(
        self,
        connection: _Connection,
        *,
        error: Exception,
        emit_event: bool,
    ) -> _CleanupOutcome:
        task = self._begin_connection_shutdown(
            connection,
            error=error,
            emit_event=emit_event,
        )
        return await asyncio.shield(task)

    def _begin_connection_shutdown(
        self,
        connection: _Connection,
        *,
        error: Exception,
        emit_event: bool,
    ) -> asyncio.Task[_CleanupOutcome]:
        task = connection.cleanup_task
        if task is not None:
            return task
        if connection.state not in {
            _ConnectionState.STARTING,
            _ConnectionState.READY,
        }:
            raise RuntimeError(
                "Codex connection entered a terminal state without a cleanup task."
            )
        was_ready = connection.state is _ConnectionState.READY
        _transition_connection(
            connection,
            expected=connection.state,
            target=_ConnectionState.CLOSING,
        )
        connection.terminal_error = error
        for pending in tuple(connection.pending.values()):
            if not pending.future.done():
                pending.future.set_exception(error)
        connection.clear_server_requests()
        if emit_event and was_ready:
            self._emit_connection_lost(connection.id, str(error))
        task = asyncio.create_task(
            self._run_connection_cleanup(connection),
            name=f"fcc-codex-app-server-cleanup-{connection.id}",
        )
        connection.cleanup_task = task
        return task

    async def _run_connection_cleanup(self, connection: _Connection) -> _CleanupOutcome:
        cleanup_error: CodexConnectionError | None = None
        process = connection.process
        try:
            if process is not None:
                cleanup_error = await _stop_process(process)
        except Exception as exc:
            cleanup_error = CodexConnectionError(
                "Codex app-server cleanup failed while stopping its process: "
                f"{type(exc).__name__}."
            )

        tasks = tuple(
            task
            for task in (connection.reader_task, connection.stderr_task)
            if task is not None
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                async with asyncio.timeout(_TERMINATE_SECONDS):
                    await asyncio.gather(*tasks, return_exceptions=True)
            except TimeoutError:
                cleanup_error = cleanup_error or CodexConnectionError(
                    "Codex app-server I/O tasks did not stop within the deadline."
                )

        if process is not None and process.pid is not None:
            if process.returncode is None:
                cleanup_error = cleanup_error or CodexConnectionError(
                    "Codex app-server could not be reaped after forced termination."
                )
            else:
                unregister_pid(process.pid)

        _transition_connection(
            connection,
            expected=_ConnectionState.CLOSING,
            target=_ConnectionState.CLOSED,
        )
        outcome = _CleanupOutcome(error=cleanup_error)
        connection.cleanup_outcome = outcome
        terminal_error = connection.terminal_error
        if cleanup_error is None:
            logger.info(
                "Codex app-server closed: connection_id={} reason={}",
                connection.id,
                type(terminal_error).__name__,
            )
            async with self._connection_lock:
                if self._connection is connection:
                    self._connection = None
        else:
            logger.warning(
                "Codex app-server cleanup incomplete: "
                "connection_id={} reason={} cleanup_error={}",
                connection.id,
                type(terminal_error).__name__,
                cleanup_error,
            )
        return outcome

    def _emit_connection_lost(self, connection_id: str, message: str) -> None:
        event = CodexConnectionLost(
            connection_id=connection_id,
            message=_bounded_text(message),
        )
        if self._events.full():
            while not self._events.empty():
                with suppress(asyncio.QueueEmpty):
                    self._events.get_nowait()
        self._events.put_nowait(event)

    def _finish_event_stream(self) -> None:
        if self._events.full():
            with suppress(asyncio.QueueEmpty):
                self._events.get_nowait()
        self._events.put_nowait(_EventStreamClosed())


def _transition_connection(
    connection: _Connection,
    *,
    expected: _ConnectionState,
    target: _ConnectionState,
) -> None:
    if connection.state is not expected:
        raise RuntimeError(
            "Invalid Codex connection transition: "
            f"{connection.state.value} -> {target.value}."
        )
    connection.state = target


def _connection_process(connection: _Connection) -> asyncio.subprocess.Process:
    process = connection.process
    if process is None:
        raise CodexConnectionError("Codex app-server process is unavailable.")
    return process


def _require_requestable_connection(
    client: CodexAppServerClient,
    connection: _Connection,
    *,
    during_startup: bool,
) -> None:
    expected = _ConnectionState.STARTING if during_startup else _ConnectionState.READY
    if (
        client._connection is not connection
        or connection.state is not expected
        or (client._closed and not during_startup)
    ):
        raise CodexConnectionError("Codex app-server connection is closed.")


def create_codex_app_server_client(
    *,
    settings: Settings,
    proxy_root_url: str,
    base_env: Mapping[str, str] | None = None,
) -> CodexAppServerClient:
    """Build the production client without starting Codex until first use."""

    environment = dict(os.environ if base_env is None else base_env)
    installed: CodexAvailability | None = None
    availability_lock = asyncio.Lock()

    async def inspect() -> CodexAvailability:
        nonlocal installed
        if installed is not None:
            return installed
        async with availability_lock:
            if installed is None:
                installed = await _inspect_codex_installation(environment)
            return installed

    async def prepare() -> CodexAppServerProcessPlan:
        availability = await inspect()
        if not availability.available or availability.binary_path is None:
            raise CodexUnavailableError(
                availability.reason or "Codex CLI is not installed."
            )
        return await _prepare_codex_app_server_process_plan(
            settings=settings,
            proxy_root_url=proxy_root_url,
            base_env=environment,
            binary_path=availability.binary_path,
            version=availability.version,
        )

    return CodexAppServerClient(prepare, availability_factory=inspect)


async def prepare_codex_app_server_process_plan(
    *,
    settings: Settings,
    proxy_root_url: str,
    base_env: Mapping[str, str],
) -> CodexAppServerProcessPlan:
    """Prepare the Direct child with the same ephemeral config as `fcc-codex`."""

    environment = dict(base_env)
    availability = await _inspect_codex_installation(environment)
    if not availability.available or availability.binary_path is None:
        raise CodexUnavailableError(
            availability.reason or "Codex CLI is not installed."
        )
    return await _prepare_codex_app_server_process_plan(
        settings=settings,
        proxy_root_url=proxy_root_url,
        base_env=environment,
        binary_path=availability.binary_path,
        version=availability.version,
    )


async def _inspect_codex_installation(
    environment: Mapping[str, str],
) -> CodexAvailability:
    binary_path = shutil.which(
        codex_binary_name(),
        path=_environment_path(environment),
    )
    if binary_path is None:
        return CodexAvailability(
            available=False,
            binary_path=None,
            version=None,
            reason=(
                "Codex CLI is not installed. Install it with: "
                "npm install -g @openai/codex"
            ),
        )
    version = await asyncio.to_thread(_read_codex_version, binary_path, environment)
    return CodexAvailability(
        available=True,
        binary_path=binary_path,
        version=version,
        reason=None,
    )


async def _prepare_codex_app_server_process_plan(
    *,
    settings: Settings,
    proxy_root_url: str,
    base_env: Mapping[str, str],
    binary_path: str,
    version: str | None,
) -> CodexAppServerProcessPlan:
    catalog = await asyncio.to_thread(
        codex_model_catalog_plan,
        proxy_root_url,
        settings,
    )
    launcher = build_codex_launcher_plan(
        binary_path=binary_path,
        argv=("app-server", "--stdio"),
        settings=settings,
        proxy_root_url=proxy_root_url,
        catalog_config_args=catalog.config_args,
        catalog_models=catalog.models,
        base_env=base_env,
    )
    return CodexAppServerProcessPlan(
        command=launcher.command,
        env=launcher.env,
        binary_path=binary_path,
        version=version,
    )


def _read_codex_version(binary_path: str, env: Mapping[str, str]) -> str | None:
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            env=env,
            errors="replace",
            text=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    version = result.stdout.strip()
    return _bounded_text(version, limit=200) or None


def _environment_path(environment: Mapping[str, str]) -> str | None:
    return next(
        (value for key, value in environment.items() if key.casefold() == "path"),
        None,
    )


async def _stop_process(
    process: asyncio.subprocess.Process,
) -> CodexConnectionError | None:
    stdin = process.stdin
    if stdin is not None:
        stdin.close()
        try:
            async with asyncio.timeout(_TERMINATE_SECONDS):
                await stdin.wait_closed()
        except TimeoutError, BrokenPipeError, ConnectionResetError:
            pass
    if process.returncode is not None:
        await process.wait()
        return None
    try:
        async with asyncio.timeout(_GRACEFUL_CLOSE_SECONDS):
            await process.wait()
            return None
    except TimeoutError:
        if _IS_WINDOWS and process.pid is not None:
            await asyncio.to_thread(
                kill_pid_tree_best_effort,
                process.pid,
                timeout_seconds=_TERMINATE_SECONDS,
            )
        else:
            with suppress(ProcessLookupError):
                process.terminate()
    try:
        async with asyncio.timeout(_TERMINATE_SECONDS):
            await process.wait()
            return None
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        try:
            async with asyncio.timeout(_TERMINATE_SECONDS):
                await process.wait()
        except TimeoutError:
            return CodexConnectionError(
                "Codex app-server did not exit after forced termination."
            )
    return None


def _thread_params(settings: CodexThreadSettings) -> JsonObject:
    params: JsonObject = {"cwd": settings.cwd}
    _add_optional(params, "model", settings.model)
    return params


def _add_optional(target: JsonObject, key: str, value: JsonValue) -> None:
    if value is not None:
        target[key] = value


def _thread_handle(connection_id: str, response: JsonObject) -> CodexThreadHandle:
    thread = _required_object(response, "thread", "thread response")
    return CodexThreadHandle(
        connection_id=connection_id,
        thread_id=_required_string(thread, "id", "thread response"),
        response=response,
    )


def _object_page(response: JsonObject, source: str) -> CodexObjectPage:
    return CodexObjectPage(
        records=_object_sequence(response.get("data"), source),
        next_cursor=_optional_cursor(response.get("nextCursor"), source),
        backwards_cursor=_optional_cursor(response.get("backwardsCursor"), source),
    )


def _optional_cursor(value: JsonValue, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CodexProtocolError(f"Codex {source} returned an invalid cursor.")
    return value


def _page_limit(value: int) -> int:
    if isinstance(value, bool) or value <= 0 or value > 100:
        raise ValueError("Codex page limit must be between 1 and 100.")
    return value


def _required_object(value: JsonObject, key: str, source: str) -> JsonObject:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise CodexProtocolError(f"Codex {source} omitted its {key} object.")
    return nested


def _required_string(value: JsonObject, key: str, source: str) -> str:
    nested = value.get(key)
    if not isinstance(nested, str) or not nested:
        raise CodexProtocolError(f"Codex {source} omitted its {key} value.")
    return nested


def _object_sequence(value: JsonValue, source: str) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        raise CodexProtocolError(f"Codex {source} omitted its data array.")
    records: list[JsonObject] = []
    for record in value:
        if not isinstance(record, dict):
            raise CodexProtocolError(
                f"Codex {source} returned a non-object catalog entry."
            )
        records.append(record)
    return tuple(records)


def _request_id(value: JsonValue) -> CodexRequestId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CodexProtocolError("Codex app-server emitted an invalid request id.")
    return value


def _decode_message(line: bytes) -> JsonObject:
    try:
        decoded = line.decode("utf-8")
        value = cast(JsonValue, json.loads(decoded))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodexProtocolError("Codex app-server emitted malformed JSONL.") from exc
    if not isinstance(value, dict):
        raise CodexProtocolError(
            "Codex app-server emitted a non-object protocol message."
        )
    return value


def _connection_error(exc: BaseException) -> CodexConnectionError:
    if isinstance(exc, CodexConnectionError):
        return exc
    return CodexConnectionError(
        f"Codex app-server initialization failed: {type(exc).__name__}."
    )


def _bounded_text(value: str, *, limit: int = 1000) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"

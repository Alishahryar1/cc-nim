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
from dataclasses import dataclass, field
from typing import cast

from loguru import logger

from free_claude_code.application.work import (
    CodexAppServerEvent,
    CodexAvailability,
    CodexCompatibilityError,
    CodexConnectionError,
    CodexConnectionLost,
    CodexControlCatalog,
    CodexInitialization,
    CodexNotification,
    CodexProtocolError,
    CodexRequestError,
    CodexRequestId,
    CodexServerRequest,
    CodexThreadHandle,
    CodexThreadSettings,
    CodexTurnHandle,
    CodexTurnSettings,
    CodexUnavailableError,
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
_IS_WINDOWS = os.name == "nt"

_INTERACTIVE_SERVER_METHODS = frozenset(
    {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/permissions/requestApproval",
        "item/tool/requestUserInput",
        "mcpServer/elicitation/request",
        "applyPatchApproval",
        "execCommandApproval",
    }
)


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


@dataclass(slots=True)
class _Connection:
    id: str
    process: asyncio.subprocess.Process
    version: str | None
    pending: dict[CodexRequestId, _PendingCall] = field(default_factory=dict)
    initialization: CodexInitialization | None = None
    reader_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    closed: bool = False
    shutdown_task: asyncio.Task[None] | None = None


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
        models, permission_profiles, collaboration_modes, config = await asyncio.gather(
            self._paged_objects(connection, "model/list", {}),
            self._paged_objects(
                connection,
                "permissionProfile/list",
                {"cwd": cwd},
            ),
            self._catalog_objects(connection, "collaborationMode/list", {}),
            self._request_object(
                connection,
                "config/read",
                {"cwd": cwd, "includeLayers": False},
            ),
        )
        config_value = config.get("config")
        if not isinstance(config_value, dict):
            raise CodexProtocolError("Codex config/read omitted its config object.")
        return CodexControlCatalog(
            models=models,
            collaboration_modes=collaboration_modes,
            permission_profiles=permission_profiles,
            config=config_value,
        )

    async def start_thread(self, settings: CodexThreadSettings) -> CodexThreadHandle:
        """Create one durable native Codex thread."""

        connection = await self._ensure_connection()
        response = await self._request_object(
            connection,
            "thread/start",
            _thread_params(settings),
        )
        return _thread_handle(connection.id, response)

    async def resume_thread(
        self, thread_id: str, settings: CodexThreadSettings
    ) -> CodexThreadHandle:
        """Load a durable native Codex thread into this connection."""

        connection = await self._ensure_connection()
        params = _thread_params(settings)
        params["threadId"] = thread_id
        response = await self._request_object(
            connection,
            "thread/resume",
            params,
        )
        return _thread_handle(connection.id, response)

    async def delete_thread(self, thread_id: str) -> None:
        """Hard-delete one native Codex thread and its descendants."""

        connection = await self._ensure_connection()
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
    ) -> CodexTurnHandle:
        """Submit one native turn exactly once; failures are never replayed."""

        connection = await self._ensure_connection()
        params: JsonObject = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
        }
        _add_optional(params, "model", settings.model)
        _add_optional(params, "effort", settings.effort)
        _add_optional(params, "collaborationMode", settings.collaboration_mode)
        _add_optional(params, "approvalPolicy", settings.approval_policy)
        _add_optional(params, "permissions", settings.permission_profile)
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
        result: JsonValue,
    ) -> None:
        """Answer one server request only on the generation that emitted it."""

        connection = self._connection
        if connection is None or connection.closed or connection.id != connection_id:
            raise CodexConnectionError(
                "The Codex request belongs to a closed app-server connection."
            )
        await self._write_message(
            connection,
            {"id": request_id, "result": result},
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

        shutdown_task: asyncio.Task[None] | None = None
        async with self._connection_lock:
            first_close = not self._closed
            if first_close:
                self._closed = True
            connection = self._connection
            if connection is not None:
                shutdown_task = self._begin_connection_shutdown(
                    connection,
                    error=CodexConnectionError("Codex Direct mode closed."),
                    emit_event=False,
                )
            if first_close:
                self._finish_event_stream()
        if shutdown_task is not None:
            await asyncio.shield(shutdown_task)

    async def _get_plan(self) -> CodexAppServerProcessPlan:
        plan = self._plan
        if plan is not None:
            return plan
        async with self._plan_lock:
            if self._plan is None:
                self._plan = await self._process_plan_factory()
            return self._plan

    async def _ensure_connection(self) -> _Connection:
        connection = self._connection
        if connection is not None and not connection.closed:
            return connection
        async with self._connection_lock:
            if self._closed:
                raise CodexUnavailableError("Codex Direct mode is closed.")
            connection = self._connection
            if connection is not None and not connection.closed:
                return connection
            if connection is not None:
                shutdown_task = connection.shutdown_task
                if shutdown_task is not None:
                    await asyncio.shield(shutdown_task)
            plan = await self._get_plan()
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
            if (
                process.stdin is None
                or process.stdout is None
                or process.stderr is None
            ):
                process.kill()
                await process.wait()
                raise CodexUnavailableError(
                    "Codex app-server did not expose its stdio pipes."
                )
            if process.pid is not None:
                register_pid(process.pid)
            connection = _Connection(
                id=str(uuid.uuid4()),
                process=process,
                version=plan.version,
            )
            self._connection = connection
            connection.reader_task = asyncio.create_task(
                self._reader_loop(connection),
                name="fcc-codex-app-server-reader",
            )
            connection.stderr_task = asyncio.create_task(
                self._stderr_loop(connection),
                name="fcc-codex-app-server-stderr",
            )
            try:
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
                await self._write_message(connection, {"method": "initialized"})
            except BaseException as exc:
                await self._shutdown_connection(
                    connection,
                    error=_connection_error(exc),
                    emit_event=False,
                )
                raise
            logger.info(
                "Codex app-server initialized: connection_id={} version={}",
                connection.id,
                connection.version or "unknown",
            )
            return connection

    async def _request_object(
        self,
        connection: _Connection,
        method: str,
        params: JsonObject,
        *,
        timeout: float | None = _REQUEST_TIMEOUT_SECONDS,
    ) -> JsonObject:
        result = await self._request(
            connection,
            method,
            params,
            timeout=timeout,
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
    ) -> JsonValue:
        if connection.closed:
            raise CodexConnectionError("Codex app-server connection is closed.")
        request_id = self._next_request_id
        self._next_request_id += 1
        future = asyncio.get_running_loop().create_future()
        connection.pending[request_id] = _PendingCall(method=method, future=future)
        try:
            await self._write_message(
                connection,
                {"id": request_id, "method": method, "params": params},
            )
            if timeout is None:
                return await asyncio.shield(future)
            async with asyncio.timeout(timeout):
                return await asyncio.shield(future)
        except TimeoutError as exc:
            connection.pending.pop(request_id, None)
            raise CodexConnectionError(
                f"Codex {method} did not respond within {timeout:g} seconds."
            ) from exc
        except asyncio.CancelledError:
            connection.pending.pop(request_id, None)
            raise
        except BaseException:
            connection.pending.pop(request_id, None)
            raise

    async def _write_message(
        self, connection: _Connection, message: JsonObject
    ) -> None:
        if connection.closed:
            raise CodexConnectionError("Codex app-server connection is closed.")
        stdin = connection.process.stdin
        if stdin is None:
            raise CodexConnectionError("Codex app-server stdin is unavailable.")
        encoded = (
            json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        try:
            async with self._writer_lock:
                stdin.write(encoded)
                await stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            error = CodexConnectionError("Could not write to Codex app-server.")
            await self._shutdown_connection(
                connection,
                error=error,
                emit_event=True,
            )
            raise error from exc

    async def _reader_loop(self, connection: _Connection) -> None:
        stdout = connection.process.stdout
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
                    break
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
            await self._shutdown_connection(
                connection,
                error=exc,
                emit_event=True,
            )
            return
        except Exception as exc:
            await self._shutdown_connection(
                connection,
                error=CodexConnectionError(
                    f"Codex app-server reader failed: {type(exc).__name__}."
                ),
                emit_event=True,
            )
            return
        if not connection.closed:
            return_code = await connection.process.wait()
            await self._shutdown_connection(
                connection,
                error=CodexConnectionError(
                    f"Codex app-server closed its connection (exit code {return_code})."
                ),
                emit_event=True,
            )

    async def _stderr_loop(self, connection: _Connection) -> None:
        stderr = connection.process.stderr
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
        method = message.get("method")
        request_id_value = message.get("id")
        has_id = "id" in message
        if isinstance(method, str):
            params = message.get("params", {})
            if has_id:
                request_id = _request_id(request_id_value)
                return await self._handle_server_request(
                    connection,
                    request_id=request_id,
                    method=method,
                    params=params,
                )
            return await self._emit(
                connection,
                CodexNotification(
                    connection_id=connection.id,
                    method=method,
                    params=params,
                ),
            )
        if has_id:
            request_id = _request_id(request_id_value)
            pending = connection.pending.get(request_id)
            if pending is None:
                return True
            if "result" in message and "error" not in message:
                connection.pending.pop(request_id, None)
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
                    f"Codex {version} does not support required method "
                    f"{pending.method}; update Codex and try again."
                )
            else:
                failure = CodexRequestError(
                    method=pending.method,
                    code=code_value,
                    message=text,
                )
            connection.pending.pop(request_id, None)
            if not pending.future.done():
                pending.future.set_exception(failure)
            return True
        raise CodexProtocolError(
            "Codex app-server emitted an unrecognized protocol message."
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
            return await self._emit(
                connection,
                CodexServerRequest(
                    connection_id=connection.id,
                    request_id=request_id,
                    method=method,
                    params=params,
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
        return True

    async def _emit(self, connection: _Connection, event: CodexAppServerEvent) -> bool:
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            await self._shutdown_connection(
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

    async def _catalog_objects(
        self,
        connection: _Connection,
        method: str,
        params: JsonObject,
    ) -> tuple[JsonObject, ...]:
        response = await self._request_object(connection, method, params)
        return _object_sequence(response.get("data"), method)

    async def _shutdown_connection(
        self,
        connection: _Connection,
        *,
        error: Exception,
        emit_event: bool,
    ) -> None:
        task = self._begin_connection_shutdown(
            connection,
            error=error,
            emit_event=emit_event,
        )
        await asyncio.shield(task)

    def _begin_connection_shutdown(
        self,
        connection: _Connection,
        *,
        error: Exception,
        emit_event: bool,
    ) -> asyncio.Task[None]:
        task = connection.shutdown_task
        if task is not None:
            return task
        connection.closed = True
        for pending in tuple(connection.pending.values()):
            if not pending.future.done():
                pending.future.set_exception(error)
        connection.pending.clear()
        if emit_event:
            self._emit_connection_lost(connection.id, str(error))
        task = asyncio.create_task(
            self._run_connection_shutdown(
                connection,
                error=error,
            ),
            name=f"fcc-codex-app-server-shutdown-{connection.id}",
        )
        connection.shutdown_task = task
        return task

    async def _run_connection_shutdown(
        self,
        connection: _Connection,
        *,
        error: Exception,
    ) -> None:
        try:
            try:
                await _stop_process(connection.process)
            finally:
                tasks = tuple(
                    task
                    for task in (connection.reader_task, connection.stderr_task)
                    if task is not None and not task.done()
                )
                if tasks:
                    for task in tasks:
                        task.cancel()
                    try:
                        async with asyncio.timeout(_TERMINATE_SECONDS):
                            await asyncio.gather(*tasks, return_exceptions=True)
                    except TimeoutError:
                        logger.warning(
                            "Codex app-server I/O tasks did not stop within the "
                            "deadline: connection_id={}",
                            connection.id,
                        )
                if (
                    connection.process.pid is not None
                    and connection.process.returncode is not None
                ):
                    unregister_pid(connection.process.pid)
            logger.info(
                "Codex app-server closed: connection_id={} reason={}",
                connection.id,
                type(error).__name__,
            )
        finally:
            if self._connection is connection:
                self._connection = None

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


async def _stop_process(process: asyncio.subprocess.Process) -> None:
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
        return
    try:
        async with asyncio.timeout(_GRACEFUL_CLOSE_SECONDS):
            await process.wait()
            return
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
            return
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        try:
            async with asyncio.timeout(_TERMINATE_SECONDS):
                await process.wait()
        except TimeoutError as exc:
            raise CodexConnectionError(
                "Codex app-server did not exit after forced termination."
            ) from exc


def _thread_params(settings: CodexThreadSettings) -> JsonObject:
    params: JsonObject = {"cwd": settings.cwd}
    _add_optional(params, "model", settings.model)
    _add_optional(params, "approvalPolicy", settings.approval_policy)
    _add_optional(params, "permissions", settings.permission_profile)
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

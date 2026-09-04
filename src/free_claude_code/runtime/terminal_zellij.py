"""Private Zellij runtime backing process-lifetime Terminal Sessions."""

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import psutil

from free_claude_code.application.terminal import (
    TerminalClientPort,
    TerminalClientRole,
    TerminalEngineHostPort,
    TerminalEngineSessionPort,
    TerminalEngineSnapshot,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

ZELLIJ_VERSION = "0.45.1"
_PANE_ID = "terminal_0"
_COMMAND_TIMEOUT_SECONDS = 5.0
_LAYOUT = "layout { pane; }"
_CONFIG = """\
keybinds clear-defaults=true {
}
simplified_ui true
pane_frames false
default_mode "locked"
mouse_mode true
scroll_buffer_size 10000
mirror_session true
session_serialization false
serialize_pane_viewport false
disable_session_metadata true
web_server false
web_sharing "disabled"
show_startup_tips false
show_release_notes false
advanced_mouse_actions false
mouse_scroll_resize false
scroll_mode_sync false
mouse_hover_effects false
mouse_hover_tips false
visual_bell false
focus_follows_mouse false
mouse_click_through true
host_notification_protocol "off"
support_kitty_keyboard_protocol false
support_kitty_graphics_protocol false
stacked_resize false
plugins {
}
load_plugins {
}
"""


class TerminalClientFactory(Protocol):
    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> TerminalClientPort: ...


class ProcessContainment(Protocol):
    async def establish(self, server: psutil.Process) -> None: ...

    async def terminate(self) -> None: ...

    async def close(self) -> None: ...


class _ManagedTerminalClient(TerminalClientPort):
    def __init__(
        self,
        client: TerminalClientPort,
        release: Callable[[_ManagedTerminalClient], None],
    ) -> None:
        self._client = client
        self._release = release
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def read(self) -> bytes:
        return await self._client.read()

    async def write(self, data: str) -> None:
        await self._client.write(data)

    async def resize(self, rows: int, columns: int) -> None:
        await self._client.resize(rows, columns)

    async def wait(self) -> int | None:
        return await self._client.wait()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            await self._client.close()
            self._closed = True
            self._release(self)


class ZellijTerminalEngineSession(TerminalEngineSessionPort):
    def __init__(
        self,
        *,
        host: ZellijTerminalEngineHost,
        session_name: str,
        cwd: Path,
        environment: Mapping[str, str],
        server: psutil.Process,
        root: psutil.Process,
        containment: ProcessContainment,
        rows: int,
        columns: int,
    ) -> None:
        self._host = host
        self._session_name = session_name
        self._cwd = cwd
        self._environment = dict(environment)
        self._server = server
        self._server_created = server.create_time()
        self._root = root
        self._root_created = root.create_time()
        self._containment = containment
        self._rows = rows
        self._columns = columns
        self._clients: set[_ManagedTerminalClient] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._terminated = False
        self._closed = False

    async def open_client(
        self,
        role: TerminalClientRole,
        *,
        rows: int,
        columns: int,
    ) -> TerminalClientPort:
        async with self._lifecycle_lock:
            self._require_open()
            command = "attach" if role is TerminalClientRole.CONTROLLER else "watch"
            client = await self._host.spawn_client(
                (*self._host.command_prefix, command, self._session_name),
                cwd=self._cwd,
                env=self._environment,
                rows=rows,
                columns=columns,
            )
            managed = _ManagedTerminalClient(client, self._clients.discard)
            self._clients.add(managed)
            return managed

    async def snapshot(self) -> TerminalEngineSnapshot:
        async with self._lifecycle_lock:
            self._require_open()
            return await self._host.read_snapshot(
                self._session_name, cwd=self._cwd, env=self._environment
            )

    async def resize(self, rows: int, columns: int) -> None:
        async with self._lifecycle_lock:
            self._require_open()
            self._rows = rows
            self._columns = columns
            clients = tuple(self._clients)
        results = await asyncio.gather(
            *(client.resize(rows, columns) for client in clients),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("Could not resize every Zellij client.") from failures[0]

    async def wait_root(self) -> int | None:
        self._verify_identity(self._root, self._root_created)
        return await asyncio.to_thread(self._root.wait)

    async def terminate_tree(self) -> None:
        async with self._lifecycle_lock:
            if self._terminated:
                return
            clients = tuple(self._clients)
            results = await asyncio.gather(
                *(client.close() for client in clients), return_exceptions=True
            )
            client_failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            with suppress(Exception):
                await self._host.delete_engine_session(self._session_name)
            await self._containment.terminate()
            self._assert_gone(self._root.pid, self._root_created)
            self._assert_gone(self._server.pid, self._server_created)
            if client_failures:
                raise RuntimeError(
                    "Could not close every Zellij terminal client."
                ) from client_failures[0]
            self._terminated = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
        await self.terminate_tree()
        await self._containment.close()
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        self._host.release_session(self._session_name, self)

    def _require_open(self) -> None:
        if self._closed or self._terminated:
            raise RuntimeError("Zellij terminal session is closed.")

    @staticmethod
    def _verify_identity(process: psutil.Process, created: float) -> None:
        if not process.is_running() or process.create_time() != created:
            raise psutil.NoSuchProcess(process.pid)

    @staticmethod
    def _assert_gone(pid: int, created: float) -> None:
        try:
            process = psutil.Process(pid)
            if process.create_time() == created and process.is_running():
                raise RuntimeError("Terminal process remained alive after cleanup.")
        except psutil.NoSuchProcess:
            pass


class ZellijTerminalEngineHost(TerminalEngineHostPort):
    """Own one isolated Zellij socket namespace for the FCC process."""

    def __init__(
        self,
        *,
        binary: Path,
        config: Path,
        sockets: Path,
        data: Path,
        lock_path: Path,
        client_factory: TerminalClientFactory,
        containment_factory: Callable[[], ProcessContainment],
    ) -> None:
        self._binary = binary
        self._config = config
        self._sockets = sockets
        self._data = data
        self._lock = InterprocessFileLock(lock_path)
        self._client_factory = client_factory
        self._containment_factory = containment_factory
        self._sessions: dict[str, ZellijTerminalEngineSession] = {}
        self._started = False
        self._lifecycle_lock = asyncio.Lock()

    @property
    def command_prefix(self) -> tuple[str, ...]:
        return (
            str(self._binary),
            "--data-dir",
            str(self._data),
            "--config",
            str(self._config),
        )

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            version = await _run_process((str(self._binary), "--version"))
            if version.stdout.decode("utf-8", errors="replace").strip() != (
                f"zellij {ZELLIJ_VERSION}"
            ):
                raise RuntimeError("Managed Zellij version does not match FCC.")
            acquired = await asyncio.to_thread(self._lock.acquire)
            if not acquired:
                raise RuntimeError("Terminal Sessions is open in another FCC process.")
            try:
                await asyncio.to_thread(self._prepare_directories)
                await self._delete_all_engine_sessions()
            except BaseException:
                self._lock.release()
                raise
            self._started = True

    async def create_session(
        self,
        *,
        session_name: str,
        command: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> TerminalEngineSessionPort:
        async with self._lifecycle_lock:
            if not self._started:
                raise RuntimeError("Zellij terminal engine is unavailable.")
            if session_name in self._sessions:
                raise RuntimeError("Zellij terminal session already exists.")

        environment = self._environment(env)
        launch_command, expected_executable = _launch_command(command, cwd, environment)
        creator: TerminalClientPort | None = None
        containment = self._containment_factory()
        try:
            creator = await self.spawn_client(
                (
                    *self.command_prefix,
                    "--layout-string",
                    _LAYOUT,
                    "attach",
                    "--create-background",
                    session_name,
                    "--",
                    *launch_command,
                ),
                cwd=cwd,
                env=environment,
                rows=rows,
                columns=columns,
            )
            await asyncio.wait_for(creator.wait(), timeout=_COMMAND_TIMEOUT_SECONDS)
            server = await asyncio.to_thread(
                self._wait_for_server_process, session_name
            )
            await self._wait_for_pane(session_name, cwd, environment)
            await creator.close()
            creator = None
            root = await asyncio.to_thread(
                _find_root_process, server, expected_executable
            )
            await asyncio.to_thread(
                _verify_root_command, root, expected_executable, launch_command[1:]
            )
            await containment.establish(server)
            session = ZellijTerminalEngineSession(
                host=self,
                session_name=session_name,
                cwd=cwd,
                environment=environment,
                server=server,
                root=root,
                containment=containment,
                rows=rows,
                columns=columns,
            )
            async with self._lifecycle_lock:
                if not self._started or session_name in self._sessions:
                    raise RuntimeError(
                        "Zellij terminal engine changed during creation."
                    )
                self._sessions[session_name] = session
            return session
        except BaseException:
            if creator is not None:
                with suppress(Exception):
                    await creator.close()
            with suppress(Exception):
                await self.delete_engine_session(session_name)
            with suppress(Exception):
                await containment.terminate()
            raise

    async def spawn_client(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> TerminalClientPort:
        return await self._client_factory.spawn(
            argv, cwd=cwd, env=env, rows=rows, columns=columns
        )

    async def read_snapshot(
        self, session_name: str, *, cwd: Path, env: Mapping[str, str]
    ) -> TerminalEngineSnapshot:
        process = await asyncio.create_subprocess_exec(
            *self.command_prefix,
            "--session",
            session_name,
            "subscribe",
            "--pane-id",
            _PANE_ID,
            "--scrollback",
            "--format",
            "json",
            "--ansi",
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=_creation_flags(),
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Zellij subscribe stdout was unavailable.")
            line = await asyncio.wait_for(
                process.stdout.readline(), timeout=_COMMAND_TIMEOUT_SECONDS
            )
            if not line:
                stderr = b""
                if process.stderr is not None:
                    stderr = await process.stderr.read()
                raise RuntimeError(
                    "Zellij did not return a terminal snapshot "
                    f"({stderr.decode('utf-8', errors='replace').strip()})."
                )
            return _parse_snapshot(line)
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()

    async def delete_engine_session(self, session_name: str) -> None:
        result = await _run_process(
            (*self.command_prefix, "kill-session", session_name),
            env=self._environment(os.environ),
            check=False,
        )
        if result.returncode not in (0, 1, 2):
            raise RuntimeError("Zellij could not delete its managed session.")

    def release_session(
        self, session_name: str, session: ZellijTerminalEngineSession
    ) -> None:
        if self._sessions.get(session_name) is session:
            del self._sessions[session_name]

    async def close(self) -> None:
        async with self._lifecycle_lock:
            sessions = tuple(self._sessions.values())
        results = await asyncio.gather(
            *(session.close() for session in sessions), return_exceptions=True
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("Could not close every Zellij session.") from failures[0]
        async with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            await asyncio.to_thread(self._lock.release)

    async def _delete_all_engine_sessions(self) -> None:
        result = await _run_process(
            (*self.command_prefix, "kill-all-sessions", "--yes"),
            env=self._environment(os.environ),
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("Could not clean stale FCC terminal sessions.")

    async def _wait_for_pane(
        self, session_name: str, cwd: Path, env: Mapping[str, str]
    ) -> dict[str, object]:
        deadline = asyncio.get_running_loop().time() + _COMMAND_TIMEOUT_SECONDS
        while True:
            result = await _run_process(
                (
                    *self.command_prefix,
                    "--session",
                    session_name,
                    "action",
                    "list-panes",
                    "--all",
                    "--json",
                ),
                cwd=cwd,
                env=env,
                check=False,
            )
            if result.returncode == 0:
                pane = _parse_single_terminal_pane(result.stdout)
                if pane is not None:
                    return pane
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("Zellij did not publish its terminal pane.")
            await asyncio.sleep(0.05)

    def _wait_for_server_process(self, session_name: str) -> psutil.Process:
        deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
        while True:
            matches = self._server_processes(session_name)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1 or time.monotonic() >= deadline:
                raise RuntimeError(
                    "Could not identify the unique Zellij server process."
                )
            time.sleep(0.05)

    def _server_processes(self, session_name: str) -> list[psutil.Process]:
        expected_binary = _resolved(self._binary)
        expected_socket = _resolved(self._sockets) / "contract_version_1" / session_name
        matches: list[psutil.Process] = []
        for process in psutil.process_iter(["exe", "cmdline"]):
            try:
                executable = process.info.get("exe")
                command = process.info.get("cmdline")
                if not isinstance(executable, str) or not isinstance(command, list):
                    continue
                if _resolved(Path(executable)) != expected_binary:
                    continue
                values = [value for value in command if isinstance(value, str)]
                if "--server" not in values:
                    continue
                if not values or _resolved(Path(values[-1])) != expected_socket:
                    continue
                matches.append(process)
            except OSError, psutil.Error:
                continue
        return matches

    def _prepare_directories(self) -> None:
        for directory in (
            self._config.parent,
            self._sockets,
            self._data,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            _owner_only(directory, 0o700)
        temporary = self._config.with_suffix(".tmp")
        temporary.write_text(_CONFIG, encoding="utf-8", newline="\n")
        _owner_only(temporary, 0o600)
        temporary.replace(self._config)
        _owner_only(self._config, 0o600)

    def _environment(self, values: Mapping[str, str]) -> dict[str, str]:
        environment = dict(values)
        environment["ZELLIJ_SOCKET_DIR"] = str(self._sockets)
        environment["ZELLIJ_CONFIG_FILE"] = str(self._config)
        return environment


async def _run_process(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=_creation_flags(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_COMMAND_TIMEOUT_SECONDS
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("Managed Zellij command timed out.") from None
    if process.returncode is None:
        raise RuntimeError("Managed Zellij command did not report an exit status.")
    result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if check and result.returncode != 0:
        raise RuntimeError(
            "Managed Zellij command failed "
            f"({stderr.decode('utf-8', errors='replace').strip()})."
        )
    return result


def _parse_snapshot(payload: bytes) -> TerminalEngineSnapshot:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Zellij returned an invalid terminal snapshot.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Zellij terminal snapshot was not an object.")
    if (
        value.get("event") != "pane_update"
        or value.get("pane_id") != _PANE_ID
        or value.get("is_initial") is not True
    ):
        raise RuntimeError("Zellij terminal snapshot had an unexpected identity.")
    scrollback = _line_list(value.get("scrollback"), allow_none=True)
    viewport = _line_list(value.get("viewport"), allow_none=False)
    return TerminalEngineSnapshot(
        scrollback=_render_lines(scrollback),
        viewport=_render_lines(viewport),
    )


def _line_list(value: object, *, allow_none: bool) -> tuple[str, ...]:
    if value is None and allow_none:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError("Zellij terminal snapshot lines were invalid.")
    return tuple(value)


def _render_lines(lines: tuple[str, ...]) -> bytes:
    if not lines:
        return b""
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def _parse_single_terminal_pane(payload: bytes) -> dict[str, object] | None:
    try:
        value = json.loads(payload)
    except UnicodeDecodeError, json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    terminals = [
        item
        for item in value
        if isinstance(item, dict)
        and item.get("is_plugin") is False
        and item.get("id") == 0
    ]
    if len(terminals) != 1:
        return None
    pane = terminals[0]
    if pane.get("exited") is not False or not isinstance(
        pane.get("terminal_command"), str
    ):
        return None
    return pane


def _find_root_process(server: psutil.Process, executable: Path) -> psutil.Process:
    expected = _resolved(executable)
    deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
    while True:
        matches: list[psutil.Process] = []
        for child in server.children(recursive=False):
            try:
                path = child.exe()
                if path and _resolved(Path(path)) == expected:
                    matches.append(child)
            except OSError, psutil.Error:
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 or time.monotonic() >= deadline:
            raise RuntimeError("Could not identify the unique root shell process.")
        time.sleep(0.05)


def _verify_root_command(
    root: psutil.Process, executable: Path, arguments: Sequence[str]
) -> None:
    actual = root.cmdline()
    if not actual or _resolved(Path(actual[0])) != _resolved(executable):
        raise RuntimeError("Zellij root shell executable did not match.")
    if tuple(actual[1:]) != tuple(arguments):
        raise RuntimeError("Zellij root shell arguments did not match.")


def _launch_command(
    command: Sequence[str], cwd: Path, environment: dict[str, str]
) -> tuple[tuple[str, ...], Path]:
    if not command:
        raise RuntimeError("Terminal shell command was empty.")
    requested = Path(command[0])
    if requested.is_absolute():
        executable = requested
    elif requested.parent != Path("."):
        executable = cwd / requested
    else:
        located = shutil.which(command[0], path=environment.get("PATH"))
        if located is None:
            raise RuntimeError("Terminal shell executable was not found.")
        executable = Path(located)
    executable = _resolved(executable)
    if not executable.is_file():
        raise RuntimeError("Terminal shell executable was not found.")
    if os.name != "nt":
        return tuple(command), executable

    existing_path = environment.get("PATH", "")
    environment["PATH"] = str(executable.parent) + (
        os.pathsep + existing_path if existing_path else ""
    )
    return (executable.name, *command[1:]), executable


def _resolved(path: Path) -> Path:
    return Path(os.path.normcase(os.path.abspath(path)))


def _creation_flags() -> int:
    return 0x08000000 if os.name == "nt" else 0


def _owner_only(path: Path, mode: int) -> None:
    if os.name != "nt":
        path.chmod(mode)

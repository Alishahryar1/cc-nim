"""Harness-owned session identity and launch commands."""

import asyncio
import json
import os
import shutil
import signal
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from free_claude_code.application.browser_sessions import (
    BrowserSessionHarness,
    BrowserSessionUnavailableError,
    HarnessAvailability,
)
from free_claude_code.cli.launchers.claude import claude_binary_name
from free_claude_code.cli.launchers.codex import codex_binary_name
from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.core.version import package_version

CODEX_CREATE_TIMEOUT_SECONDS = 15.0
HELPER_STOP_TIMEOUT_SECONDS = 3.0
HELPER_STDERR_LIMIT_BYTES = 8192


class HarnessDriver:
    """Own one harness's executable discovery, native ID, and argv contract."""

    harness: BrowserSessionHarness
    wrapper_name: str
    client_name: str
    allocates_native_session: bool = False

    def availability(self) -> HarnessAvailability:
        if _resolve_command(self.wrapper_name) is None:
            return HarnessAvailability(
                self.harness,
                False,
                f"Update Free Claude Code to install {self.wrapper_name}.",
            )
        if shutil.which(self.client_name) is None:
            return HarnessAvailability(
                self.harness,
                False,
                f"Install {self.client_name} to use this harness.",
            )
        return HarnessAvailability(self.harness, True)

    async def create_native_id(self, project_path: Path) -> str:
        return str(uuid4())

    async def discard_native_id(self, native_id: str, project_path: Path) -> None:
        """Discard an uncommitted native identity, if creation allocated one."""

    def command(self, native_id: str, *, started_once: bool) -> list[str]:
        raise NotImplementedError

    def _wrapper(self) -> str:
        wrapper = _resolve_command(self.wrapper_name)
        if wrapper is None:
            raise BrowserSessionUnavailableError(
                f"{self.wrapper_name} is not installed. Update Free Claude Code."
            )
        if shutil.which(self.client_name) is None:
            raise BrowserSessionUnavailableError(
                f"{self.client_name} is not installed."
            )
        return wrapper


class ClaudeDriver(HarnessDriver):
    harness = BrowserSessionHarness.CLAUDE
    wrapper_name = "fcc-claude"
    client_name = claude_binary_name()

    def command(self, native_id: str, *, started_once: bool) -> list[str]:
        flag = "--resume" if started_once else "--session-id"
        return [self._wrapper(), flag, native_id]


class PiDriver(HarnessDriver):
    harness = BrowserSessionHarness.PI
    wrapper_name = "fcc-pi"
    client_name = "pi"

    def command(self, native_id: str, *, started_once: bool) -> list[str]:
        return [self._wrapper(), "--session-id", native_id]


class CodexDriver(HarnessDriver):
    harness = BrowserSessionHarness.CODEX
    wrapper_name = "fcc-codex"
    client_name = codex_binary_name()
    allocates_native_session = True

    async def create_native_id(self, project_path: Path) -> str:
        wrapper = self._wrapper()
        try:
            return await asyncio.wait_for(
                _create_codex_thread(wrapper, project_path),
                timeout=CODEX_CREATE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise BrowserSessionUnavailableError(
                "Codex did not create a session before the local timeout."
            ) from exc

    async def discard_native_id(self, native_id: str, project_path: Path) -> None:
        wrapper = self._wrapper()
        operation_path = project_path if project_path.is_dir() else Path.home()
        try:
            await asyncio.wait_for(
                _delete_codex_thread(wrapper, operation_path, native_id),
                timeout=CODEX_CREATE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise BrowserSessionUnavailableError(
                "Codex did not discard the local session before the timeout."
            ) from exc

    def command(self, native_id: str, *, started_once: bool) -> list[str]:
        return [self._wrapper(), "resume", native_id]


class HarnessDriverRegistry:
    """Fixed registry for the three customer-facing native harnesses."""

    def __init__(self, drivers: Sequence[HarnessDriver] | None = None) -> None:
        selected = drivers or (ClaudeDriver(), CodexDriver(), PiDriver())
        self._drivers = {driver.harness: driver for driver in selected}

    def driver(self, harness: BrowserSessionHarness) -> HarnessDriver:
        driver = self._drivers.get(harness)
        if driver is None:
            raise BrowserSessionUnavailableError(
                f"Harness {harness.value!r} is unavailable."
            )
        return driver

    def availability(self) -> tuple[HarnessAvailability, ...]:
        return tuple(
            self.driver(harness).availability() for harness in BrowserSessionHarness
        )


def terminal_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the inherited process environment with browser PTY capabilities."""

    env = dict(os.environ if base is None else base)
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    return env


def _resolve_command(name: str) -> str | None:
    if resolved := shutil.which(name):
        return resolved
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [scripts_dir / name]
    if os.name == "nt":
        candidates.insert(0, scripts_dir / f"{name}.exe")
    return next((str(path) for path in candidates if path.is_file()), None)


async def _create_codex_thread(wrapper: str, project_path: Path) -> str:
    response = await _codex_app_server_request(
        wrapper,
        project_path,
        "thread/start",
        {
            "cwd": str(project_path),
            "serviceName": "free_claude_code",
        },
    )
    result = response.get("result")
    thread = result.get("thread") if isinstance(result, dict) else None
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    if not isinstance(thread_id, str) or not thread_id:
        raise BrowserSessionUnavailableError(
            "Codex returned an invalid session identifier."
        )
    return thread_id


async def _delete_codex_thread(
    wrapper: str, project_path: Path, thread_id: str
) -> None:
    await _codex_app_server_request(
        wrapper,
        project_path,
        "thread/delete",
        {"threadId": thread_id},
    )


async def _codex_app_server_request(
    wrapper: str,
    project_path: Path,
    method: str,
    params: dict[str, object],
) -> dict[str, object]:
    process = await asyncio.create_subprocess_exec(
        wrapper,
        "app-server",
        cwd=str(project_path),
        env=terminal_environment(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    register_pid(process.pid)
    stderr_task = asyncio.create_task(_drain_stderr(process.stderr))
    try:
        if process.stdin is None or process.stdout is None:
            raise BrowserSessionUnavailableError(
                "Codex local session operation did not expose its protocol."
            )
        await _send_json(
            process.stdin,
            {
                "method": "initialize",
                "id": 0,
                "params": {
                    "clientInfo": {
                        "name": "free_claude_code",
                        "title": "Free Claude Code",
                        "version": package_version(),
                    }
                },
            },
        )
        await _read_response(process.stdout, 0)
        await _send_json(process.stdin, {"method": "initialized", "params": {}})
        await _send_json(
            process.stdin,
            {
                "method": method,
                "id": 1,
                "params": params,
            },
        )
        response = await _read_response(process.stdout, 1)
        await _finish_codex_app_server(process)
        return response
    except (BrokenPipeError, ConnectionError, json.JSONDecodeError) as exc:
        raise BrowserSessionUnavailableError(
            "Codex local session operation ended before completion."
        ) from exc
    finally:
        await _stop_helper_process(process)
        stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        unregister_pid(process.pid)


async def _send_json(writer: asyncio.StreamWriter, payload: object) -> None:
    writer.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    await writer.drain()


async def _read_response(
    reader: asyncio.StreamReader, request_id: int
) -> dict[str, object]:
    while line := await reader.readline():
        payload = json.loads(line)
        if not isinstance(payload, dict) or payload.get("id") != request_id:
            continue
        error = payload.get("error")
        if error is not None:
            raise BrowserSessionUnavailableError(
                "Codex declined the local session operation."
            )
        return payload
    raise BrowserSessionUnavailableError(
        "Codex local session operation ended before completion."
    )


async def _drain_stderr(reader: asyncio.StreamReader | None) -> bytes:
    if reader is None:
        return b""
    tail = bytearray()
    while chunk := await reader.read(4096):
        tail.extend(chunk)
        if len(tail) > HELPER_STDERR_LIMIT_BYTES:
            del tail[:-HELPER_STDERR_LIMIT_BYTES]
    return bytes(tail)


async def _finish_codex_app_server(
    process: asyncio.subprocess.Process,
) -> None:
    """Close App Server cleanly so Codex commits local thread changes."""

    if process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()
        with suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.wait_closed()
    try:
        returncode = await asyncio.wait_for(
            process.wait(), timeout=HELPER_STOP_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise BrowserSessionUnavailableError(
            "Codex did not finish saving local session changes."
        ) from exc
    if returncode != 0:
        raise BrowserSessionUnavailableError(
            "Codex could not save local session changes."
        )


async def _stop_helper_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        await asyncio.to_thread(kill_pid_tree_best_effort, process.pid)
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=HELPER_STOP_TIMEOUT_SECONDS)
    except TimeoutError:
        if os.name == "nt":
            await asyncio.to_thread(kill_pid_tree_best_effort, process.pid)
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

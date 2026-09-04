"""POSIX PTY clients for the managed terminal engine."""

import asyncio
import os
import signal
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import cast

import psutil
from ptyprocess import PtyProcess

from free_claude_code.application.terminal import TerminalClientPort
from free_claude_code.cli.process_registry import register_pid, unregister_pid

_READ_SIZE = 16 * 1024


class PosixProcessContainment:
    """Own every process in the Zellij server's distinct POSIX session."""

    def __init__(self) -> None:
        self._session_id: int | None = None

    async def establish(self, server: psutil.Process) -> None:
        session_id = await asyncio.to_thread(_session_id, server.pid)
        if session_id == _session_id(0):
            raise RuntimeError("Zellij did not create an isolated POSIX session.")
        self._session_id = session_id

    async def terminate(self) -> None:
        await asyncio.to_thread(self._terminate_sync)

    async def close(self) -> None:
        await self.terminate()

    def _terminate_sync(self) -> None:
        session_id = self._session_id
        if session_id is None:
            return
        processes = _session_processes(session_id)
        for process in processes:
            with suppress(psutil.NoSuchProcess):
                process.send_signal(signal.SIGTERM)
        _, alive = psutil.wait_procs(processes, timeout=1)
        for process in alive:
            with suppress(psutil.NoSuchProcess):
                process.kill()
        _, alive = psutil.wait_procs(alive, timeout=1)
        if alive:
            raise RuntimeError("POSIX terminal session still has live processes.")
        self._session_id = None


def _session_processes(session_id: int) -> list[psutil.Process]:
    processes: list[psutil.Process] = []
    for process in psutil.process_iter(["pid"]):
        try:
            if _session_id(process.pid) == session_id:
                processes.append(process)
        except ProcessLookupError, PermissionError, psutil.NoSuchProcess:
            continue
    return processes


def _session_id(pid: int) -> int:
    getsid = cast(Callable[[int], int], vars(os)["getsid"])
    return getsid(pid)


class PosixTerminalClient(TerminalClientPort):
    def __init__(self, process: PtyProcess) -> None:
        self._process = process
        self._close_lock = threading.Lock()
        self._closed = False
        register_pid(self.pid)

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    async def read(self) -> bytes:
        return await asyncio.to_thread(self._read_sync)

    async def write(self, data: str) -> None:
        await asyncio.to_thread(self._process.write, data.encode("utf-8"))

    async def resize(self, rows: int, columns: int) -> None:
        await asyncio.to_thread(self._process.setwinsize, rows, columns)

    async def wait(self) -> int | None:
        return await asyncio.to_thread(self._process.wait)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _read_sync(self) -> bytes:
        try:
            return self._process.read(_READ_SIZE)
        except EOFError, OSError:
            return b""

    def _close_sync(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._process.close(force=True)
            self._closed = True
            unregister_pid(self.pid)


class PosixTerminalClientFactory:
    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> PosixTerminalClient:
        child_env = dict(env)
        child_env["TERM"] = "xterm-256color"
        process = await asyncio.to_thread(
            PtyProcess.spawn,
            list(argv),
            cwd=str(cwd),
            env=child_env,
            dimensions=(rows, columns),
        )
        return PosixTerminalClient(process)

"""POSIX PTY adapter for process-lifetime Terminal Sessions."""

import asyncio
import os
import signal
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from ptyprocess import PtyProcess

from free_claude_code.application.terminal import TerminalProcessPort
from free_claude_code.cli.process_registry import register_pid, unregister_pid

_READ_SIZE = 16 * 1024
_TERMINATE_GRACE_SECONDS = 1.0


class PosixTerminalProcess(TerminalProcessPort):
    def __init__(self, process: PtyProcess) -> None:
        self._process = process
        self._close_lock = threading.Lock()
        self._closed = False
        register_pid(self.pid)

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    @property
    def alive(self) -> bool:
        return bool(self._process.isalive())

    async def read(self) -> bytes:
        return await asyncio.to_thread(self._read_sync)

    async def write(self, data: bytes) -> None:
        await asyncio.to_thread(self._process.write, data)

    async def resize(self, rows: int, columns: int) -> None:
        await asyncio.to_thread(self._process.setwinsize, rows, columns)

    async def wait(self) -> int | None:
        return await asyncio.to_thread(self._process.wait)

    async def terminate_tree(self) -> None:
        await asyncio.to_thread(self._terminate_tree_sync)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync, False)

    def _read_sync(self) -> bytes:
        try:
            return self._process.read(_READ_SIZE)
        except EOFError, OSError:
            return b""

    def _terminate_tree_sync(self) -> None:
        if not self.alive:
            return
        try:
            # ptyprocess starts the child as the PTY process-group leader.
            os.kill(-self.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while self.alive and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.alive:
            with suppress(ProcessLookupError):
                os.kill(-self.pid, 9)

    def _close_sync(self, force: bool) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._process.close(force=force)
            self._closed = True
            unregister_pid(self.pid)


class PosixTerminalProcessFactory:
    """Spawn the user's configured shell in a POSIX PTY."""

    async def spawn(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> PosixTerminalProcess:
        argv = _posix_shell_argv(env)
        child_env = dict(env)
        child_env["TERM"] = "xterm-256color"
        process = await asyncio.to_thread(
            PtyProcess.spawn,
            argv,
            cwd=str(cwd),
            env=child_env,
            dimensions=(rows, columns),
        )
        return PosixTerminalProcess(process)


def _posix_shell_argv(env: Mapping[str, str]) -> list[str]:
    configured = env.get("SHELL")
    if configured:
        shell = Path(configured).expanduser()
        if shell.is_file() and os.access(shell, os.X_OK):
            return [str(shell)]
    fallback = Path("/bin/sh")
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return [str(fallback)]
    raise FileNotFoundError("No supported POSIX system shell was found.")

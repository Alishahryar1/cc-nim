"""Windows ConPTY adapter for process-lifetime Terminal Sessions."""

import asyncio
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from winpty import PtyProcess

from free_claude_code.application.terminal import TerminalProcessPort
from free_claude_code.cli.process_registry import register_pid, unregister_pid

_READ_SIZE = 16 * 1024


class WindowsTerminalProcess(TerminalProcessPort):
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
        return bool(self._process.pty.isalive())

    async def read(self) -> bytes:
        return await asyncio.to_thread(self._read_sync)

    async def write(self, data: bytes) -> None:
        value = data.decode("utf-8")
        await asyncio.to_thread(self._process.write, value)

    async def resize(self, rows: int, columns: int) -> None:
        await asyncio.to_thread(self._process.setwinsize, rows, columns)

    async def wait(self) -> int | None:
        return await asyncio.to_thread(self._wait_sync)

    async def terminate_tree(self) -> None:
        await asyncio.to_thread(self._terminate_tree_sync)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync, False)

    def _read_sync(self) -> bytes:
        try:
            return self._process.read(_READ_SIZE).encode("utf-8")
        except EOFError, OSError:
            return b""

    def _wait_sync(self) -> int | None:
        while self._process.pty.isalive():
            time.sleep(0.05)
        return self._process.pty.get_exitstatus()

    def _terminate_tree_sync(self) -> None:
        if self.alive:
            subprocess.run(
                ["taskkill", "/PID", str(self.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        self._close_sync(True)

    def _close_sync(self, force: bool) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._process.close(force=force)
            self._closed = True
            unregister_pid(self.pid)


class WindowsTerminalProcessFactory:
    """Spawn the preferred installed PowerShell in a ConPTY."""

    async def spawn(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> WindowsTerminalProcess:
        argv = _windows_shell_argv(env)
        process = await asyncio.to_thread(
            PtyProcess.spawn,
            argv,
            cwd=str(cwd),
            env=dict(env),
            dimensions=(rows, columns),
        )
        return WindowsTerminalProcess(process)


def _windows_shell_argv(env: Mapping[str, str]) -> Sequence[str]:
    path = env.get("PATH")
    for command in ("pwsh", "powershell"):
        executable = shutil.which(command, path=path)
        if executable is not None:
            return (executable, "-NoLogo")

    comspec = env.get("COMSPEC")
    if comspec:
        executable = Path(comspec)
        if executable.is_file():
            return (str(executable),)
    raise FileNotFoundError("No supported Windows system shell was found.")

"""Cross-platform PTY adapter for browser-managed harness processes."""

import importlib
import os
import signal
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)

IS_WINDOWS = os.name == "nt"


class TerminalProcess(Protocol):
    """Blocking process operations isolated behind ``asyncio.to_thread``."""

    @property
    def pid(self) -> int: ...

    def read(self, size: int = 4096) -> str: ...

    def write(self, data: str) -> None: ...

    def resize(self, columns: int, rows: int) -> None: ...

    def wait(self) -> int: ...

    def terminate_tree(self) -> None: ...

    def close(self) -> None: ...


class TerminalSpawner(Protocol):
    """Injectable direct-process creation seam used by the lifecycle manager."""

    def spawn(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        columns: int,
        rows: int,
    ) -> TerminalProcess: ...


class NativeTerminalProcess:
    """Normalize pywinpty and ptyprocess behind one small interface."""

    def __init__(self, process: Any) -> None:
        self._process = process
        self._pid = int(process.pid)
        self._closed = False
        register_pid(self._pid)

    @property
    def pid(self) -> int:
        return self._pid

    def read(self, size: int = 4096) -> str:
        return str(self._process.read(size))

    def write(self, data: str) -> None:
        self._process.write(data)

    def resize(self, columns: int, rows: int) -> None:
        self._process.setwinsize(rows, columns)

    def wait(self) -> int:
        result = self._process.wait()
        if isinstance(result, int):
            return result
        exit_status = getattr(self._process, "exitstatus", None)
        return int(exit_status) if isinstance(exit_status, int) else 0

    def terminate_tree(self) -> None:
        if IS_WINDOWS:
            kill_pid_tree_best_effort(self._pid)
            return
        try:
            os.killpg(self._pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._process.close(force=True)
        finally:
            unregister_pid(self._pid)


class TerminalProcessFactory:
    """Spawn direct argv commands under the platform-native PTY."""

    def spawn(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        columns: int,
        rows: int,
    ) -> TerminalProcess:
        argv = list(command)
        if IS_WINDOWS:
            module = importlib.import_module("winpty")
            process = module.PtyProcess.spawn(
                argv,
                cwd=cwd,
                env=dict(env),
                dimensions=(rows, columns),
            )
        else:
            module = importlib.import_module("ptyprocess")
            process = module.PtyProcessUnicode.spawn(
                argv,
                cwd=cwd,
                env=dict(env),
                dimensions=(rows, columns),
            )
        return NativeTerminalProcess(process)

"""Windows ConPTY clients for the managed terminal engine."""

import asyncio
import ctypes
import threading
import time
from collections.abc import Mapping, Sequence
from ctypes import wintypes
from pathlib import Path

import psutil
from winpty import PtyProcess

from free_claude_code.application.terminal import TerminalClientPort
from free_claude_code.cli.process_registry import register_pid, unregister_pid

_READ_SIZE = 16 * 1024
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsProcessContainment:
    """Kill-on-close Job Object containing one Zellij server tree."""

    def __init__(self) -> None:
        self._handle: int | None = None
        self._identities: tuple[tuple[int, float], ...] = ()

    async def establish(self, server: psutil.Process) -> None:
        await asyncio.to_thread(self._establish_sync, server)

    async def terminate(self) -> None:
        await asyncio.to_thread(self._terminate_sync)

    async def close(self) -> None:
        await self.terminate()

    def _establish_sync(self, server: psutil.Process) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())

        identities: dict[int, float] = {}
        try:
            stable_rounds = 0
            previous: set[int] = set()
            while stable_rounds < 2:
                processes = [server, *server.children(recursive=True)]
                current = {process.pid for process in processes}
                for process in processes:
                    if process.pid in identities:
                        continue
                    process_handle = kernel32.OpenProcess(
                        _PROCESS_TERMINATE
                        | _PROCESS_SET_QUOTA
                        | _PROCESS_QUERY_LIMITED_INFORMATION,
                        False,
                        process.pid,
                    )
                    if not process_handle:
                        if not process.is_running():
                            continue
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if not kernel32.AssignProcessToJobObject(
                            handle, process_handle
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        identities[process.pid] = process.create_time()
                    finally:
                        kernel32.CloseHandle(process_handle)
                stable_rounds = stable_rounds + 1 if current == previous else 0
                previous = current
                if stable_rounds < 2:
                    time.sleep(0.05)
        except BaseException:
            kernel32.CloseHandle(handle)
            raise
        self._handle = int(handle)
        self._identities = tuple(identities.items())

    def _terminate_sync(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
        processes: list[psutil.Process] = []
        for pid, created in self._identities:
            process = _matching_process(pid, created)
            if process is not None:
                processes.append(process)
        _, alive = psutil.wait_procs(processes, timeout=2)
        if alive:
            raise RuntimeError("Windows terminal Job Object still has live processes.")


def _matching_process(pid: int, created: float) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        if process.create_time() != created:
            raise RuntimeError("Terminal process identity changed before cleanup.")
        return process
    except psutil.NoSuchProcess:
        return None


class WindowsTerminalClient(TerminalClientPort):
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
        await asyncio.to_thread(self._process.write, data)

    async def resize(self, rows: int, columns: int) -> None:
        await asyncio.to_thread(self._process.setwinsize, rows, columns)

    async def wait(self) -> int | None:
        return await asyncio.to_thread(self._wait_sync)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _read_sync(self) -> bytes:
        try:
            return self._process.read(_READ_SIZE).encode("utf-8")
        except EOFError, OSError:
            return b""

    def _wait_sync(self) -> int | None:
        # PtyProcess.wait() calls isalive(), which marks the wrapper closed
        # before its reader sockets are released. Poll the underlying PTY so
        # close() still performs its normal socket cleanup after process exit.
        while self._process.pty.isalive():
            time.sleep(0.1)
        return self._process.pty.get_exitstatus()

    def _close_sync(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._process.close(force=True)
            self._closed = True
            unregister_pid(self.pid)


class WindowsTerminalClientFactory:
    async def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        rows: int,
        columns: int,
    ) -> WindowsTerminalClient:
        process = await asyncio.to_thread(
            PtyProcess.spawn,
            tuple(argv),
            cwd=str(cwd),
            env=dict(env),
            dimensions=(rows, columns),
        )
        return WindowsTerminalClient(process)

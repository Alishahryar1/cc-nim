"""Platform-neutral lifecycle for the FCC desktop shell."""

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Protocol

from free_claude_code.cli.commands import (
    ServerStatus,
    ServerSupervisor,
    load_server_settings,
    open_admin_when_ready,
    schedule_open_admin_browser,
)
from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.config.paths import config_dir_path
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import get_settings

if os.name == "nt":
    import msvcrt

    def _try_lock(handle: BinaryIO) -> bool:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: BinaryIO) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DesktopTray(Protocol):
    """UI loop owned by the platform tray adapter."""

    def run(self) -> None: ...

    def stop(self) -> None: ...


class DesktopTrayFactory(Protocol):
    """Construct a tray adapter around a desktop controller."""

    def __call__(self, controller: DesktopController) -> DesktopTray: ...


class ServerOwner(Protocol):
    """Server lifecycle used by the desktop controller."""

    @property
    def status(self) -> ServerStatus: ...

    def schedule_run(self) -> bool: ...

    def run(self, *, open_admin_browser: bool | None = None) -> None: ...

    def request_restart(self) -> bool: ...

    def request_stop(self) -> None: ...


class DesktopInstanceLock:
    """Cross-process lock automatically released by the operating system."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or config_dir_path() / "desktop.lock"
        self._handle: BinaryIO | None = None

    def acquire(self) -> bool:
        """Acquire the singleton lock without waiting."""

        if self._handle is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        if not _try_lock(handle):
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        """Release a held lock; repeated calls are harmless."""

        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            _unlock(handle)
        finally:
            handle.close()


class DesktopController:
    """Coordinate one tray loop with one in-process FCC server owner."""

    def __init__(
        self,
        supervisor: ServerOwner,
        tray_factory: DesktopTrayFactory,
        open_admin: Callable[[], None],
    ) -> None:
        self._supervisor = supervisor
        self._open_admin = open_admin
        self._thread_lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._tray = tray_factory(self)

    @property
    def status(self) -> ServerStatus:
        return self._supervisor.status

    def run(self) -> None:
        """Run the tray on this thread and the FCC server on its owned worker."""

        self._start_server()
        try:
            self._tray.run()
        finally:
            self._supervisor.request_stop()
            self._tray.stop()
            with self._thread_lock:
                thread = self._server_thread
            if thread is not None:
                thread.join()

    def open_admin(self) -> None:
        self._open_admin()

    def restart_server(self) -> None:
        """Restart an active server or relaunch one that exited unexpectedly."""

        with self._thread_lock:
            thread = self._server_thread
        if thread is not None and thread.is_alive():
            self._supervisor.request_restart()
            return
        self._start_server()

    def quit(self) -> None:
        """Close the server gracefully and end the platform tray loop."""

        self._supervisor.request_stop()
        self._tray.stop()

    def _start_server(self) -> None:
        with self._thread_lock:
            if self._server_thread is not None and self._server_thread.is_alive():
                return
            if not self._supervisor.schedule_run():
                return
            self._server_thread = threading.Thread(
                target=self._run_server,
                name="fcc-desktop-server",
            )
            self._server_thread.start()

    def _run_server(self) -> None:
        self._supervisor.run(open_admin_browser=False)


def launch_desktop(tray_factory: DesktopTrayFactory) -> None:
    """Start the singleton desktop host or focus the already running FCC UI."""

    settings = load_server_settings()
    instance_lock = DesktopInstanceLock()
    if not instance_lock.acquire():
        open_admin_when_ready(settings)
        return

    try:
        if preflight_proxy(local_proxy_root_url(settings)) is None:
            open_admin_when_ready(settings)
            return

        supervisor = ServerSupervisor(console_logging=False)

        def open_current_admin() -> None:
            schedule_open_admin_browser(get_settings())

        DesktopController(supervisor, tray_factory, open_current_admin).run()
    finally:
        instance_lock.release()

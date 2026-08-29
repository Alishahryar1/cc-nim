"""Platform-neutral lifecycle for the FCC desktop shell."""

import threading
from collections.abc import Callable
from typing import Protocol

from loguru import logger

from free_claude_code.cli.claude_desktop import configure_claude_desktop_config
from free_claude_code.cli.commands import (
    ServerStatus,
    ServerSupervisor,
    load_server_settings,
    open_admin_when_ready,
    schedule_open_admin_browser,
)
from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.cli.tls_proxy import (
    CaddyTlsProxy,
    verified_https_gateway_url,
)
from free_claude_code.config.loader import get_settings
from free_claude_code.config.paths import config_dir_path
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.interprocess_lock import InterprocessFileLock


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

    def desktop_gateway_url(self) -> str | None: ...


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

    def desktop_gateway_url(self) -> str | None:
        """The live desktop-scoped gateway URL the server is serving.

        Delegates to the server owner so the desktop integration (the
        Claude Desktop config merge) reads the endpoint the active
        generation actually published — the TLS-prefixed HTTPS URL when a
        front is up, the plain-HTTP fallback otherwise — instead of
        re-deriving it. ``None`` before the server publishes one.
        """

        return self._supervisor.desktop_gateway_url()

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
        self._supervisor.run()


def launch_desktop(tray_factory: DesktopTrayFactory) -> None:
    """Start the singleton desktop host or focus the already running FCC UI."""

    settings = load_server_settings()
    instance_lock = InterprocessFileLock(config_dir_path() / "desktop.lock")
    if not instance_lock.acquire():
        open_admin_when_ready(settings)
        return

    tls_proxy = CaddyTlsProxy(settings)
    try:
        if preflight_proxy(local_proxy_root_url(settings)) is None:
            _merge_verified_gateway(settings)
            open_admin_when_ready(settings)
            return

        if tls_proxy.start():
            _merge_verified_gateway(settings)
        else:
            logger.warning(
                "Skipping Claude Desktop config merge: no HTTPS front is "
                "available for the desktop gateway.",
            )

        supervisor = ServerSupervisor(console_logging=False)

        def open_current_admin() -> None:
            schedule_open_admin_browser(get_settings())

        DesktopController(supervisor, tray_factory, open_current_admin).run()
    finally:
        tls_proxy.stop()
        instance_lock.release()


def _merge_verified_gateway(settings: Settings) -> None:
    """Merge the routing block only against a currently verified HTTPS front.

    The merge writes the gateway credential, so the gateway URL is re-verified
    at write time and passed explicitly: a front that verified at startup can
    stop verifying before the config is written, and the default URL
    resolution would then silently fall back to a plain-``http://`` gateway
    while still embedding the reusable proxy token. Requiring a
    ``verified_https_gateway_url`` result guarantees the block never carries
    an unverified or cleartext endpoint. Best-effort: a failed merge is
    logged, never fatal to the desktop lifecycle.

    On the existing-server path this is probe-only adoption — a short-lived
    launch never spawns its own TLS proxy (it would orphan the child), so a
    front is only reused if it already proves it belongs to this install.
    """

    gateway_url = verified_https_gateway_url(settings)
    if gateway_url is None:
        logger.warning(
            "Skipping Claude Desktop config merge: no verified HTTPS front "
            "is available for the desktop gateway. This short-lived launch "
            "cannot bring one up without orphaning it, so stop the running "
            "HTTP-only FCC server first, then start the FCC desktop host "
            "(which manages the front) or enable the caddy TLS proxy.",
        )
        return
    try:
        configure_claude_desktop_config(settings=settings, gateway_base_url=gateway_url)
    except Exception as exc:  # pragma: no cover - defensive; merge already guards
        logger.warning("Claude Desktop config merge failed: {}", exc)

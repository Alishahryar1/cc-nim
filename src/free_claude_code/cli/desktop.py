"""Foreground lifecycle for the tray-less FCC desktop host.

One front, one server, one process: acquire the singleton lock, bring up
the managed HTTPS front, run the FCC server on a worker thread, print the
gateway URL once, and block the main thread on SIGINT/SIGTERM. No tray,
no console fallback, no plain-HTTP fallback — if the front cannot come up
the host fails fast.
"""

import signal
import threading

from free_claude_code.cli.commands import (
    ServerSupervisor,
    load_server_settings,
    open_admin_when_ready,
)
from free_claude_code.cli.tls_proxy import (
    CaddyTlsProxy,
    desktop_gateway_base_url,
)
from free_claude_code.config.paths import config_dir_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock

_stop_event = threading.Event()


def _wait_for_signal() -> None:
    """Block the host until SIGINT/SIGTERM sets the stop event."""

    _stop_event.wait()


def _shutdown_requested() -> bool:
    return _stop_event.is_set()


def launch_desktop() -> None:
    """Run the FCC desktop host: one front, one server, foreground."""

    settings = load_server_settings()
    instance_lock = InterprocessFileLock(config_dir_path() / "desktop.lock")
    if not instance_lock.acquire():
        # A second launch focuses the already running instance's Admin UI.
        open_admin_when_ready(settings)
        return

    _stop_event.clear()
    tls_front = CaddyTlsProxy(settings)
    server_thread: threading.Thread | None = None
    supervisor: ServerSupervisor | None = None
    try:
        if settings.tls_proxy_enabled:
            tls_front.start()  # raises FrontStartError -> fail fast below

        supervisor = ServerSupervisor(console_logging=False)
        if not supervisor.schedule_run():
            raise RuntimeError("FCC server could not be scheduled.")
        server_thread = threading.Thread(
            target=supervisor.run, name="fcc-desktop-server"
        )
        server_thread.start()
        gateway_url = desktop_gateway_base_url(settings)
        print(f"Claude Desktop gateway: {gateway_url}", flush=True)

        # signal.signal is only valid on the main thread; the host owns it.
        signal.signal(signal.SIGINT, lambda *_: _stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: _stop_event.set())
        _wait_for_signal()
        supervisor.request_stop()
    finally:
        tls_front.stop()
        if server_thread is not None:
            server_thread.join()
        instance_lock.release()

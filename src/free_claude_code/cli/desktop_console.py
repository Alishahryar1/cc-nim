"""Tray-less desktop mode for Linux sessions without a status-area backend."""

import threading

from free_claude_code.cli.desktop import DesktopController, launch_desktop
from free_claude_code.cli.desktop_tray import PystrayDesktopTray, tray_is_available


class ConsoleDesktopTray:
    """Null tray that keeps the controller lifecycle intact without UI.

    ``run()`` blocks until ``stop()`` (or SIGINT interrupts the wait), so
    ``DesktopController.run()`` starts and drains the server exactly as it
    does behind a native tray.
    """

    def __init__(self, controller: DesktopController) -> None:
        self._stop_event = threading.Event()

    def run(self) -> None:
        self._stop_event.wait()

    def stop(self) -> None:
        self._stop_event.set()


def launch() -> None:
    """Launch the tray when available, otherwise fall back to console mode."""

    available, reason = tray_is_available()
    if available:
        launch_desktop(PystrayDesktopTray)
        return
    print(
        "Native system tray is unavailable on this session; "
        "running in console mode instead.",
        flush=True,
    )
    print(f"Reason: {reason}", flush=True)
    print(
        "Claude Desktop is routed through this server automatically when "
        "installed; launch it from your app launcher as usual.",
        flush=True,
    )
    launch_desktop(ConsoleDesktopTray)

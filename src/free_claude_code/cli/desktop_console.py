"""Tray-less desktop mode for Linux sessions without a status-area backend."""

import threading

from free_claude_code.cli.desktop import DesktopController, launch_desktop


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


def _print_console_notice(reason: str) -> None:
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


def launch() -> None:
    """Launch the tray when available, otherwise fall back to console mode."""
    # The import selects pystray's platform backend; on a headless Linux
    # session that raises before any probe can run, so a failed import is
    # reported as an unavailable tray rather than crashing the fallback.
    try:
        from free_claude_code.cli.desktop_tray import (
            PystrayDesktopTray,
            tray_is_available,
        )
    except Exception as exc:
        _print_console_notice(f"the native tray adapter failed to load: {exc}")
    else:
        available, reason = tray_is_available()
        if available:
            launch_desktop(PystrayDesktopTray)
            return
        _print_console_notice(reason)
    launch_desktop(ConsoleDesktopTray)

"""Tray-less desktop mode for Linux sessions without a status-area backend."""

import threading

from free_claude_code.cli.desktop import DesktopController, launch_desktop

GATEWAY_URL_POLL_SECONDS = 0.5


class ConsoleDesktopTray:
    """Null tray that keeps the controller lifecycle intact without UI.

    ``run()`` blocks until ``stop()`` (or SIGINT interrupts the wait), so
    ``DesktopController.run()`` starts and drains the server exactly as it
    does behind a native tray. While it blocks it consumes the published
    desktop gateway URL and prints it — re-printing whenever the value
    changes, e.g. the plain-HTTP fallback upgrading to the TLS-prefixed URL
    — so the console surface reports the live endpoint the same way the
    native tray's status notification does.
    """

    def __init__(self, controller: DesktopController) -> None:
        self._controller = controller
        self._stop_event = threading.Event()

    def run(self) -> None:
        reported: str | None = None
        while not self._stop_event.wait(timeout=GATEWAY_URL_POLL_SECONDS):
            line = console_gateway_line(self._controller)
            if line is not None and line != reported:
                print(line, flush=True)
                reported = line

    def stop(self) -> None:
        self._stop_event.set()


def console_gateway_line(controller: DesktopController) -> str | None:
    """Console notice for the live gateway URL, or ``None`` before publication.

    Consumes ``DesktopController.desktop_gateway_url()`` so the tray-less
    console surface reports the endpoint the active server generation
    actually published — the TLS-prefixed HTTPS URL when a front verifies,
    the plain-HTTP fallback otherwise — instead of a static claim.
    """

    gateway_url = controller.desktop_gateway_url()
    if gateway_url is None:
        return None
    return f"Claude Desktop gateway: {gateway_url}"


def _print_console_notice(reason: str) -> None:
    print(
        "Native system tray is unavailable on this session; "
        "running in console mode instead.",
        flush=True,
    )
    print(f"Reason: {reason}", flush=True)


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

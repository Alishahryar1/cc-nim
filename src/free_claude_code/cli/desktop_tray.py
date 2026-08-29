"""pystray adapter for the Windows tray, macOS menu bar, and Linux status area."""

from io import BytesIO

from PIL import Image
from pystray import Icon, Menu, MenuItem

from free_claude_code.cli.desktop import DesktopController, launch_desktop
from free_claude_code.cli.desktop_assets import app_icon_bytes


class PystrayDesktopTray:
    """Render desktop lifecycle actions through the native status area."""

    def __init__(self, controller: DesktopController) -> None:
        self._controller = controller
        self._icon = Icon(
            "free-claude-code",
            _create_icon(),
            "Free Claude Code",
            Menu(
                MenuItem("Open Admin", self._open_admin, default=True),
                MenuItem("Check Server Status", self._check_status),
                MenuItem("Restart Server", self._restart_server),
                Menu.SEPARATOR,
                MenuItem("Quit", self._quit),
            ),
        )

    def run(self) -> None:
        self._icon.run()

    def stop(self) -> None:
        self._icon.stop()

    def _open_admin(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.open_admin()

    def _check_status(self, _icon: Icon, _item: MenuItem) -> None:
        self._icon.notify(status_notification(self._controller), "Free Claude Code")

    def _restart_server(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.restart_server()

    def _quit(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.quit()


def status_notification(controller: DesktopController) -> str:
    """Status text for the tray, including the live desktop gateway URL.

    Consumes ``DesktopController.desktop_gateway_url()`` so the desktop
    surface reports the endpoint the active server generation actually
    published — the TLS-prefixed HTTPS URL when a front verifies, the
    plain-HTTP fallback otherwise — instead of only the process state.
    """

    gateway_url = controller.desktop_gateway_url()
    if gateway_url is None:
        return f"Server is {controller.status}."
    return f"Server is {controller.status}. Gateway: {gateway_url}"


def _create_icon() -> Image.Image:
    """Load the same branded artwork used by native desktop launchers."""

    with Image.open(BytesIO(app_icon_bytes(".png"))) as image:
        return image.convert("RGBA")


_TRAY_BACKEND_HINT = (
    "no supported tray backend is available; install "
    "'gir1.2-ayatanaappindicator3-0.1' (Debian/Ubuntu) or "
    "'libappindicator-gtk3' (Fedora), or use an X11 session"
)


def tray_is_available() -> tuple[bool, str]:
    """Probe whether a native tray icon can be constructed on this desktop.

    Constructing the pystray ``Icon`` resolves the platform backend module;
    on Linux that import fails when no AppIndicator/GTK/X11 backend exists.
    The probe discards the constructed icon without ever showing it.

    Returns ``(True, "")`` when a tray can run, otherwise ``(False, reason)``
    with an actionable message for console fallback mode.
    """

    try:
        image = _create_icon()
    except Exception as exc:
        return False, f"tray icon artwork failed to load: {exc}"
    try:
        Icon("free-claude-code", image, "Free Claude Code")
    except Exception as exc:
        if isinstance(exc, ImportError):
            return False, _TRAY_BACKEND_HINT
        return False, str(exc)
    return True, ""


def launch() -> None:
    """Launch the supported native tray adapter."""

    launch_desktop(PystrayDesktopTray)

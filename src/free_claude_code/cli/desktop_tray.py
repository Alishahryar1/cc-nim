"""pystray adapter for the Windows tray and macOS menu bar."""

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from free_claude_code.cli.desktop import DesktopController, launch_desktop


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
        self._icon.notify(
            f"Server is {self._controller.status}.",
            "Free Claude Code",
        )

    def _restart_server(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.restart_server()

    def _quit(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.quit()


def _create_icon() -> Image.Image:
    """Build a crisp scalable tray glyph without a packaged binary asset."""

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill="#111827")
    color = "#60A5FA"
    draw.line((19, 17, 19, 48), fill=color, width=7)
    draw.line((19, 18, 46, 18), fill=color, width=7)
    draw.line((19, 32, 40, 32), fill=color, width=7)
    return image


def launch() -> None:
    """Launch the supported native tray adapter."""

    launch_desktop(PystrayDesktopTray)

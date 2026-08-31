"""Lightweight entrypoint for the optional FCC desktop shell."""

import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.cli import desktop_console
from free_claude_code.cli.desktop_assets import export_app_icon


def launch(argv: Sequence[str] | None = None) -> None:
    """Export installer assets or launch the supported native tray adapter."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--export-icon":
        export_app_icon(Path(args[1]))
        return
    if args:
        print("Usage: fcc-desktop [--export-icon PATH]", file=sys.stderr)
        raise SystemExit(2)
    if sys.platform not in {"darwin", "win32", "linux"}:
        print("FCC Desktop is supported on Windows, macOS, and Linux.", file=sys.stderr)
        raise SystemExit(1)

    if sys.platform == "linux":
        desktop_console.launch()
        return

    # Import lazily so a headless Linux session without a pystray backend
    # can still run ``--export-icon`` and the console-mode fallback; the
    # tray adapter is only needed on this native-tray branch.
    from free_claude_code.cli import desktop_tray

    desktop_tray.launch()

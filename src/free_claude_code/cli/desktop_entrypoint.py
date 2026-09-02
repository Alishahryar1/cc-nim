"""Lightweight entrypoint for the FCC desktop host (all platforms)."""

import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.cli.desktop import launch_desktop
from free_claude_code.cli.desktop_assets import export_app_icon


def launch(argv: Sequence[str] | None = None) -> None:
    """Export installer assets or run the foreground desktop host."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--export-icon":
        export_app_icon(Path(args[1]))
        return
    if args:
        print("Usage: fcc-desktop [--export-icon PATH]", file=sys.stderr)
        raise SystemExit(2)
    launch_desktop()

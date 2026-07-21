"""Lightweight platform gate for the optional FCC desktop dependencies."""

import sys

if sys.platform in {"darwin", "win32"}:
    from free_claude_code.cli.desktop_tray import launch
else:

    def launch() -> None:
        """Reject platforms for which FCC does not install a desktop shell."""

        print("FCC Desktop is supported on Windows and macOS.", file=sys.stderr)
        raise SystemExit(1)

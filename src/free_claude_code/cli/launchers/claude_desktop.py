"""Installed ``fcc-claude-desktop`` console script.

Thin launcher shim: every config-merge, unmerge, and binary-spawn behavior
lives in :mod:`free_claude_code.cli.claude_desktop` so the tray shell, the
desktop startup hook, and this entry point share one implementation. This
module only preserves the console-script surface used by the install and
uninstall scripts.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.cli import claude_desktop as routing

CLAUDE_DESKTOP_BINARY = routing.CLAUDE_DESKTOP_BINARY

configure_claude_desktop_config = routing.configure_claude_desktop_config
unconfigure_claude_desktop_config = routing.unconfigure_claude_desktop_config
_config_path = routing._config_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fcc-claude-desktop",
        description=(
            "Launch Claude Desktop with the Free Claude Code routing layer, "
            "or merge/reverse the FCC routing block in claude_desktop_config.json."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--configure",
        action="store_true",
        help="Merge the FCC 3P-gateway block into Claude Desktop's config.",
    )
    mode.add_argument(
        "--unconfigure",
        action="store_true",
        help="Reverse the merge applied by --configure.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Override the Claude Desktop config path (primarily for tests).",
    )
    parser.add_argument(
        "binary_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the claude-desktop binary.",
    )
    return parser


def launch(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``fcc-claude-desktop`` console script."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = _build_argparser().parse_args(args)

    if parsed.configure:
        changed = configure_claude_desktop_config(parsed.config_path)
        target = parsed.config_path or _config_path()
        print(f"{'Updated' if changed else 'Already merged'}: {target}")
        return

    if parsed.unconfigure:
        changed = unconfigure_claude_desktop_config(parsed.config_path)
        target = parsed.config_path or _config_path()
        print(f"{'Removed FCC block' if changed else 'Nothing to remove'}: {target}")
        return

    try:
        process = routing.launch_binary(tuple(parsed.binary_args))
    except FileNotFoundError:
        print(
            f"Could not find '{CLAUDE_DESKTOP_BINARY}' binary.",
            file=sys.stderr,
        )
        print(
            "Install Claude Desktop from https://claude.ai/download.",
            file=sys.stderr,
        )
        raise SystemExit(127) from None
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    raise SystemExit(return_code)

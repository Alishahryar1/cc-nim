"""Lightweight entry points for installed Free Claude Code commands."""

import sys
from collections.abc import Sequence

from free_claude_code.core.version import package_version


def serve(argv: Sequence[str] | None = None) -> None:
    """Start the FastAPI server (registered as ``fcc-server``)."""
    if _print_version_if_requested(argv):
        return

    # Keep the server composition root off metadata-only command paths.
    from free_claude_code.cli.commands import serve as run_server

    run_server()


def init(argv: Sequence[str] | None = None) -> None:
    """Scaffold config at ~/.fcc/.env (registered as ``fcc-init``)."""
    if _print_version_if_requested(argv):
        return

    # Config initialization shares command infrastructure with the server.
    from free_claude_code.cli.commands import init as initialize_config

    initialize_config()


def _print_version_if_requested(argv: Sequence[str] | None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    if "--version" not in args:
        return False
    print(f"free-claude-code {package_version()}")
    return True

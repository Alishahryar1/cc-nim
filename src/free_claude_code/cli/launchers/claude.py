"""Installed `fcc-claude` launcher."""

import os
import sys
from collections.abc import Sequence

from free_claude_code.cli.claude_env import (
    CLAUDE_BINARY_NAME,
    build_claude_proxy_env,
)
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url

from .common import preflight_proxy, resolve_client_binary, run_client_process

_DISPLAY_NAME = "Claude Code"
_INSTALL_HINT = "Install Claude Code with: npm install -g @anthropic-ai/claude-code"
_DEFAULT_PERMISSION_MODE_ARGS = ("--permission-mode", "default")
_PERMISSION_MODE_FLAG = "--permission-mode"
_BYPASS_PERMISSIONS_FLAG = "--dangerously-skip-permissions"


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Claude Code with Free Claude Code proxy environment variables."""

    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    binary_name = claude_binary_name()
    binary_path = resolve_client_binary(
        binary_name=binary_name,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    args = list(sys.argv[1:] if argv is None else argv)
    run_client_process(
        command=build_claude_launcher_command(binary_path=binary_path, argv=args),
        env=build_claude_proxy_env(
            proxy_root_url=proxy_root_url,
            auth_token=settings.proxy_auth_token,
            base_env=os.environ,
        ),
        binary_name=binary_name,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )


def claude_binary_name() -> str:
    """Return the Claude Code binary name."""

    return CLAUDE_BINARY_NAME


def build_claude_launcher_command(
    *, binary_path: str, argv: Sequence[str]
) -> list[str]:
    """Start Claude outside auto mode unless the user selected a mode."""

    args = list(argv)
    if not _has_explicit_permission_mode(args):
        args[:0] = _DEFAULT_PERMISSION_MODE_ARGS

    return [binary_path, *args]


def _has_explicit_permission_mode(argv: Sequence[str]) -> bool:
    """Return whether CLI arguments explicitly choose a permission mode."""

    for arg in argv:
        if arg == "--":
            break
        if arg in (
            _PERMISSION_MODE_FLAG,
            _BYPASS_PERMISSIONS_FLAG,
        ) or arg.startswith(f"{_PERMISSION_MODE_FLAG}="):
            return True

    return False

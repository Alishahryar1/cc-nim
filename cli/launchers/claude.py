"""Installed `fcc-claude` launcher."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

from api.admin_urls import local_proxy_root_url
from cli.claude_env import (
    CLAUDE_BINARY_NAME,
    CLAUDE_CODE_AUTO_COMPACT_WINDOW,
    claude_auth_token,
)
from config.settings import get_settings

from .common import preflight_proxy, resolve_client_binary, run_client_process

_DISPLAY_NAME = "Claude Code"
_INSTALL_HINT = "Install Claude Code with: npm install -g @anthropic-ai/claude-code"


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
        env=build_claude_launcher_env(
            proxy_root_url=proxy_root_url,
            auth_token=settings.anthropic_auth_token,
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
    """Return the Claude wrapper command without changing user arguments."""

    return [binary_path, *argv]


def build_claude_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a Claude Code environment that targets the local proxy."""

    env = {
        key: value
        for key, value in base_env.items()
        if not key.startswith("ANTHROPIC_")
    }
    env["ANTHROPIC_BASE_URL"] = proxy_root_url
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = CLAUDE_CODE_AUTO_COMPACT_WINDOW
    env["ANTHROPIC_AUTH_TOKEN"] = claude_auth_token(auth_token)
    return env

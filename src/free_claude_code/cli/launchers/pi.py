"""Installed `fcc-pi` launcher."""

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.cli.proxy_auth import proxy_auth_token
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import get_settings

from .common import preflight_proxy, resolve_client_binary, run_client_process

_API_KEY_ENV = "FCC_PI_API_KEY"
_BASE_URL_ENV = "FCC_PI_BASE_URL"
_BINARY_NAME = "pi"
_DISPLAY_NAME = "Pi"
_HELP_TIMEOUT_SECONDS = 5.0
_MODEL_SCOPE = "free-claude-code/**"
_REQUIRED_HELP_MARKERS = ("--extension", "--models")
_PASSTHROUGH_COMMANDS = frozenset(
    {"config", "install", "list", "remove", "uninstall", "update"}
)
_PASSTHROUGH_FLAGS = frozenset({"--help", "-h", "--version", "-v"})


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Pi with a process-local Free Claude Code provider."""

    args = list(sys.argv[1:] if argv is None else argv)
    install_hint = pi_install_hint()
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=install_hint,
    )
    if not pi_binary_is_compatible(binary_path):
        print(
            f"The 'pi' command at {binary_path} is not a compatible Pi Coding Agent.",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(126)

    if is_pi_passthrough(args):
        run_client_process(
            command=[binary_path, *args],
            env=os.environ,
            binary_name=_BINARY_NAME,
            display_name=_DISPLAY_NAME,
            install_hint=install_hint,
        )
        return

    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    extension_path = pi_extension_path()
    if not extension_path.is_file():
        print(
            "Free Claude Code's bundled Pi extension is missing. Reinstall FCC.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    run_client_process(
        command=build_pi_launcher_command(
            binary_path=binary_path,
            extension_path=extension_path,
            argv=args,
        ),
        env=build_pi_launcher_env(
            proxy_root_url=proxy_root_url,
            auth_token=settings.anthropic_auth_token,
            base_env=os.environ,
        ),
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=install_hint,
    )


def build_pi_launcher_command(
    *,
    binary_path: str,
    extension_path: Path,
    argv: Sequence[str],
) -> list[str]:
    """Return a Pi session command with ephemeral FCC provider registration."""

    return [
        binary_path,
        "-e",
        str(extension_path),
        "--models",
        _MODEL_SCOPE,
        *argv,
    ]


def build_pi_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a Pi environment containing only FCC-owned proxy variables."""

    env = with_local_proxy_bypass(
        {
            key: value
            for key, value in base_env.items()
            if not key.startswith("FCC_PI_")
        },
        proxy_root_url=proxy_root_url,
    )
    env[_BASE_URL_ENV] = proxy_root_url.rstrip("/")
    env[_API_KEY_ENV] = proxy_auth_token(auth_token)
    return env


def is_pi_passthrough(argv: Sequence[str]) -> bool:
    """Return whether Pi must receive argv unchanged as a non-session command."""

    return bool(argv) and (
        argv[0] in _PASSTHROUGH_COMMANDS or argv[0] in _PASSTHROUGH_FLAGS
    )


def pi_extension_path() -> Path:
    """Return the absolute installed path to the bundled Pi extension."""

    return Path(__file__).with_name("pi_extension.ts").resolve()


def pi_binary_is_compatible(binary_path: str) -> bool:
    """Return whether an executable exposes the Pi features FCC requires."""

    try:
        result = subprocess.run(
            [binary_path, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_HELP_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and all(
        marker in result.stdout for marker in _REQUIRED_HELP_MARKERS
    )


def pi_install_hint(platform: str | None = None) -> str:
    """Return Pi's official installer command for the current platform."""

    if (platform or sys.platform) == "win32":
        return 'Install Pi with: powershell -c "irm https://pi.dev/install.ps1 | iex"'
    return "Install Pi with: curl -fsSL https://pi.dev/install.sh | sh"

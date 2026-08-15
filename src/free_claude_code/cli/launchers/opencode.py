"""Installed `fcc-opencode` launcher for stable OpenCode v1."""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Never

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.core.json_types import JsonValue

from .common import preflight_proxy, resolve_client_binary, run_client_process
from .model_catalog import client_models_from_response, fetch_proxy_models_response
from .opencode_config import OPENCODE_API_KEY_ENV, OpenCodeConfig, build_opencode_config

_BINARY_NAME = "opencode"
_DISPLAY_NAME = "OpenCode CLI"
_INSTALL_HINT = "Install OpenCode from: https://opencode.ai/docs/"
_MINIMUM_VERSION = (1, 18, 18)
_VERSION_TIMEOUT_SECONDS = 5.0
_VERSION_PATTERN = re.compile(
    r"(?m)^\s*(?:(?:opencode(?:\s+version)?\s+)|v)?"
    r"(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?\s*$"
)
_PASSTHROUGH_COMMANDS = frozenset({"auth", "upgrade", "uninstall", "completion"})
_PASSTHROUGH_FLAGS = frozenset({"--help", "-h", "--version", "-v"})
_PROCESS_CONFIG_KEYS = ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT")


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch OpenCode with one child-scoped FCC provider and model catalog."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

    if is_opencode_passthrough(args):
        run_client_process(
            command=[binary_path, *args],
            env=os.environ,
            binary_name=_BINARY_NAME,
            display_name=_DISPLAY_NAME,
            install_hint=_INSTALL_HINT,
        )
        return

    require_compatible_opencode(binary_path)
    _reject_process_config_conflicts(os.environ)

    settings = get_settings()
    auth_token = settings.proxy_auth_token.strip()
    if not auth_token:
        print(
            "Free Claude Code proxy authentication token is empty.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    try:
        models = client_models_from_response(
            fetch_proxy_models_response(proxy_root_url, auth_token)
        )
        config = build_opencode_config(models, proxy_root_url=proxy_root_url)
    except Exception as exc:
        print(
            f"Could not prepare the OpenCode FCC model catalog: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    _run_with_temporary_config(
        binary_path=binary_path,
        args=args,
        config=config,
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
    )


def is_opencode_passthrough(argv: Sequence[str]) -> bool:
    """Return whether OpenCode should run without requiring a live FCC server."""

    return bool(argv) and (
        argv[0] in _PASSTHROUGH_COMMANDS
        or any(argument in _PASSTHROUGH_FLAGS for argument in argv)
    )


def require_compatible_opencode(binary_path: str) -> None:
    """Exit unless the installed OpenCode satisfies FCC's stable-v1 contract."""

    version = opencode_binary_version(binary_path)
    if version is not None and version >= _MINIMUM_VERSION:
        return

    minimum = ".".join(str(part) for part in _MINIMUM_VERSION)
    print(
        f"The OpenCode CLI at {binary_path} must be a stable release "
        f"at least version {minimum}.",
        file=sys.stderr,
    )
    print("Upgrade it with: opencode upgrade", file=sys.stderr)
    raise SystemExit(126)


def opencode_binary_version(binary_path: str) -> tuple[int, int, int] | None:
    """Read a semantic OpenCode version without invoking a shell."""

    try:
        result = subprocess.run(
            [binary_path, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    match = _VERSION_PATTERN.search(result.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def build_opencode_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    config_path: Path,
    overlay: Mapping[str, JsonValue],
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build an isolated child environment without persisting FCC credentials."""

    env = with_local_proxy_bypass(
        {
            key: value
            for key, value in base_env.items()
            if not key.startswith("FCC_OPENCODE_") and key not in _PROCESS_CONFIG_KEYS
        },
        proxy_root_url=proxy_root_url,
    )
    env["OPENCODE_CONFIG"] = str(config_path)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        overlay,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    env[OPENCODE_API_KEY_ENV] = auth_token
    return env


def _reject_process_config_conflicts(base_env: Mapping[str, str]) -> None:
    for key in _PROCESS_CONFIG_KEYS:
        if base_env.get(key, "").strip():
            print(
                f"{key} is already set. Unset it before running fcc-opencode; "
                "ordinary opencode remains unchanged.",
                file=sys.stderr,
            )
            raise SystemExit(1)


def _run_with_temporary_config(
    *,
    binary_path: str,
    args: Sequence[str],
    config: OpenCodeConfig,
    proxy_root_url: str,
    auth_token: str,
) -> None:
    try:
        temp_config = tempfile.TemporaryDirectory(prefix="fcc-opencode-")
    except OSError as exc:
        _temporary_config_error(exc)

    with temp_config as temp_directory:
        config_path = Path(temp_directory, "opencode.json")
        try:
            config_path.write_text(
                json.dumps(config.file, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _temporary_config_error(exc)

        run_client_process(
            command=[binary_path, *args],
            env=build_opencode_launcher_env(
                proxy_root_url=proxy_root_url,
                auth_token=auth_token,
                config_path=config_path,
                overlay=config.overlay,
                base_env=os.environ,
            ),
            binary_name=_BINARY_NAME,
            display_name=_DISPLAY_NAME,
            install_hint=_INSTALL_HINT,
        )


def _temporary_config_error(exc: OSError) -> Never:
    print(
        f"Could not create temporary OpenCode configuration: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

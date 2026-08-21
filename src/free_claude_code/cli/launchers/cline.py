"""Installed `fcc-cline` launcher for the stable Cline CLI."""

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

from .cline_config import CLINE_PROVIDER_ID, ClineConfig, build_cline_config
from .common import preflight_proxy, resolve_client_binary, run_client_process
from .model_catalog import client_models_from_response, fetch_proxy_models_response

_BINARY_NAME = "cline"
_DISPLAY_NAME = "Cline CLI"
_INSTALL_HINT = (
    "Install Cline from: https://docs.cline.bot/getting-started/installing-cline"
)
_MINIMUM_VERSION = (3, 0, 55)
_VERSION_TIMEOUT_SECONDS = 5.0
_VERSION_PATTERN = re.compile(
    r"(?m)^\s*(?:cline(?:\s+version)?\s+|v)?"
    r"(\d+)\.(\d+)\.(\d+)(?:\+[0-9A-Za-z.-]+)?\s*$"
)
_PASSTHROUGH_COMMANDS = frozenset(
    {
        "auth",
        "config",
        "history",
        "h",
        "plugin",
        "skill",
        "mcp",
        "doctor",
        "update",
        "version",
    }
)
_PASSTHROUGH_FLAGS = frozenset({"--help", "-h", "--version", "-V", "--update"})
_DETACHED_COMMANDS = frozenset(
    {"connect", "schedule", "hub", "dashboard", "hook", "kanban"}
)
_DETACHED_FLAGS = frozenset({"--zen", "-z", "--kanban"})
_VALUE_OPTIONS = frozenset(
    {
        "--auto-approve",
        "-c",
        "--cwd",
        "--thinking",
        "--compaction",
        "--id",
        "-P",
        "--provider",
        "-k",
        "--key",
        "-m",
        "--model",
        "-s",
        "--system",
        "-t",
        "--timeout",
        "--config",
        "--data-dir",
        "--hooks-dir",
        "--team-name",
    }
)
_OPTIONAL_VALUE_OPTIONS = frozenset({"--retries"})
_BOOLEAN_OPTIONS = frozenset(
    {
        "-p",
        "--plan",
        "--json",
        "-i",
        "--tui",
        "-z",
        "--zen",
        "--acp",
        "--worktree",
        "--update",
        "--kanban",
        "-v",
        "--verbose",
        "-a",
        "--act",
        "-y",
        "--yolo",
        "-h",
        "--help",
        "-V",
        "--version",
    }
)


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch one attached Cline process with an FCC-only default provider."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

    if is_cline_passthrough(args):
        _run(binary_path, args, os.environ)
        return
    if detached_cline_surface(args):
        print(
            "fcc-cline supports attached terminal sessions only. "
            "Run this detached Cline surface with ordinary cline instead.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _reject_sandbox_override(args, os.environ)
    _reject_routing_overrides(args)
    require_compatible_cline(binary_path)

    settings = get_settings()
    auth_token = settings.proxy_auth_token.strip()
    if not auth_token:
        print("Free Claude Code proxy authentication token is empty.", file=sys.stderr)
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
        config = build_cline_config(
            models,
            proxy_root_url=proxy_root_url,
            auth_token=auth_token,
        )
    except Exception as exc:
        print(f"Could not prepare the Cline FCC model catalog: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    _run_with_temporary_config(
        binary_path=binary_path,
        args=args,
        config=config,
        proxy_root_url=proxy_root_url,
    )


def is_cline_passthrough(argv: Sequence[str]) -> bool:
    """Return whether Cline can run without FCC-owned process configuration."""

    before_separator = _before_separator(argv)
    if any(argument in _PASSTHROUGH_FLAGS for argument in before_separator):
        return True
    return first_root_positional(argv) in _PASSTHROUGH_COMMANDS


def detached_cline_surface(argv: Sequence[str]) -> bool:
    """Return whether arguments request a process that outlives this launcher."""

    before_separator = _before_separator(argv)
    if any(argument in _DETACHED_FLAGS for argument in before_separator):
        return True
    return first_root_positional(argv) in _DETACHED_COMMANDS


def first_root_positional(argv: Sequence[str]) -> str | None:
    """Find Cline's first root positional without taking over CLI parsing."""

    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return None
        if argument in _VALUE_OPTIONS:
            index += 2
            continue
        if _is_joined_value_option(argument):
            index += 1
            continue
        if argument in _OPTIONAL_VALUE_OPTIONS:
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if argument.startswith("--retries=") or argument in _BOOLEAN_OPTIONS:
            index += 1
            continue
        if argument.startswith("-"):
            return None
        return argument
    return None


def require_compatible_cline(binary_path: str) -> None:
    """Exit unless the installed Cline satisfies FCC's released contract."""

    version = cline_binary_version(binary_path)
    if version is not None and version >= _MINIMUM_VERSION:
        return

    minimum = ".".join(str(part) for part in _MINIMUM_VERSION)
    print(
        f"The Cline CLI at {binary_path} must be a stable release at least version {minimum}.",
        file=sys.stderr,
    )
    print("Upgrade it with: cline update", file=sys.stderr)
    raise SystemExit(126)


def cline_binary_version(binary_path: str) -> tuple[int, int, int] | None:
    """Read a stable semantic Cline version without invoking a shell."""

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


def build_cline_launcher_env(
    *,
    proxy_root_url: str,
    providers_path: Path,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build the child-only Cline environment while preserving native state."""

    env = with_local_proxy_bypass(base_env, proxy_root_url=proxy_root_url)
    env["CLINE_PROVIDER_SETTINGS_PATH"] = str(providers_path)
    env["CLINE_SESSION_BACKEND_MODE"] = "local"
    return env


def _run_with_temporary_config(
    *,
    binary_path: str,
    args: Sequence[str],
    config: ClineConfig,
    proxy_root_url: str,
) -> None:
    try:
        temp_config = tempfile.TemporaryDirectory(prefix="fcc-cline-")
    except OSError as exc:
        _temporary_config_error(exc)

    with temp_config as temp_directory:
        settings_directory = Path(temp_directory, "settings")
        providers_path = settings_directory / "providers.json"
        models_path = settings_directory / "models.json"
        try:
            settings_directory.mkdir(mode=0o700)
            _write_private_json(providers_path, config.providers)
            _write_private_json(models_path, config.models)
        except OSError as exc:
            _temporary_config_error(exc)

        _run(
            binary_path,
            ["--provider", CLINE_PROVIDER_ID, *args],
            build_cline_launcher_env(
                proxy_root_url=proxy_root_url,
                providers_path=providers_path,
                base_env=os.environ,
            ),
        )


def _write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=True, indent=2)
        output.write("\n")


def _run(binary_path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    run_client_process(
        command=[binary_path, *args],
        env=env,
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )


def _reject_routing_overrides(argv: Sequence[str]) -> None:
    for argument in _before_separator(argv):
        if (
            argument in {"-P", "--provider", "-k", "--key"}
            or argument.startswith(("--provider=", "--key="))
            or (argument.startswith(("-P", "-k")) and len(argument) > 2)
        ):
            print(
                "fcc-cline owns the Cline provider and API key. "
                "Use ordinary cline for a different provider.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def _reject_sandbox_override(argv: Sequence[str], base_env: Mapping[str, str]) -> None:
    has_data_dir = any(
        argument == "--data-dir" or argument.startswith("--data-dir=")
        for argument in _before_separator(argv)
    )
    sandbox_enabled = base_env.get("CLINE_SANDBOX", "").strip() == "1"
    if not has_data_dir and not sandbox_enabled:
        return

    print(
        "fcc-cline cannot use Cline sandbox data overrides because Cline replaces "
        "the ephemeral FCC provider file in that mode. Use ordinary cline for "
        "--data-dir or CLINE_SANDBOX=1.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _before_separator(argv: Sequence[str]) -> Sequence[str]:
    try:
        return argv[: argv.index("--")]
    except ValueError:
        return argv


def _is_joined_value_option(argument: str) -> bool:
    if any(
        argument.startswith(f"{option}=")
        for option in _VALUE_OPTIONS
        if option.startswith("--")
    ):
        return True
    return any(
        argument.startswith(option) and len(argument) > len(option)
        for option in _VALUE_OPTIONS
        if option.startswith("-") and not option.startswith("--")
    )


def _temporary_config_error(exc: OSError) -> Never:
    print(f"Could not create temporary Cline configuration: {exc}", file=sys.stderr)
    raise SystemExit(1) from None

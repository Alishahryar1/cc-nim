"""Installed `fcc-hermes` launcher for attached Hermes Agent sessions."""

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Never

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url

from .common import preflight_proxy, resolve_client_binary, run_client_process
from .hermes_config import (
    HERMES_KEY_ENV_PREFIX,
    HermesManagedConfig,
    build_hermes_managed_config,
)
from .model_catalog import client_models_from_response, fetch_proxy_models_response

_BINARY_NAME = "hermes"
_DISPLAY_NAME = "Hermes Agent"
_INSTALL_URL = "https://hermes-agent.nousresearch.com/docs/installation"
_INSTALL_HINT = f"Install Hermes Agent from: {_INSTALL_URL}"
_MINIMUM_VERSION = (0, 20, 4)
_VERSION_TIMEOUT_SECONDS = 5.0
_ACTIVATION_TIMEOUT_SECONDS = 15.0
_VERSION_PATTERN = re.compile(r"(?i)\b(?:Hermes Agent\s+)?v?(\d+)\.(\d+)\.(\d+)\b")
_PASSTHROUGH_FLAGS = frozenset({"-h", "--help", "-V", "--version"})
_DETACHED_COMMANDS = frozenset(
    {
        "acp",
        "rl",
        "gateway",
        "cron",
        "serve",
        "dashboard",
        "gui",
        "desktop",
        "proxy",
    }
)
_GLOBAL_VALUE_OPTIONS = frozenset(
    {
        "-z",
        "--oneshot",
        "--usage-file",
        "-m",
        "--model",
        "--provider",
        "--reasoning",
        "-t",
        "--toolsets",
        "-r",
        "--resume",
        "--in",
        "-s",
        "--skills",
        "-p",
        "--profile",
    }
)
_GLOBAL_OPTIONAL_VALUE_OPTIONS = frozenset({"-c", "--continue"})
_GLOBAL_BOOLEAN_OPTIONS = frozenset(
    {
        "--no-restore-cwd",
        "-w",
        "--worktree",
        "--accept-hooks",
        "--yolo",
        "--pass-session-id",
        "--ignore-user-config",
        "--ignore-rules",
        "--safe-mode",
        "--tui",
        "--cli",
        "--dev",
    }
)


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch one attached Hermes process with an FCC-owned default route."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

    if is_hermes_passthrough(args):
        _run(binary_path, args, os.environ)
        return
    if detached_hermes_surface(args):
        print(
            "fcc-hermes supports attached terminal sessions only. "
            "Run this detached Hermes surface with ordinary hermes instead.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    profile_args = hermes_profile_args(args)
    routed_args, selected_model = _without_routing_overrides(args)
    _reject_managed_policy(os.environ)
    require_compatible_hermes(binary_path)

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
        managed = build_hermes_managed_config(
            models,
            proxy_root_url=proxy_root_url,
            nonce=secrets.token_hex(8),
            selected_model=selected_model,
        )
    except Exception as exc:
        print(f"Could not prepare the Hermes FCC model catalog: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    _run_with_managed_config(
        binary_path=binary_path,
        args=routed_args,
        profile_args=profile_args,
        managed=managed,
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
    )


def is_hermes_passthrough(argv: Sequence[str]) -> bool:
    """Return whether Hermes should run without FCC process configuration."""

    before_separator = _before_separator(argv)
    if any(argument in _PASSTHROUGH_FLAGS for argument in before_separator):
        return True
    root = first_root_positional(argv)
    return root not in {None, "chat"} and not detached_hermes_surface(argv)


def detached_hermes_surface(argv: Sequence[str]) -> bool:
    """Return whether arguments request a process that can outlive the wrapper."""

    root = first_root_positional(argv)
    if root in _DETACHED_COMMANDS:
        return True
    return root == "mcp" and _second_positional(argv) == "serve"


def first_root_positional(argv: Sequence[str]) -> str | None:
    """Find Hermes's root command without copying its full argparse tree."""

    positionals = _root_positionals(argv, limit=1)
    return positionals[0] if positionals else None


def hermes_profile_args(argv: Sequence[str]) -> tuple[str, ...]:
    """Return the first explicit profile selector Hermes will consume."""

    before_separator = _before_separator(argv)
    index = 0
    while index < len(before_separator):
        argument = before_separator[index]
        if argument in {"-p", "--profile"}:
            if index + 1 < len(before_separator):
                return argument, before_separator[index + 1]
            return ()
        if argument.startswith("--profile="):
            return (argument,)
        index = _next_root_index(before_separator, index)
    return ()


def require_compatible_hermes(binary_path: str) -> None:
    """Exit unless the installed Hermes satisfies FCC's audited contract."""

    version = hermes_binary_version(binary_path)
    if version is not None and version >= _MINIMUM_VERSION:
        return

    minimum = ".".join(str(part) for part in _MINIMUM_VERSION)
    print(
        f"The Hermes Agent at {binary_path} must be at least version {minimum}.",
        file=sys.stderr,
    )
    print(f"Upgrade it from: {_INSTALL_URL}", file=sys.stderr)
    raise SystemExit(126)


def hermes_binary_version(binary_path: str) -> tuple[int, int, int] | None:
    """Read a Hermes semantic version without invoking a shell."""

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


def build_hermes_launcher_env(
    *,
    managed_directory: Path,
    key_env: str,
    auth_token: str,
    proxy_root_url: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build a child-only Hermes environment while preserving native state."""

    filtered = {
        key: value
        for key, value in base_env.items()
        if not key.startswith(HERMES_KEY_ENV_PREFIX)
        and key
        not in {
            "HERMES_MANAGED_DIR",
            "HERMES_INFERENCE_MODEL",
            "HERMES_INFERENCE_PROVIDER",
        }
    }
    env = with_local_proxy_bypass(filtered, proxy_root_url=proxy_root_url)
    env["HERMES_MANAGED_DIR"] = str(managed_directory)
    env[key_env] = auth_token
    return env


def _run_with_managed_config(
    *,
    binary_path: str,
    args: Sequence[str],
    profile_args: Sequence[str],
    managed: HermesManagedConfig,
    proxy_root_url: str,
    auth_token: str,
) -> None:
    try:
        temp_config = tempfile.TemporaryDirectory(prefix="fcc-hermes-")
    except OSError as exc:
        _temporary_config_error(exc)

    with temp_config as temp_directory:
        managed_directory = Path(temp_directory)
        config_path = managed_directory / "config.yaml"
        try:
            if os.name != "nt":
                managed_directory.chmod(0o700)
            _write_private_json(config_path, managed.config)
        except OSError as exc:
            _temporary_config_error(exc)

        child_env = build_hermes_launcher_env(
            managed_directory=managed_directory,
            key_env=managed.key_env,
            auth_token=auth_token,
            proxy_root_url=proxy_root_url,
            base_env=os.environ,
        )
        _require_overlay_activation(
            binary_path=binary_path,
            profile_args=profile_args,
            expected_provider=managed.provider_ref,
            env=child_env,
        )
        _run(
            binary_path,
            [
                "--provider",
                managed.provider_ref,
                "--model",
                managed.selected_model,
                *args,
            ],
            child_env,
        )


def _require_overlay_activation(
    *,
    binary_path: str,
    profile_args: Sequence[str],
    expected_provider: str,
    env: Mapping[str, str],
) -> None:
    command = [
        binary_path,
        *profile_args,
        "config",
        "get",
        "model.provider",
        "--json",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_ACTIVATION_TIMEOUT_SECONDS,
            env=dict(env),
        )
    except OSError, subprocess.TimeoutExpired:
        _overlay_activation_error()
    if result.returncode != 0:
        _overlay_activation_error()

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        _overlay_activation_error()
    try:
        active_provider = json.loads(lines[-1])
    except json.JSONDecodeError:
        _overlay_activation_error()
    if active_provider != expected_provider:
        _overlay_activation_error()


def _without_routing_overrides(argv: Sequence[str]) -> tuple[list[str], str | None]:
    args = list(argv)
    cleaned: list[str] = []
    selected_model: str | None = None
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            cleaned.extend(args[index:])
            break
        if argument == "--provider" or argument.startswith("--provider="):
            _routing_override_error()
        if argument in {"-m", "--model"}:
            if index + 1 >= len(args) or args[index + 1] == "--":
                _model_value_error()
            if selected_model is not None:
                _duplicate_model_error()
            selected_model = args[index + 1]
            index += 2
            continue
        if argument.startswith("--model="):
            value = argument.partition("=")[2]
            if not value:
                _model_value_error()
            if selected_model is not None:
                _duplicate_model_error()
            selected_model = value
            index += 1
            continue
        if argument.startswith("-m") and len(argument) > 2:
            if selected_model is not None:
                _duplicate_model_error()
            selected_model = argument[2:]
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned, selected_model


def _reject_managed_policy(base_env: Mapping[str, str]) -> None:
    if base_env.get("HERMES_MANAGED_DIR", "").strip() or _system_policy_exists():
        print(
            "fcc-hermes will not replace an existing Hermes managed policy. "
            "Use ordinary hermes or remove the administrator-managed scope.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _system_policy_exists() -> bool:
    return os.name != "nt" and Path("/etc/hermes").exists()


def _root_positionals(argv: Sequence[str], *, limit: int) -> list[str]:
    before_separator = _before_separator(argv)
    positionals: list[str] = []
    index = 0
    while index < len(before_separator) and len(positionals) < limit:
        argument = before_separator[index]
        if argument in _GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if _is_joined_global_value(argument):
            index += 1
            continue
        if argument in _GLOBAL_OPTIONAL_VALUE_OPTIONS:
            if index + 1 < len(before_separator) and not before_separator[
                index + 1
            ].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if argument in _GLOBAL_BOOLEAN_OPTIONS or argument in _PASSTHROUGH_FLAGS:
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        positionals.append(argument)
        index += 1
    return positionals


def _second_positional(argv: Sequence[str]) -> str | None:
    positionals = _root_positionals(argv, limit=2)
    return positionals[1] if len(positionals) == 2 else None


def _next_root_index(argv: Sequence[str], index: int) -> int:
    argument = argv[index]
    if argument in _GLOBAL_VALUE_OPTIONS:
        return min(index + 2, len(argv))
    if (
        argument in _GLOBAL_OPTIONAL_VALUE_OPTIONS
        and index + 1 < len(argv)
        and not argv[index + 1].startswith("-")
    ):
        return index + 2
    return index + 1


def _is_joined_global_value(argument: str) -> bool:
    if any(
        argument.startswith(f"{option}=")
        for option in _GLOBAL_VALUE_OPTIONS
        if option.startswith("--")
    ):
        return True
    return any(
        argument.startswith(option) and len(argument) > len(option)
        for option in _GLOBAL_VALUE_OPTIONS
        if option.startswith("-") and not option.startswith("--")
    )


def _before_separator(argv: Sequence[str]) -> Sequence[str]:
    try:
        return argv[: argv.index("--")]
    except ValueError:
        return argv


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


def _routing_override_error() -> Never:
    print(
        "fcc-hermes owns the Hermes provider. "
        "Use ordinary hermes for a different provider.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _model_value_error() -> Never:
    print("fcc-hermes requires a value after --model/-m.", file=sys.stderr)
    raise SystemExit(2)


def _duplicate_model_error() -> Never:
    print("fcc-hermes accepts only one --model/-m override.", file=sys.stderr)
    raise SystemExit(2)


def _temporary_config_error(exc: OSError) -> Never:
    print(f"Could not create temporary Hermes configuration: {exc}", file=sys.stderr)
    raise SystemExit(1) from None


def _overlay_activation_error() -> Never:
    print(
        "Hermes did not honor the temporary Free Claude Code route. "
        "Check the selected Hermes profile and managed-policy settings.",
        file=sys.stderr,
    )
    raise SystemExit(1)

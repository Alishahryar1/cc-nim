"""Installed `fcc-muse` launcher for attached Muse Code sessions."""

import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url

from .common import (
    preflight_proxy,
    proxy_v1_url,
    resolve_client_binary,
    run_client_process,
)
from .model_catalog import (
    ClientModel,
    client_models_from_response,
    fetch_proxy_models_response,
)

_BINARY_NAME = "muse"
_DISPLAY_NAME = "Muse Code"
_INSTALL_HINT = (
    "Install Muse Code on macOS/Linux/WSL with "
    "`curl -fsSL https://dev.meta.ai/install.sh | bash`. "
    "Meta does not currently publish an official Windows installer."
)
_MINIMUM_VERSION = (0, 2, 1)
_VERSION_TIMEOUT_SECONDS = 5.0
_VERSION_PATTERN = re.compile(
    r"(?m)^\s*Muse Code\s+(\d+)\.(\d+)\.(\d+)"
    r"(?:\s+\([^\r\n]+\))?\s*$"
)
_PASSTHROUGH_COMMANDS = frozenset(
    {
        "auth",
        "config",
        "export",
        "init",
        "login",
        "logout",
        "sandbox",
        "session-message",
        "skills",
        "trace",
    }
)
_PASSTHROUGH_FLAGS = frozenset({"-h", "--help", "-V", "--version"})
_ROOT_VALUE_OPTIONS = frozenset(
    {
        "--agents",
        "--provider",
        "--preset",
        "--model",
        "--reasoning-effort",
        "--base-url",
        "--image",
        "--workspace",
        "--worktree-base",
        "--worktree-existing",
        "--approval-mode",
        "--approval-judge",
        "--echo-delay-ms",
        "--sandbox-network",
    }
)
_ROOT_OPTIONAL_VALUE_OPTIONS = frozenset({"-w", "--worktree"})
_WORKTREE_VALUES = frozenset({"off", "create", "existing"})
_ROOT_BOOLEAN_OPTIONS = frozenset(
    {
        "--parallel-tool-calls",
        "--no-parallel-tool-calls",
        "--subagent-worktree-isolation",
        "--no-session-log",
        "--yolo",
        "--trust-workspace",
        "--disable-approval",
        "--disable-sandbox",
        "--disable-write",
        "--disable-shell",
        "--enable-shell-tool",
    }
)
_ROUTE_OVERRIDE_OPTIONS = frozenset({"--provider", "--base-url"})
_ROUTING_ENV_KEYS = frozenset(
    {"META_API_KEY", "MUSE_MODEL", "MUSE_CUSTOM_HEADERS", "MUSE_WWW_ROUTING"}
)


class MuseInvocationMode(StrEnum):
    """How `fcc-muse` treats one Muse invocation."""

    PASSTHROUGH = "passthrough"
    TUI = "tui"
    EXEC = "exec"
    RESUME = "resume"


@dataclass(frozen=True, slots=True)
class MuseInvocation:
    """The minimal Muse command classification FCC needs for flag placement."""

    mode: MuseInvocationMode
    command_index: int | None = None


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch one attached Muse Code process through FCC."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    invocation = classify_muse_invocation(args)
    if invocation.mode is MuseInvocationMode.PASSTHROUGH:
        _run(binary_path, args, os.environ)
        return

    _reject_routing_overrides(args, invocation)
    require_compatible_muse(binary_path)

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
        if not models:
            raise ValueError("model catalog contains no routable models")
    except Exception as exc:
        print(f"Could not prepare the Muse FCC model catalog: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    try:
        selected_model, cleaned_args = _select_and_remove_model(args, models)
    except ValueError as exc:
        print(f"Invalid Muse Code model selection: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    child_env = build_muse_launcher_env(
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        base_env=os.environ,
    )
    command_args = _routed_args(
        cleaned_args,
        invocation.mode,
        proxy_root_url=proxy_root_url,
        selected_model=selected_model,
    )
    _run(binary_path, command_args, child_env)


def classify_muse_invocation(argv: Sequence[str]) -> MuseInvocation:
    """Classify only Muse surfaces whose attached lifecycle FCC can own."""

    before_separator = _before_separator(argv)
    if any(argument in _PASSTHROUGH_FLAGS for argument in before_separator):
        return MuseInvocation(MuseInvocationMode.PASSTHROUGH)
    if before_separator and before_separator[0] in _PASSTHROUGH_COMMANDS:
        return MuseInvocation(MuseInvocationMode.PASSTHROUGH)
    if before_separator and before_separator[0] == "exec":
        return MuseInvocation(MuseInvocationMode.EXEC, command_index=0)

    positional_index = _first_root_positional_index(before_separator)
    if positional_index is not None and before_separator[positional_index] == "resume":
        return MuseInvocation(MuseInvocationMode.RESUME, command_index=positional_index)
    return MuseInvocation(MuseInvocationMode.TUI)


def muse_binary_version(binary_path: str) -> tuple[int, int, int] | None:
    """Read the released Muse semantic version without invoking a shell."""

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


def require_compatible_muse(binary_path: str) -> None:
    """Exit unless Muse satisfies FCC's released base-URL contract."""

    version = muse_binary_version(binary_path)
    if version is not None and version >= _MINIMUM_VERSION:
        return

    minimum = ".".join(str(part) for part in _MINIMUM_VERSION)
    print(
        f"The Muse Code CLI at {binary_path} must be at least version {minimum}.",
        file=sys.stderr,
    )
    print(_INSTALL_HINT, file=sys.stderr)
    raise SystemExit(1)


def build_muse_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build the child-only Muse route while preserving native Muse state."""

    filtered = {
        key: value
        for key, value in base_env.items()
        if not key.startswith("FCC_MUSE_") and key not in _ROUTING_ENV_KEYS
    }
    env = with_local_proxy_bypass(filtered, proxy_root_url=proxy_root_url)
    env["META_API_KEY"] = auth_token
    return env


def _reject_routing_overrides(argv: Sequence[str], invocation: MuseInvocation) -> None:
    for argument in _before_separator(argv):
        if argument in _ROUTE_OVERRIDE_OPTIONS or any(
            argument.startswith(f"{option}=") for option in _ROUTE_OVERRIDE_OPTIONS
        ):
            print(
                "fcc-muse owns the attached Muse provider and base URL. "
                "Use ordinary muse for a native Meta route.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if invocation.mode is MuseInvocationMode.EXEC and argument == "--api-key-stdin":
            print(
                "fcc-muse supplies the attached Muse credential. "
                "Use ordinary muse exec with --api-key-stdin for a native route.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def _select_and_remove_model(
    argv: Sequence[str], models: Sequence[ClientModel]
) -> tuple[str, list[str]]:
    selected: str | None = None
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            cleaned.extend(argv[index:])
            break
        if argument == "--model":
            if index + 1 >= len(argv) or not argv[index + 1]:
                raise ValueError("--model requires a model value")
            if selected is not None:
                raise ValueError("multiple --model values are not supported")
            selected = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--model="):
            value = argument.partition("=")[2]
            if not value:
                raise ValueError("--model requires a model value")
            if selected is not None:
                raise ValueError("multiple --model values are not supported")
            selected = value
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    chosen = selected or models[0].wire_slug
    available = {model.wire_slug for model in models}
    if chosen not in available:
        raise ValueError(f"model {chosen!r} is not in the current FCC catalog")
    return chosen, cleaned


def _routed_args(
    argv: Sequence[str],
    mode: MuseInvocationMode,
    *,
    proxy_root_url: str,
    selected_model: str,
) -> list[str]:
    route = [
        "--provider",
        "meta",
        "--base-url",
        proxy_v1_url(proxy_root_url),
        "--model",
        selected_model,
    ]
    args = list(argv)
    if mode is MuseInvocationMode.EXEC:
        return ["exec", *route, *args[1:]]
    if mode is MuseInvocationMode.RESUME:
        command_index = _first_root_positional_index(_before_separator(args))
        if command_index is None or args[command_index] != "resume":
            raise ValueError("Muse resume command could not be located")
        return [*args[: command_index + 1], *route, *args[command_index + 1 :]]
    return [*route, *args]


def _first_root_positional_index(argv: Sequence[str]) -> int | None:
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return index + 1 if index + 1 < len(argv) else None
        if argument in _ROOT_VALUE_OPTIONS:
            index += 2
            continue
        if _is_joined_root_value_option(argument):
            index += 1
            continue
        if argument in _ROOT_OPTIONAL_VALUE_OPTIONS:
            if index + 1 < len(argv) and argv[index + 1] in _WORKTREE_VALUES:
                index += 2
            else:
                index += 1
            continue
        if argument in _ROOT_BOOLEAN_OPTIONS or argument in _PASSTHROUGH_FLAGS:
            index += 1
            continue
        if argument.startswith("-"):
            return None
        return index
    return None


def _is_joined_root_value_option(argument: str) -> bool:
    if any(
        argument.startswith(f"{option}=")
        for option in _ROOT_VALUE_OPTIONS | _ROOT_OPTIONAL_VALUE_OPTIONS
        if option.startswith("--")
    ):
        return True
    return argument.startswith("-w") and len(argument) > 2


def _before_separator(argv: Sequence[str]) -> Sequence[str]:
    try:
        return argv[: argv.index("--")]
    except ValueError:
        return argv


def _run(binary_path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    run_client_process(
        command=[binary_path, *args],
        env=env,
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

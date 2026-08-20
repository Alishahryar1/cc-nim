"""Installed `fcc-grok` launcher for attached Grok Build sessions."""

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Never

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

_BINARY_NAME = "grok"
_DISPLAY_NAME = "Grok Build"
_INSTALL_HINT = (
    "Install Grok Build with `irm https://x.ai/cli/install.ps1 | iex` on Windows "
    "or `curl -fsSL https://x.ai/cli/install.sh | bash` on macOS/Linux."
)
_MINIMUM_VERSION = (1, 0, 5)
_VERSION_TIMEOUT_SECONDS = 5.0
_VERSION_PATTERN = re.compile(
    r"(?im)^\s*(?:grok\s+)?v?(\d+)\.(\d+)\.(\d+)(?:\s+\([^\r\n]*\))?\s*$"
)
_PASSTHROUGH_COMMANDS = frozenset(
    {
        "completions",
        "disk-usage",
        "doctor",
        "du",
        "export",
        "login",
        "logout",
        "mcp",
        "memory",
        "plugin",
        "sessions",
        "setup",
        "share",
        "trace",
        "update",
        "v",
        "version",
        "worktree",
        "wrap",
    }
)
_CONFIGURED_COMMANDS = frozenset({"inspect", "models"})
_REJECTED_COMMANDS = frozenset({"dashboard", "leader", "workspace"})
_PASSTHROUGH_FLAGS = frozenset({"-h", "--help", "-v", "-V", "--version"})
_ROUTE_BYPASS_FLAGS = frozenset(
    {
        "--cli-chat-proxy-base-url",
        "--force-login",
        "--leader",
        "--oauth",
        "--xai-api-base-url",
    }
)
_VALUE_OPTIONS = frozenset(
    {
        "--agent",
        "--agent-profile",
        "--agents",
        "--allow",
        "--allowedTools",
        "--append-system-prompt",
        "--background-wait-timeout",
        "--cli-chat-proxy-base-url",
        "--client-identifier",
        "--compaction-detail",
        "--compaction-mode",
        "--cwd",
        "--debug-file",
        "--deny",
        "--disallowed-tools",
        "--disallowedTools",
        "--effort",
        "--hunk-tracker-mode",
        "--installer",
        "--json-schema",
        "--leader-socket",
        "--load",
        "--max-turns",
        "--model",
        "--output-format",
        "--permission-mode",
        "--plugin-dir",
        "--prompt-file",
        "--prompt-json",
        "--reasoning-effort",
        "--ref",
        "--rules",
        "--sandbox",
        "--session-id",
        "--single",
        "--storage-mode",
        "--system-prompt",
        "--system-prompt-override",
        "--tools",
        "--worktree-ref",
        "--xai-api-base-url",
        "-m",
        "-p",
        "-s",
    }
)
_OPTIONAL_VALUE_OPTIONS = frozenset({"--resume", "--worktree", "-r", "-w"})
_PROCESS_CONFIG_KEYS = ("GROK_CONFIG", "GROK_CONFIG_PATH")
_ROUTING_ENV_KEYS = frozenset(
    {
        "GROK_CODE_XAI_API_KEY",
        "GROK_DEFAULT_MODEL",
        "GROK_IMAGE_DESCRIPTION_MODEL",
        "GROK_MODELS_BASE_URL",
        "GROK_MODELS_LIST_URL",
        "GROK_PROMPT_SUGGESTIONS_MODEL",
        "GROK_SESSION_SUMMARY_MODEL",
        "GROK_WEB_SEARCH_MODEL",
        "GROK_XAI_API_BASE_URL",
        "XAI_API_KEY",
    }
)


class GrokInvocationMode(StrEnum):
    """How `fcc-grok` treats one Grok Build invocation."""

    PASSTHROUGH = "passthrough"
    CONFIGURED = "configured"
    ROUTED = "routed"


@dataclass(frozen=True, slots=True)
class GrokInvocation:
    """A minimal command classification without replacing Grok's parser."""

    mode: GrokInvocationMode
    agent_index: int | None = None


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch one attached Grok Build process through FCC."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    invocation = classify_grok_invocation(args)
    if invocation.mode is GrokInvocationMode.PASSTHROUGH:
        _run(binary_path, args, os.environ)
        return

    _reject_process_config_conflicts(os.environ)
    _reject_routing_overrides(args)
    require_compatible_grok(binary_path)

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
        print(
            f"Could not prepare the Grok Build FCC model catalog: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    try:
        selected_model = _selected_model(args, models)
    except ValueError as exc:
        print(f"Invalid Grok Build model selection: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    child_env = build_grok_launcher_env(
        models=models,
        selected_model=selected_model,
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        base_env=os.environ,
    )

    command_args = (
        _args_with_safety_flags(args, invocation)
        if invocation.mode is GrokInvocationMode.ROUTED
        else args
    )
    _run(binary_path, command_args, child_env)


def classify_grok_invocation(argv: Sequence[str]) -> GrokInvocation:
    """Classify only the Grok surfaces whose lifecycle FCC can own."""

    before_separator = _before_separator(argv)
    if any(argument in _PASSTHROUGH_FLAGS for argument in before_separator):
        return GrokInvocation(GrokInvocationMode.PASSTHROUGH)

    positional_index = _first_positional_index(before_separator)
    if positional_index is None:
        return GrokInvocation(GrokInvocationMode.ROUTED)

    command = before_separator[positional_index]
    if command in _PASSTHROUGH_COMMANDS:
        return GrokInvocation(GrokInvocationMode.PASSTHROUGH)
    if command in _CONFIGURED_COMMANDS:
        return GrokInvocation(GrokInvocationMode.CONFIGURED)
    if command in _REJECTED_COMMANDS:
        _detached_surface_error(command)
    if command != "agent":
        return GrokInvocation(GrokInvocationMode.ROUTED)

    mode_index = _first_positional_index(before_separator, start=positional_index + 1)
    mode = before_separator[mode_index] if mode_index is not None else None
    if mode == "stdio":
        return GrokInvocation(GrokInvocationMode.ROUTED, agent_index=positional_index)
    _detached_surface_error(f"agent {mode}" if mode else "agent")


def require_compatible_grok(binary_path: str) -> None:
    """Exit unless Grok Build satisfies FCC's stable custom-model contract."""

    version = grok_binary_version(binary_path)
    if version is not None and version >= _MINIMUM_VERSION:
        return

    minimum = ".".join(str(part) for part in _MINIMUM_VERSION)
    print(
        f"The Grok Build CLI at {binary_path} must be a stable release at least "
        f"version {minimum}.",
        file=sys.stderr,
    )
    print(_INSTALL_HINT, file=sys.stderr)
    raise SystemExit(1)


def grok_binary_version(binary_path: str) -> tuple[int, int, int] | None:
    """Read a stable Grok Build version without invoking a shell."""

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


def build_grok_launcher_env(
    *,
    models: Sequence[ClientModel],
    selected_model: str,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build a child-only Grok route while preserving native Grok state."""

    filtered = {
        key: value
        for key, value in base_env.items()
        if not key.startswith("FCC_GROK_") and key not in _ROUTING_ENV_KEYS
    }
    env = with_local_proxy_bypass(filtered, proxy_root_url=proxy_root_url)
    v1_url = proxy_v1_url(proxy_root_url)
    env["GROK_XAI_API_BASE_URL"] = v1_url
    env["GROK_MODELS_BASE_URL"] = v1_url
    env["GROK_MODELS_LIST_URL"] = f"{v1_url}/models?view=responses"
    env["GROK_DEFAULT_MODEL"] = selected_model
    env["GROK_IMAGE_DESCRIPTION_MODEL"] = selected_model
    env["GROK_PROMPT_SUGGESTIONS_MODEL"] = selected_model
    env["GROK_SESSION_SUMMARY_MODEL"] = selected_model
    env["GROK_WEB_SEARCH_MODEL"] = selected_model
    env["XAI_API_KEY"] = auth_token
    env["GROK_CONFIG"] = json.dumps(
        {
            "models": {"allowed_models": [model.wire_slug for model in models]},
            "shell_environment_policy": {"ignore_default_excludes": False},
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return env


def _selected_model(argv: Sequence[str], models: Sequence[ClientModel]) -> str:
    supplied = _model_override(argv)
    if supplied is None:
        return models[0].wire_slug
    available = {model.wire_slug for model in models}
    if supplied not in available:
        raise ValueError(f"model {supplied!r} is not in the current FCC catalog")
    return supplied


def _model_override(argv: Sequence[str]) -> str | None:
    selected: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            break
        if argument in {"-m", "--model"}:
            if index + 1 >= len(argv) or not argv[index + 1]:
                raise ValueError(f"{argument} requires a model value")
            selected = _merge_model_override(selected, argv[index + 1])
            index += 2
            continue
        if argument.startswith("--model="):
            value = argument.partition("=")[2]
            if not value:
                raise ValueError("--model requires a model value")
            selected = _merge_model_override(selected, value)
            index += 1
            continue
        if argument.startswith("-m") and len(argument) > 2:
            selected = _merge_model_override(selected, argument[2:])
            index += 1
            continue
        if argument in _VALUE_OPTIONS:
            index += 2
            continue
        if argument in _OPTIONAL_VALUE_OPTIONS:
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        index += 1
    return selected


def _merge_model_override(current: str | None, candidate: str) -> str:
    if current is not None and current != candidate:
        raise ValueError("conflicting --model values are not supported")
    return candidate


def _args_with_safety_flags(
    argv: Sequence[str], invocation: GrokInvocation
) -> list[str]:
    args = list(argv)
    agent_index = invocation.agent_index
    if "--disable-web-search" not in _before_separator(args):
        args.insert(0, "--disable-web-search")
        if agent_index is not None:
            agent_index += 1

    if agent_index is None:
        if "--no-leader" not in _before_separator(args):
            args.insert(0, "--no-leader")
        return args

    if "--no-leader" not in _before_separator(args)[agent_index + 1 :]:
        args.insert(agent_index + 1, "--no-leader")
    return args


def _reject_process_config_conflicts(base_env: Mapping[str, str]) -> None:
    for key in _PROCESS_CONFIG_KEYS:
        if base_env.get(key, "").strip():
            print(
                f"{key} is already set. Unset it before running fcc-grok; "
                "ordinary grok remains unchanged.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def _reject_routing_overrides(argv: Sequence[str]) -> None:
    for argument in _before_separator(argv):
        if argument in _ROUTE_BYPASS_FLAGS or any(
            argument.startswith(f"{flag}=") for flag in _ROUTE_BYPASS_FLAGS
        ):
            print(
                "fcc-grok owns the attached Grok Build route. "
                "Use ordinary grok for xAI login, custom endpoints, or leader mode.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def _first_positional_index(argv: Sequence[str], *, start: int = 0) -> int | None:
    index = start
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return index + 1 if index + 1 < len(argv) else None
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
        if argument.startswith("-"):
            index += 1
            continue
        return index
    return None


def _is_joined_value_option(argument: str) -> bool:
    if any(
        argument.startswith(f"{option}=")
        for option in _VALUE_OPTIONS | _OPTIONAL_VALUE_OPTIONS
        if option.startswith("--")
    ):
        return True
    return any(
        argument.startswith(option) and len(argument) > len(option)
        for option in _VALUE_OPTIONS | _OPTIONAL_VALUE_OPTIONS
        if option.startswith("-") and not option.startswith("--")
    )


def _before_separator(argv: Sequence[str]) -> Sequence[str]:
    try:
        return argv[: argv.index("--")]
    except ValueError:
        return argv


def _detached_surface_error(surface: str) -> Never:
    print(
        f"fcc-grok cannot own the lifecycle of Grok Build {surface!r}. "
        "Use ordinary grok for detached, leader, workspace, or dashboard modes.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _run(binary_path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    run_client_process(
        command=[binary_path, *args],
        env=env,
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

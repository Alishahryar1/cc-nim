"""Installed `fcc-dsh` launcher for attached DeepSeek Harness sessions."""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Never

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.core.json_types import JsonValue

from .common import preflight_proxy, resolve_client_binary, run_client_process
from .dsh_config import (
    DSH_API_KEY_ENV,
    DSH_ENV_PREFIX,
    build_dsh_launch_config,
)
from .model_catalog import (
    ClientModel,
    client_models_from_response,
    fetch_proxy_models_response,
)

_BINARY_NAME = "dsh"
_DISPLAY_NAME = "DeepSeek Harness"
_SUPPORTED_VERSION = "0.1.0-rc.8"
_INSTALL_COMMAND = f"npm install -g @deepseek-ai/dsh@{_SUPPORTED_VERSION}"
_INSTALL_HINT = (
    f"Install the supported DeepSeek Harness release with: {_INSTALL_COMMAND}"
)
_VERSION_TIMEOUT_SECONDS = 5.0
_VERSION_PATTERN = re.compile(
    r"(?im)^\s*(?:dsh\s+)?v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?)\s*$"
)
_SUPPORTED_PROFILES = frozenset({"web", "headless"})
_ROOT_HELP_FLAGS = frozenset({"-h", "--help"})
_ROOT_VERSION_FLAGS = frozenset({"-V", "--version"})


@dataclass(frozen=True, slots=True)
class DshInvocation:
    """One minimally classified DSH command line."""

    args: tuple[str, ...]
    patch_index: int | None

    @property
    def is_routed(self) -> bool:
        return self.patch_index is not None


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch one attached DSH Web or headless process through FCC."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    invocation = classify_dsh_invocation(args)
    if not invocation.is_routed:
        _run(binary_path, invocation.args, os.environ)
        return

    require_compatible_dsh(binary_path)
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
    except Exception as exc:
        print(
            f"Could not prepare the DeepSeek Harness FCC model catalog: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    _run_with_dsh_config(
        binary_path=binary_path,
        invocation=invocation,
        models=models,
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        provider_progress_timeout=settings.provider_progress_timeout,
    )


def classify_dsh_invocation(argv: Sequence[str]) -> DshInvocation:
    """Classify only DSH's documented launcher prefix and supported profiles."""

    args = list(argv)
    if not args:
        return DshInvocation(args=("web",), patch_index=1)
    if args[0] == "plugin":
        return DshInvocation(args=tuple(args), patch_index=None)
    if args[0] == "web":
        return _classify_web(args)
    return _classify_profile(args)


def require_compatible_dsh(binary_path: str) -> None:
    """Exit unless DSH exactly matches FCC's audited preview contract."""

    version = dsh_binary_version(binary_path)
    if version == _SUPPORTED_VERSION:
        return

    found = version or "an unrecognized version"
    print(
        f"fcc-dsh requires DeepSeek Harness {_SUPPORTED_VERSION}; found {found}.",
        file=sys.stderr,
    )
    print(f"Install it with: {_INSTALL_COMMAND}", file=sys.stderr)
    raise SystemExit(126)


def dsh_binary_version(binary_path: str) -> str | None:
    """Read DSH's semantic preview version without invoking a shell."""

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
    return match.group(1) if match is not None else None


def build_dsh_launcher_env(
    *,
    auth_token: str,
    proxy_root_url: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build a child-only DSH environment while preserving native state."""

    filtered = {
        key: value
        for key, value in base_env.items()
        if not key.startswith(DSH_ENV_PREFIX)
    }
    env = with_local_proxy_bypass(filtered, proxy_root_url=proxy_root_url)
    env[DSH_API_KEY_ENV] = auth_token
    env["DSH_TELEMETRY_DISABLED"] = "1"
    return env


def _classify_web(args: list[str]) -> DshInvocation:
    patch_index = len(args)
    default_dump = False
    index = 1
    while index < len(args):
        argument = args[index]
        if argument == "--patch" or argument.startswith("--patch="):
            _patch_override_error()
        if argument == "--dump-default-config":
            default_dump = True
            index += 1
            continue
        if argument == "--dump-config":
            index += 1
            continue
        patch_index = index
        break

    if default_dump or (
        patch_index < len(args) and args[patch_index] in _ROOT_HELP_FLAGS
    ):
        return DshInvocation(args=tuple(args), patch_index=None)
    return DshInvocation(args=tuple(args), patch_index=patch_index)


def _classify_profile(args: list[str]) -> DshInvocation:
    profile: str | None = None
    patch_index = len(args)
    default_dump = False
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in _ROOT_VERSION_FLAGS:
            return DshInvocation(args=tuple(args), patch_index=None)
        if argument in _ROOT_HELP_FLAGS and profile is None:
            return DshInvocation(args=tuple(args), patch_index=None)
        if argument == "--profile":
            if index + 1 >= len(args) or not args[index + 1]:
                _profile_value_error()
            profile = args[index + 1]
            index += 2
            continue
        if argument.startswith("--profile="):
            profile = argument.partition("=")[2]
            if not profile:
                _profile_value_error()
            index += 1
            continue
        if argument == "--patch" or argument.startswith("--patch="):
            _patch_override_error()
        if argument == "--dump-default-config":
            default_dump = True
            index += 1
            continue
        if argument == "--dump-config":
            index += 1
            continue
        patch_index = index
        break

    if profile is not None and profile not in _SUPPORTED_PROFILES:
        _unsupported_profile_error(profile)
    if default_dump:
        return DshInvocation(args=tuple(args), patch_index=None)
    if profile is not None:
        if patch_index < len(args) and args[patch_index] in _ROOT_HELP_FLAGS:
            return DshInvocation(args=tuple(args), patch_index=None)
        return DshInvocation(args=tuple(args), patch_index=patch_index)

    normalized = ["web", *args]
    return DshInvocation(args=tuple(normalized), patch_index=1)


def _run_with_dsh_config(
    *,
    binary_path: str,
    invocation: DshInvocation,
    models: tuple[ClientModel, ...],
    proxy_root_url: str,
    auth_token: str,
    provider_progress_timeout: float,
) -> None:
    try:
        temp_config = tempfile.TemporaryDirectory(prefix="fcc-dsh-")
    except OSError as exc:
        _temporary_config_error(exc)

    with temp_config as temp_directory:
        directory = Path(temp_directory)
        patch_path = directory / "fcc.patch.yml"
        settings_path = directory / "settings.yaml"
        credentials_path = directory / ".credentials.yaml"
        try:
            if os.name != "nt":
                directory.chmod(0o700)
            config = build_dsh_launch_config(
                models,
                proxy_root_url=proxy_root_url,
                settings_path=settings_path,
                credentials_path=credentials_path,
                provider_progress_timeout=provider_progress_timeout,
            )
            _write_private_json(patch_path, config.patch)
            _write_private_json(settings_path, {})
            _write_private_json(credentials_path, {})
        except ValueError as exc:
            print(
                f"Could not prepare DeepSeek Harness configuration: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        except OSError as exc:
            _temporary_config_error(exc)

        child_env = build_dsh_launcher_env(
            auth_token=auth_token,
            proxy_root_url=proxy_root_url,
            base_env=os.environ,
        )
        _run(
            binary_path,
            _args_with_patch(invocation, patch_path),
            child_env,
        )


def _args_with_patch(invocation: DshInvocation, patch_path: Path) -> list[str]:
    if invocation.patch_index is None:
        raise ValueError("native DSH passthrough cannot receive an FCC patch")
    args = list(invocation.args)
    args[invocation.patch_index : invocation.patch_index] = ["--patch", str(patch_path)]
    return args


def _write_private_json(path: Path, payload: JsonValue) -> None:
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


def _profile_value_error() -> Never:
    print("fcc-dsh requires a value after --profile.", file=sys.stderr)
    raise SystemExit(2)


def _patch_override_error() -> Never:
    print(
        "fcc-dsh owns the final DeepSeek Harness patch. "
        "Use ordinary dsh for custom profile patches.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _unsupported_profile_error(profile: str) -> Never:
    print(
        f"fcc-dsh supports only the web and headless profiles, not {profile!r}. "
        "Use ordinary dsh for custom profiles.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _temporary_config_error(exc: OSError) -> Never:
    print(
        f"Could not create temporary DeepSeek Harness configuration: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

"""Installed `fcc-aider` launcher for the official Aider CLI."""

import json
import os
import secrets
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Never

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.config.loader import get_settings
from free_claude_code.config.paths import aider_temp_dir_path
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.core.json_types import JsonValue

from .aider_config import (
    AIDER_API_KEY_ENV_PREFIX,
    AiderConfig,
    build_aider_config,
)
from .common import preflight_proxy, resolve_client_binary, run_client_process
from .model_catalog import (
    ClientModel,
    client_models_from_response,
    fetch_proxy_models_response,
)

_BINARY_NAME = "aider"
_DISPLAY_NAME = "Aider"
_INSTALL_HINT = "Install Aider from: https://aider.chat/docs/install.html"
_PASSTHROUGH_FLAGS = frozenset(
    {
        "-h",
        "--help",
        "--version",
        "--just-check-update",
        "--upgrade",
        "--update",
        "--install-main-branch",
        "--shell-completions",
    }
)
_ROUTE_FILE_OPTIONS = ("--model-settings-file", "--model-metadata-file")
_MODEL_OPTIONS = ("--model", "--weak-model", "--editor-model")
_SETTINGS_FILENAME = "model-settings.yml"
_METADATA_FILENAME = "model-metadata.json"
_AIDER_VALIDATION_KEY = "ANTHROPIC_API_KEY=fcc-local"


@dataclass(frozen=True, slots=True)
class AiderConfigFiles:
    """Private file paths owned for the lifetime of one Aider launch."""

    directory: Path
    settings_path: Path
    metadata_path: Path


class _AiderConfigFilesError(RuntimeError):
    """Failure while preparing launcher-owned Aider files."""


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Aider with a process-local FCC Messages model catalog."""

    args = list(sys.argv[1:] if argv is None else argv)
    binary_path = resolve_client_binary(
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

    if is_aider_passthrough(args):
        _run(binary_path, args, os.environ)
        return

    reject_aider_route_file_overrides(args)
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
            fetch_proxy_models_response(
                proxy_root_url,
                auth_token,
                view="messages",
            )
        )
        api_key_env = new_aider_api_key_env_name()
        config = build_aider_config(
            models,
            messages_url=f"{proxy_root_url.rstrip('/')}/v1/messages",
            api_key_env=api_key_env,
        )
    except Exception as exc:
        print(f"Could not prepare the Aider FCC model catalog: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    try:
        with temporary_aider_config_files(config) as files:
            child_args = normalized_aider_arguments(
                args,
                models=models,
                settings_path=files.settings_path,
                metadata_path=files.metadata_path,
            )
            child_env = build_aider_launcher_env(
                proxy_root_url=proxy_root_url,
                api_key_env=api_key_env,
                auth_token=auth_token,
                base_env=os.environ,
            )
            _run(binary_path, child_args, child_env)
    except _AiderConfigFilesError as exc:
        print(f"Could not create temporary Aider configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def is_aider_passthrough(argv: Sequence[str]) -> bool:
    """Return whether Aider can run without FCC-owned process configuration."""

    return any(argument in _PASSTHROUGH_FLAGS for argument in _before_separator(argv))


def reject_aider_route_file_overrides(argv: Sequence[str]) -> None:
    """Reject command-line paths that would replace FCC's process overlay."""

    for argument in _before_separator(argv):
        if argument in _ROUTE_FILE_OPTIONS or any(
            argument.startswith(f"{option}=") for option in _ROUTE_FILE_OPTIONS
        ):
            print(
                "fcc-aider owns the model settings and metadata files for its FCC "
                "route. Use ordinary aider or Aider's normal home/project model "
                "files for other route/catalog customization.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def normalized_aider_arguments(
    argv: Sequence[str],
    *,
    models: tuple[ClientModel, ...],
    settings_path: Path,
    metadata_path: Path,
) -> list[str]:
    """Replace Aider's three model roles with canonical FCC model names."""

    if not models:
        raise ValueError("Aider requires at least one routable FCC model")

    before, separator_and_after = _split_separator(argv)
    selected: dict[str, str] = {}
    remaining: list[str] = []
    index = 0
    while index < len(before):
        argument = before[index]
        option = next(
            (
                candidate
                for candidate in _MODEL_OPTIONS
                if argument == candidate or argument.startswith(f"{candidate}=")
            ),
            None,
        )
        if option is None:
            remaining.append(argument)
            index += 1
            continue
        if option in selected:
            _model_argument_error(f"{option} may be provided only once")

        if argument == option:
            value_index = index + 1
            if value_index >= len(before) or before[value_index].startswith("-"):
                _model_argument_error(f"{option} requires one model value")
            value = before[value_index]
            index += 2
        else:
            value = argument.removeprefix(f"{option}=")
            index += 1
        if not value.strip():
            _model_argument_error(f"{option} requires one model value")
        selected[option] = value

    canonical_by_input: dict[str, str] = {}
    for model in models:
        canonical = f"anthropic/{model.wire_slug}"
        canonical_by_input[model.wire_slug] = canonical
        canonical_by_input[canonical] = canonical

    resolved: dict[str, str] = {}
    for option, value in selected.items():
        canonical = canonical_by_input.get(value)
        if canonical is None:
            _model_argument_error(
                f"{option} model {value!r} is not in the current FCC model catalog"
            )
        resolved[option] = canonical

    main_model = resolved.get("--model", f"anthropic/{models[0].wire_slug}")
    weak_model = resolved.get("--weak-model", main_model)
    editor_model = resolved.get("--editor-model", main_model)
    owned = [
        "--model",
        main_model,
        "--weak-model",
        weak_model,
        "--editor-model",
        editor_model,
        "--model-settings-file",
        str(settings_path),
        "--model-metadata-file",
        str(metadata_path),
        "--set-env",
        _AIDER_VALIDATION_KEY,
    ]
    return [*remaining, *owned, *separator_and_after]


def new_aider_api_key_env_name() -> str:
    """Return one unguessable process-local environment variable name."""

    return f"{AIDER_API_KEY_ENV_PREFIX}{secrets.token_hex(16).upper()}"


def build_aider_launcher_env(
    *,
    proxy_root_url: str,
    api_key_env: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Build Aider's child-only environment while preserving native state."""

    prefix = AIDER_API_KEY_ENV_PREFIX.casefold()
    clean_env = {
        key: value
        for key, value in base_env.items()
        if not key.casefold().startswith(prefix)
    }
    env = with_local_proxy_bypass(clean_env, proxy_root_url=proxy_root_url)
    env[api_key_env] = auth_token
    return env


@contextmanager
def temporary_aider_config_files(
    config: AiderConfig,
) -> Iterator[AiderConfigFiles]:
    """Write one private Aider config pair and remove its launch directory."""

    with ExitStack() as stack:
        try:
            base_directory = aider_temp_dir_path()
            base_directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            if os.name != "nt":
                base_directory.chmod(0o700)
            temp_directory = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="fcc-aider-", dir=base_directory)
            )
            directory = Path(temp_directory)
            if os.name != "nt":
                directory.chmod(0o700)
            files = AiderConfigFiles(
                directory=directory,
                settings_path=directory / _SETTINGS_FILENAME,
                metadata_path=directory / _METADATA_FILENAME,
            )
            _write_private_json(files.settings_path, config.settings)
            _write_private_json(files.metadata_path, config.metadata)
        except OSError as exc:
            raise _AiderConfigFilesError(str(exc)) from exc
        yield files


def _write_private_json(path: Path, payload: JsonValue) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=True, indent=2)
        output.write("\n")


def _before_separator(argv: Sequence[str]) -> Sequence[str]:
    try:
        return argv[: argv.index("--")]
    except ValueError:
        return argv


def _split_separator(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    try:
        index = argv.index("--")
    except ValueError:
        return list(argv), []
    return list(argv[:index]), list(argv[index:])


def _model_argument_error(message: str) -> Never:
    print(f"Invalid fcc-aider model option: {message}.", file=sys.stderr)
    raise SystemExit(2)


def _run(binary_path: str, args: Sequence[str], env: Mapping[str, str]) -> None:
    run_client_process(
        command=[binary_path, *args],
        env=env,
        binary_name=_BINARY_NAME,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )

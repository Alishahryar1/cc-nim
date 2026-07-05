"""Dotenv file discovery and explicit dotenv override helpers."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .paths import managed_env_path

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"


def repo_env_path() -> Path:
    """Return the repo-local env path."""

    return Path(".env")


def explicit_env_path(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the explicit FCC_ENV_FILE path, when configured."""

    source = env if env is not None else os.environ
    if explicit := source.get("FCC_ENV_FILE"):
        return Path(explicit)
    return None


def settings_env_files(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Return Settings dotenv files in low-to-high precedence order."""

    files: list[Path] = [
        repo_env_path(),
        managed_env_path(),
    ]
    if explicit := explicit_env_path(env):
        files.append(explicit)
    return tuple(files)


def configured_env_files(model_config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the env files currently configured for a Settings model."""

    configured = model_config.get("env_file")
    if configured is None:
        return ()
    if isinstance(configured, (str, Path)):
        return (Path(configured),)
    return tuple(Path(item) for item in configured)


def env_file_value(path: Path, key: str) -> str | None:
    """Return a dotenv value when the file explicitly defines the key."""

    if not path.is_file():
        return None

    try:
        values = dotenv_values(path)
    except OSError:
        return None

    if key not in values:
        return None
    value = values[key]
    return "" if value is None else value


def env_file_override(model_config: Mapping[str, Any], key: str) -> str | None:
    """Return the last configured dotenv value that explicitly defines a key."""

    configured_value: str | None = None
    for env_file in configured_env_files(model_config):
        value = env_file_value(env_file, key)
        if value is not None:
            configured_value = value
    return configured_value


def process_env_key_is_effective(
    model_config: Mapping[str, Any],
    key: str,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a key is coming from process env instead of configured dotenv."""

    source = env if env is not None else os.environ
    if env_file_override(model_config, key) is not None:
        return False
    return key in source

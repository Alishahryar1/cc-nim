"""Admin config source loading and source precedence."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

from config.env_files import (
    explicit_env_path as configured_explicit_env_path,
)
from config.env_files import (
    repo_env_path as configured_repo_env_path,
)
from config.env_files import (
    settings_env_files,
)
from config.env_template import load_env_template_or_empty

from .manifest import FIELDS

SourceType = Literal[
    "default",
    "template",
    "repo_env",
    "managed_env",
    "explicit_env_file",
    "process",
]


def repo_env_path() -> Path:
    """Return the repo-local env path."""

    return configured_repo_env_path()


def explicit_env_path() -> Path | None:
    """Return the explicit FCC_ENV_FILE path, when configured."""

    return configured_explicit_env_path(os.environ)


def configured_env_files() -> tuple[tuple[SourceType, Path], ...]:
    """Return dotenv files in low-to-high precedence order."""

    source_names: tuple[SourceType, ...] = (
        "repo_env",
        "managed_env",
        "explicit_env_file",
    )
    return tuple(zip(source_names, settings_env_files(), strict=False))


def dotenv_values_from_text(text: str) -> dict[str, str]:
    """Parse dotenv text into string values."""

    values = dotenv_values(stream=StringIO(text))
    return {key: "" if value is None else value for key, value in values.items()}


def template_values() -> dict[str, str]:
    """Return .env.example values plus manifest defaults for newer fields."""

    values = dotenv_values_from_text(load_env_template_or_empty())
    for field in FIELDS:
        values.setdefault(field.key, field.default)
    return values


def dotenv_values_from_file(path: Path) -> dict[str, str]:
    """Return dotenv values from a file, or an empty mapping when absent."""

    if not path.is_file():
        return {}
    values = dotenv_values(path)
    return {key: "" if value is None else value for key, value in values.items()}


def is_locked_source(source: SourceType) -> bool:
    """Return whether an admin value source must not be overwritten."""

    return source in {"process", "explicit_env_file"}

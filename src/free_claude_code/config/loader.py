"""Canonical managed-config loading, precedence, provenance, and caching."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType

from free_claude_code.core.interprocess_lock import InterprocessFileLock

from .env_files import ANTHROPIC_AUTH_TOKEN_ENV, dotenv_values_from_file
from .env_migrations import consolidate_managed_config, settings_env_keys
from .paths import config_lock_path, managed_env_path
from .settings import Settings


class ConfigSource(StrEnum):
    """Live owner of one effective setting value."""

    DEFAULT = "default"
    MANAGED = "managed_env"
    PROCESS = "process"


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """One validated settings model and its effective field provenance."""

    settings: Settings
    sources: Mapping[str, ConfigSource]


def resolve_settings_snapshot(
    env: Mapping[str, str] | None = None,
) -> SettingsSnapshot:
    """Resolve one uncached Settings snapshot from managed and process state."""

    process = env if env is not None else os.environ
    lock = InterprocessFileLock(config_lock_path())
    if not lock.acquire(wait=True, timeout=10.0):
        raise TimeoutError(
            f"Could not acquire managed-config lock: {config_lock_path()}"
        )
    try:
        consolidate_managed_config(process)
    finally:
        lock.release()

    managed_path = managed_env_path()
    managed = dotenv_values_from_file(managed_path) if managed_path.is_file() else {}
    return compose_settings_snapshot(managed, process)


def compose_settings_snapshot(
    managed: Mapping[str, str],
    env: Mapping[str, str] | None = None,
) -> SettingsSnapshot:
    """Validate prospective managed values with live process precedence."""

    process = env if env is not None else os.environ
    aliases = _settings_aliases()
    recognized = settings_env_keys()
    values: dict[str, str] = {
        key: value for key, value in managed.items() if key in recognized
    }
    sources = dict.fromkeys(Settings.model_fields, ConfigSource.DEFAULT)
    for key in values:
        if name := aliases.get(key):
            sources[name] = ConfigSource.MANAGED

    managed_owns_token = bool(values.get(ANTHROPIC_AUTH_TOKEN_ENV, "").strip())
    for key in recognized:
        if key not in process:
            continue
        if key == ANTHROPIC_AUTH_TOKEN_ENV and managed_owns_token:
            continue
        value = process[key]
        if key == ANTHROPIC_AUTH_TOKEN_ENV and not value.strip():
            values.pop(key, None)
            sources[aliases[key]] = ConfigSource.DEFAULT
            continue
        values[key] = value
        if name := aliases.get(key):
            sources[name] = ConfigSource.PROCESS

    settings = Settings.model_validate(values)
    return SettingsSnapshot(
        settings=settings,
        sources=MappingProxyType(sources),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings model."""

    return resolve_settings_snapshot().settings


def clear_settings_cache() -> None:
    """Discard the runtime Settings cache after a committed configuration change."""

    get_settings.cache_clear()


def _settings_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, field in Settings.model_fields.items():
        if name == "nim":
            continue
        alias = field.validation_alias
        if not isinstance(alias, str):
            raise AssertionError(f"Settings field {name!r} needs one string alias")
        aliases[alias] = name
    return aliases

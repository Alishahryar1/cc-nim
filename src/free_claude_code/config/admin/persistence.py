"""Sparse managed-config validation, preview, and atomic persistence."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from free_claude_code.config.env_files import (
    FCC_CONFIG_SCHEMA_ENV,
    dotenv_values_from_file,
)
from free_claude_code.config.env_migrations import (
    CONFIG_SCHEMA_VERSION,
    atomic_write_managed_config,
    recognized_env_keys,
    render_managed_config,
)
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject

from .manifest import FIELD_BY_KEY
from .state import ConfigInputValue
from .validation import settings_from_values
from .values import MASKED_SECRET, is_locked_source, load_value_state, normalize_for_env

_PROVIDER_PROXY_ATTRS = frozenset(
    descriptor.proxy_attr
    for descriptor in PROVIDER_CATALOG.values()
    if descriptor.proxy_attr is not None
)
_PROVIDER_PROXY_KEYS = tuple(
    field.key
    for field in FIELD_BY_KEY.values()
    if field.settings_attr in _PROVIDER_PROXY_ATTRS
)
_PROVIDER_PROXY_ERROR = (
    "must be a proxy URL with a supported scheme and host "
    "(for example http://127.0.0.1:8080 or socks5://127.0.0.1:1080)"
)


@dataclass(frozen=True, slots=True)
class PreparedAdminUpdate:
    """Validated Admin update ready for an atomic managed-file commit."""

    target_values: dict[str, str]
    settings: Settings | None
    errors: tuple[str, ...]
    pending_fields: tuple[str, ...]
    path: Path

    @property
    def valid(self) -> bool:
        return self.settings is not None and not self.errors

    def validation_response(self) -> JsonObject:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "env_preview": render_managed_config(
                self.target_values,
                mask_secrets=True,
            ),
        }

    def applied_response(self) -> JsonObject:
        if not self.valid:
            return self.validation_response() | {
                "applied": False,
                "pending_fields": [],
            }
        return {
            "applied": True,
            "valid": True,
            "errors": [],
            "env_preview": render_managed_config(
                self.target_values,
                mask_secrets=True,
            ),
            "path": str(self.path),
            "pending_fields": list(self.pending_fields),
        }


def target_values_with_updates(
    updates: Mapping[str, ConfigInputValue],
) -> dict[str, str]:
    """Return sparse managed state after applying valid partial-update semantics."""

    state = load_value_state()
    path = managed_env_path()
    values = dotenv_values_from_file(path) if path.is_file() else {}
    known = recognized_env_keys()
    for key in tuple(values):
        if key in known and key != FCC_CONFIG_SCHEMA_ENV and not values[key].strip():
            values.pop(key)

    for key, submitted in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or is_locked_source(state[key].source):
            continue
        if field.secret and (
            submitted == MASKED_SECRET
            or (isinstance(submitted, str) and not submitted.strip())
        ):
            continue
        value = normalize_for_env(submitted)
        if value is None or (field.nullable and not value):
            values.pop(key, None)
        else:
            values[key] = value

    values[FCC_CONFIG_SCHEMA_ENV] = CONFIG_SCHEMA_VERSION
    return values


def validate_updates(updates: Mapping[str, ConfigInputValue]) -> JsonObject:
    """Validate partial Admin updates and return a masked sparse preview."""

    return prepare_admin_update(updates).validation_response()


def changed_pending_fields(
    updates: Mapping[str, ConfigInputValue],
    *,
    settings: Settings,
) -> list[str]:
    """Return changed fields that require manual runtime action."""

    state = load_value_state()
    pending: list[str] = []
    for key, submitted in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or is_locked_source(state[key].source):
            continue
        if field.secret and (
            submitted == MASKED_SECRET
            or (isinstance(submitted, str) and not submitted.strip())
        ):
            continue
        requires_restart = field.restart_required or field.session_sensitive
        if not requires_restart:
            requires_restart = _active_voice_credential(settings) == key
        if not requires_restart:
            continue
        if normalize_for_env(submitted) == state[key].value:
            continue
        pending.append(key)
    return pending


def prepare_admin_update(
    updates: Mapping[str, ConfigInputValue],
) -> PreparedAdminUpdate:
    """Validate an update and construct its prospective Settings snapshot."""

    update_errors = _update_protocol_errors(updates)
    target_values = target_values_with_updates(updates)
    settings, settings_errors = settings_from_values(target_values)
    errors = (
        *update_errors,
        *settings_errors,
        *_provider_proxy_errors(target_values),
    )
    pending_fields = (
        tuple(changed_pending_fields(updates, settings=settings))
        if settings is not None and not errors
        else ()
    )
    return PreparedAdminUpdate(
        target_values=target_values,
        settings=settings,
        errors=errors,
        pending_fields=pending_fields,
        path=managed_env_path(),
    )


def commit_prepared_admin_update(prepared: PreparedAdminUpdate) -> JsonObject:
    """Atomically persist a previously validated Admin update."""

    if not prepared.valid:
        raise ValueError("Cannot commit an invalid Admin update")
    atomic_write_managed_config(prepared.target_values, path=prepared.path)
    return prepared.applied_response()


def _update_protocol_errors(
    updates: Mapping[str, ConfigInputValue],
) -> tuple[str, ...]:
    errors: list[str] = []
    for key, submitted in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or (field.secret and submitted == MASKED_SECRET):
            continue
        if submitted is None and not field.nullable:
            errors.append(f"{key}: this setting cannot be removed")
            continue
        if (
            not field.secret
            and isinstance(submitted, str)
            and not submitted.strip()
            and not field.nullable
        ):
            errors.append(f"{key}: this setting cannot be blank")
    return tuple(errors)


def _provider_proxy_errors(values: Mapping[str, str]) -> tuple[str, ...]:
    errors: list[str] = []
    for key in _PROVIDER_PROXY_KEYS:
        value = values.get(key)
        if not value:
            continue
        try:
            proxy = httpx.Proxy(value)
        except httpx.InvalidURL, ValueError:
            errors.append(f"{key}: {_PROVIDER_PROXY_ERROR}")
            continue
        if not proxy.url.host:
            errors.append(f"{key}: {_PROVIDER_PROXY_ERROR}")
    return tuple(errors)


def _active_voice_credential(settings: Settings) -> str | None:
    if not settings.voice_note_enabled:
        return None
    if settings.whisper_device == "nvidia_nim":
        return "NVIDIA_NIM_API_KEY"
    return "HUGGINGFACE_API_KEY"

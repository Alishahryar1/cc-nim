"""Admin config value state and API response assembly."""

import os
from enum import Enum

from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.loader import ConfigSource, resolve_settings_snapshot
from free_claude_code.config.paths import managed_env_path
from free_claude_code.core.json_types import JsonObject

from .manifest import FIELDS, SECTIONS
from .specs import ConfigFieldSpec, ConfigOptionSpec
from .state import ConfigValueState, ValueState
from .status import provider_config_status

MASKED_SECRET = "********"


def normalize_for_env(value: object) -> str | None:
    """Normalize a submitted Admin value for sparse dotenv persistence."""

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value).strip()


def display_value(field: ConfigFieldSpec, value: str | None) -> str | None:
    """Return the Admin UI display value for a canonical config value."""

    if field.secret and value is not None:
        return MASKED_SECRET
    return value


def is_locked_source(source: str | ConfigSource) -> bool:
    """Return whether process ownership makes an Admin field read-only."""

    return str(source) == ConfigSource.PROCESS.value


def load_value_state() -> ValueState:
    """Load effective Admin values from the canonical Settings resolver."""

    snapshot = resolve_settings_snapshot()
    managed_path = managed_env_path()
    managed = dotenv_values_from_file(managed_path) if managed_path.is_file() else {}
    state: ValueState = {}
    for field in FIELDS:
        if field.settings_attr is not None:
            value = normalize_for_env(getattr(snapshot.settings, field.settings_attr))
            source = snapshot.sources[field.settings_attr].value
        elif field.key in os.environ:
            value = os.environ[field.key].strip() or None
            source = ConfigSource.PROCESS.value
        elif field.key in managed:
            value = managed[field.key].strip() or None
            source = ConfigSource.MANAGED.value
        else:
            value = field.resolved_default()
            source = ConfigSource.DEFAULT.value
        state[field.key] = ConfigValueState(value=value, source=source)
    return state


def load_config_response() -> JsonObject:
    """Return manifest and current config values for the Admin UI."""

    state = load_value_state()
    fields: list[JsonObject] = []
    for field in FIELDS:
        entry = state[field.key]
        source = entry.source
        raw_value = entry.value
        fields.append(
            {
                "key": field.key,
                "label": field.label,
                "section": field.section_id,
                "type": field.field_type,
                "value": display_value(field, raw_value),
                "configured": raw_value is not None,
                "source": source,
                "locked": is_locked_source(source),
                "nullable": field.nullable,
                "secret": field.secret,
                "advanced": field.advanced,
                "restart_required": field.restart_required,
                "session_sensitive": field.session_sensitive,
                "options": [
                    (
                        {"value": option.value, "label": option.label}
                        if isinstance(option, ConfigOptionSpec)
                        else {"value": option, "label": option}
                    )
                    for option in field.options
                ],
                "description": field.description,
            }
        )

    return {
        "sections": [
            {
                "id": section.section_id,
                "label": section.label,
                "description": section.description,
                "advanced": section.advanced,
            }
            for section in SECTIONS
        ],
        "fields": fields,
        "paths": {"managed": str(managed_env_path())},
        "provider_status": provider_config_status(state),
    }

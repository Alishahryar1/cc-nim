"""Admin config value state and API response assembly."""

import os
from typing import Any

from config.paths import managed_env_path

from .manifest import FIELD_BY_KEY, FIELDS, SECTIONS, ConfigFieldSpec
from .sources import (
    configured_env_files,
    dotenv_values_from_file,
    explicit_env_path,
    is_locked_source,
    repo_env_path,
    template_values,
)
from .status import provider_config_status

MASKED_SECRET = "********"
ValueState = dict[str, dict[str, Any]]


def normalize_for_env(value: Any) -> str:
    """Normalize a submitted admin value for dotenv persistence."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def display_value(field: ConfigFieldSpec, value: str) -> str:
    """Return the Admin UI display value for a raw config value."""

    if field.secret and value:
        return MASKED_SECRET
    return value


def load_value_state() -> ValueState:
    """Load effective admin field values and their sources."""

    values = template_values()
    sources = {key: "template" if key in values else "default" for key in FIELD_BY_KEY}

    for source, path in configured_env_files():
        file_values = dotenv_values_from_file(path)
        for key, value in file_values.items():
            if key in FIELD_BY_KEY:
                values[key] = value
                sources[key] = source

    for key in FIELD_BY_KEY:
        if key in os.environ:
            values[key] = os.environ[key]
            sources[key] = "process"

    return {
        key: {
            "value": values.get(key, ""),
            "source": sources.get(key, "default"),
        }
        for key in FIELD_BY_KEY
    }


def load_config_response() -> dict[str, Any]:
    """Return manifest and current config values for the admin UI."""

    state = load_value_state()
    fields: list[dict[str, Any]] = []
    for field in FIELDS:
        entry = state[field.key]
        source = entry["source"]
        raw_value = entry["value"]
        fields.append(
            {
                "key": field.key,
                "label": field.label,
                "section": field.section_id,
                "type": field.field_type,
                "value": display_value(field, raw_value),
                "configured": bool(str(raw_value).strip()),
                "source": source,
                "locked": is_locked_source(source),
                "secret": field.secret,
                "advanced": field.advanced,
                "restart_required": field.restart_required,
                "session_sensitive": field.session_sensitive,
                "options": list(field.options),
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
        "paths": {
            "managed": str(managed_env_path()),
            "repo": str(repo_env_path()),
            "explicit": str(explicit_env_path()) if explicit_env_path() else None,
        },
        "provider_status": provider_config_status(state),
    }

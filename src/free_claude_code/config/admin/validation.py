"""Settings-backed Admin config validation."""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from free_claude_code.config.settings import Settings

from .manifest import FIELDS, field_input_key


def validate_values(values: Mapping[str, str]) -> tuple[bool, list[str]]:
    """Validate proposed env values against the Settings model."""

    settings, errors = settings_from_values(values)
    return settings is not None, errors


def settings_from_values(
    values: Mapping[str, str],
) -> tuple[Settings | None, list[str]]:
    """Build the prospective Settings snapshot without reading dotenv files."""

    kwargs: dict[str, Any] = {"_env_file": None}
    for field in FIELDS:
        input_key = field_input_key(field)
        if input_key is None:
            continue
        kwargs[input_key] = values.get(field.key, "")

    try:
        return Settings(**kwargs), []
    except ValidationError as exc:
        return None, format_validation_errors(exc)


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Return user-readable validation errors from a Pydantic exception."""

    errors: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "Invalid value"))
        errors.append(f"{loc}: {message}" if loc else message)
    return errors

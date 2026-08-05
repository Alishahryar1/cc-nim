"""Managed env persistence, validation preview, and rendering."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from free_claude_code.config.model_refs import RESTRICTED_EMPTY_MARKER
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.settings import Settings

from .manifest import FIELD_BY_KEY, FIELDS, SECTIONS, ConfigFieldSpec
from .sources import dotenv_values_from_file, is_locked_source, template_values
from .validation import settings_from_values
from .values import MASKED_SECRET, load_value_state, normalize_for_env


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
        return self.settings is not None

    def validation_response(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "env_preview": render_env_file(self.target_values, mask_secrets=True),
        }

    def applied_response(self) -> dict[str, Any]:
        if not self.valid:
            return self.validation_response() | {
                "applied": False,
                "pending_fields": [],
            }
        return {
            "applied": True,
            "valid": True,
            "errors": [],
            "env_preview": render_env_file(
                self.target_values,
                mask_secrets=True,
            ),
            "path": str(self.path),
            "pending_fields": list(self.pending_fields),
        }


def target_values_with_updates(updates: Mapping[str, Any]) -> dict[str, str]:
    """Return managed env values after applying admin updates."""

    state = load_value_state()
    values = template_values()

    # Preserve existing managed values when present. If no managed config exists,
    # seed the first write from effective repo values to migrate legacy setups.
    managed_values = dotenv_values_from_file(managed_env_path())
    if managed_values:
        values.update(
            {key: val for key, val in managed_values.items() if key in values}
        )
    else:
        for key, entry in state.items():
            if entry["source"] in {"repo_env", "template", "default"}:
                values[key] = str(entry["value"])

    for key, value in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None:
            continue
        if is_locked_source(state[key]["source"]):
            continue
        if field.secret and value == MASKED_SECRET:
            continue
        values[key] = normalize_for_env(value)

    for field in FIELDS:
        values.setdefault(field.key, field.default)
    return values


def effective_values_for_validation(
    target_values: Mapping[str, str],
) -> dict[str, str]:
    """Return values validated after preserving locked external sources."""

    values = dict(target_values)
    for key, entry in load_value_state().items():
        if is_locked_source(entry["source"]):
            values[key] = str(entry["value"])
    return values


def validate_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Validate partial admin updates and return a masked generated env preview."""

    return prepare_admin_update(updates).validation_response()


def changed_pending_fields(
    updates: Mapping[str, Any],
    *,
    settings: Settings,
) -> list[str]:
    """Return changed fields that require manual runtime action."""

    state = load_value_state()
    pending: list[str] = []
    for key, value in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or is_locked_source(state[key]["source"]):
            continue
        if field.secret and value == MASKED_SECRET:
            continue
        requires_restart = field.restart_required or field.session_sensitive
        if not requires_restart:
            requires_restart = _active_voice_credential(settings) == key
        if not requires_restart:
            continue
        if normalize_for_env(value) == str(state[key]["value"]):
            continue
        pending.append(key)
    return pending


def _active_voice_credential(settings: Settings) -> str | None:
    if not settings.voice_note_enabled:
        return None
    if settings.whisper_device == "nvidia_nim":
        return "NVIDIA_NIM_API_KEY"
    return "HUGGINGFACE_API_KEY"


def prepare_admin_update(updates: Mapping[str, Any]) -> PreparedAdminUpdate:
    """Validate an update and construct its prospective Settings snapshot."""

    target_values = target_values_with_updates(updates)
    effective_values = effective_values_for_validation(target_values)
    settings, errors = settings_from_values(effective_values)
    pending_fields = (
        tuple(changed_pending_fields(updates, settings=settings))
        if settings is not None
        else ()
    )
    return PreparedAdminUpdate(
        target_values=target_values,
        settings=settings,
        errors=tuple(errors),
        pending_fields=pending_fields,
        path=managed_env_path(),
    )


def commit_prepared_admin_update(prepared: PreparedAdminUpdate) -> dict[str, Any]:
    """Atomically persist a previously validated Admin update."""

    if not prepared.valid:
        raise ValueError("Cannot commit an invalid Admin update")

    path = prepared.path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(
            render_env_file(prepared.target_values),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return prepared.applied_response()


ALLOWLIST_ENV_KEY = "FCC_PROVIDER_MODEL_ALLOWLIST"


def render_provider_allowlists(allowlists: Mapping[str, list[str]]) -> str:
    """Render per-provider allowlists into the FCC_PROVIDER_MODEL_ALLOWLIST value."""

    parts: list[str] = []
    for provider_id in sorted(allowlists):
        models = sorted(allowlists[provider_id])
        if models:
            parts.extend(f"{provider_id}/{model}" for model in models)
        else:
            parts.append(f"{provider_id}/{RESTRICTED_EMPTY_MARKER}")
    return ",".join(parts)


def parse_provider_allowlists(value: str) -> dict[str, list[str]]:
    """Parse the FCC_PROVIDER_MODEL_ALLOWLIST value into provider -> models.

    A provider restricted to zero models is stored with the internal
    RESTRICTED_EMPTY_MARKER and parsed back as an empty list so the restriction
    survives a managed-env round-trip.
    """

    result: dict[str, list[str]] = {}
    for item in value.split(","):
        item = item.strip()
        if not item or "/" not in item:
            continue
        provider_id, model = item.split("/", 1)
        provider_id = provider_id.strip()
        model = model.strip()
        if model == RESTRICTED_EMPTY_MARKER:
            result.setdefault(provider_id, [])
        else:
            result.setdefault(provider_id, []).append(model)
    return result


def load_provider_allowlists() -> dict[str, list[str]]:
    """Return per-provider allowlists persisted in the managed env file."""

    managed = dotenv_values_from_file(managed_env_path())
    return parse_provider_allowlists(managed.get(ALLOWLIST_ENV_KEY, ""))


def commit_provider_allowlist(
    provider_id: str,
    models: list[str],
    *,
    restricted: bool = True,
) -> dict[str, Any]:
    """Persist one provider's allowlist into the managed env file.

    When ``restricted`` is True the provider is limited to ``models`` (an empty
    list disables the provider). When False the provider entry is removed so all
    of its models are allowed again.
    """

    allowlists = load_provider_allowlists()
    if restricted:
        allowlists[provider_id] = sorted(
            model.strip() for model in models if model.strip()
        )
    else:
        allowlists.pop(provider_id, None)

    path = managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    managed = dotenv_values_from_file(path)
    managed[ALLOWLIST_ENV_KEY] = render_provider_allowlists(allowlists)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(render_env_file(managed), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return {
        "applied": True,
        "provider_id": provider_id,
        "models": allowlists.get(provider_id, []),
        "restricted": provider_id in allowlists,
    }


def quote_env_value(value: str) -> str:
    """Quote a value when dotenv syntax requires it."""

    if value == "":
        return ""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    if any(char.isspace() for char in value) or any(
        char in value for char in ('"', "#", "=", "$")
    ):
        return f'"{escaped}"'
    return value


def render_env_file(values: Mapping[str, str], *, mask_secrets: bool = False) -> str:
    """Render a complete grouped env file."""

    lines: list[str] = [
        "# Managed by Free Claude Code /admin.",
        "# Edit in the server UI when possible.",
        "",
    ]
    fields_by_section: dict[str, list[ConfigFieldSpec]] = {
        section.section_id: [] for section in SECTIONS
    }
    for field in FIELDS:
        fields_by_section.setdefault(field.section_id, []).append(field)

    for section in SECTIONS:
        lines.append(f"# {section.label}")
        for field in fields_by_section.get(section.section_id, []):
            value = values.get(field.key, field.default)
            if mask_secrets and field.secret and value:
                value = MASKED_SECRET
            lines.append(f"{field.key}={quote_env_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

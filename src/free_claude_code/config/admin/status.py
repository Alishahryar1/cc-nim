"""Provider configuration status for the Admin UI."""

from collections.abc import Mapping
from typing import Any

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)

from .manifest import FIELDS


def provider_config_status(
    state: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return provider configuration status without making network calls."""
    statuses: list[dict[str, Any]] = []
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT:
            statuses.append(
                {
                    "provider_id": provider_id,
                    "display_name": descriptor.display_name,
                    "kind": "connected_account",
                    "status": "disconnected",
                    "label": "Not connected",
                }
            )
            continue
        if descriptor.local:
            base_url: str | None = None
            if descriptor.base_url_attr is not None:
                base_url = _value_for_settings_attr(state, descriptor.base_url_attr)
            statuses.append(
                {
                    "provider_id": provider_id,
                    "display_name": descriptor.display_name,
                    "kind": "local",
                    "status": "missing_url" if not base_url else "unknown",
                    "label": "Missing URL" if not base_url else "Not checked",
                    "base_url": base_url or descriptor.default_base_url or "",
                }
            )
            continue

        configuration_attrs = descriptor.configuration_attrs()
        missing_attrs = tuple(
            attr
            for attr in configuration_attrs
            if not _value_for_settings_attr(state, attr)
        )
        configured = not missing_attrs
        configuration = " + ".join(
            _field_key_for_settings_attr(attr) for attr in configuration_attrs
        )
        missing_key = descriptor.credential_attr in missing_attrs
        statuses.append(
            {
                "provider_id": provider_id,
                "display_name": descriptor.display_name,
                "kind": "remote",
                "status": (
                    "configured"
                    if configured
                    else "missing_key"
                    if missing_key
                    else "missing_config"
                ),
                "label": (
                    "Configured"
                    if configured
                    else "Missing key"
                    if missing_key
                    else "Missing configuration"
                ),
                "configuration": configuration,
            }
        )
    return statuses


def _value_for_settings_attr(
    state: Mapping[str, Mapping[str, Any]], settings_attr: str
) -> str | None:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            value = state.get(field.key, {}).get("value", field.resolved_default())
            return str(value) if value is not None else None
    return None


def _field_key_for_settings_attr(settings_attr: str) -> str:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            return field.key
    raise AssertionError(f"No admin field owns settings attribute {settings_attr!r}")

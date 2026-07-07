"""Provider configuration status for the Admin UI."""

from collections.abc import Mapping
from typing import Any

from config.provider_catalog import PROVIDER_CATALOG

from .manifest import FIELDS


def provider_config_status(
    state: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return provider configuration status without making network calls."""

    if state is None:
        from .values import load_value_state

        state = load_value_state()
    statuses: list[dict[str, Any]] = []
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if descriptor.credential_env is None:
            base_url = ""
            if descriptor.base_url_attr is not None:
                base_url = _value_for_settings_attr(state, descriptor.base_url_attr)
            statuses.append(
                {
                    "provider_id": provider_id,
                    "display_name": descriptor.display_name,
                    "kind": "local",
                    "status": "missing_url" if not base_url.strip() else "unknown",
                    "label": "Missing URL" if not base_url.strip() else "Not checked",
                    "base_url": base_url or descriptor.default_base_url or "",
                }
            )
            continue

        value = str(state.get(descriptor.credential_env, {}).get("value", ""))
        configured = bool(value.strip())
        statuses.append(
            {
                "provider_id": provider_id,
                "display_name": descriptor.display_name,
                "kind": "remote",
                "status": "configured" if configured else "missing_key",
                "label": "Configured" if configured else "Missing key",
                "credential_env": descriptor.credential_env,
            }
        )
    return statuses


def _value_for_settings_attr(
    state: Mapping[str, Mapping[str, Any]], settings_attr: str
) -> str:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            return str(state.get(field.key, {}).get("value", field.default))
    return ""

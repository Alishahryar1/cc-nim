"""Ensure admin UI manifest exposes every catalog credential/proxy binding."""

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings


def test_provider_catalog_remote_credentials_in_admin_manifest() -> None:
    missing: list[str] = []
    wrong_attr: list[str] = []

    for provider_id, desc in PROVIDER_CATALOG.items():
        if desc.credential_env is None:
            continue
        if desc.credential_attr is None:
            missing.append(
                f"{provider_id}: credential_env set but credential_attr missing"
            )
            continue
        entry = FIELD_BY_KEY.get(desc.credential_env)
        if entry is None:
            missing.append(
                f"{provider_id}: {desc.credential_env} not in admin FIELD_BY_KEY"
            )
            continue
        if entry.settings_attr != desc.credential_attr:
            wrong_attr.append(
                f"{provider_id}: {desc.credential_env} maps settings_attr="
                f"{entry.settings_attr!r}, catalog expects "
                f"{desc.credential_attr!r}"
            )

    assert not missing and not wrong_attr, "\n".join(missing + wrong_attr)


def test_provider_catalog_base_urls_in_admin_manifest() -> None:
    missing_key: list[str] = []
    wrong_attr: list[str] = []

    for provider_id, desc in PROVIDER_CATALOG.items():
        if desc.base_url_attr is None:
            continue
        mf = Settings.model_fields[desc.base_url_attr]
        alias = mf.validation_alias
        if alias is None:
            missing_key.append(
                f"{provider_id}: {desc.base_url_attr} has no validation_alias "
                "(admin manifest expects env-backed base URL)"
            )
            continue
        env_key = str(alias)
        entry = FIELD_BY_KEY.get(env_key)
        if entry is None:
            missing_key.append(
                f"{provider_id}: base URL env {env_key} not in FIELD_BY_KEY"
            )
            continue
        if entry.settings_attr != desc.base_url_attr:
            wrong_attr.append(
                f"{provider_id}: {env_key} maps settings_attr="
                f"{entry.settings_attr!r}, catalog expects {desc.base_url_attr!r}"
            )

    assert not missing_key and not wrong_attr, "\n".join(missing_key + wrong_attr)


def test_provider_catalog_proxy_attrs_in_admin_manifest() -> None:
    missing_key: list[str] = []
    wrong_attr: list[str] = []

    for provider_id, desc in PROVIDER_CATALOG.items():
        if desc.proxy_attr is None:
            continue
        mf = Settings.model_fields[desc.proxy_attr]
        alias = mf.validation_alias
        if alias is None:
            missing_key.append(
                f"{provider_id}: {desc.proxy_attr} has no validation_alias "
                "(admin manifest expects env-backed proxy)"
            )
            continue
        env_key = str(alias)
        entry = FIELD_BY_KEY.get(env_key)
        if entry is None:
            missing_key.append(
                f"{provider_id}: proxy env {env_key} not in FIELD_BY_KEY"
            )
            continue
        if entry.settings_attr != desc.proxy_attr:
            wrong_attr.append(
                f"{provider_id}: {env_key} maps settings_attr="
                f"{entry.settings_attr!r}, catalog expects {desc.proxy_attr!r}"
            )

    assert not missing_key and not wrong_attr, "\n".join(missing_key + wrong_attr)


def test_provider_catalog_display_names_are_admin_status_source() -> None:
    from free_claude_code.config.admin.status import provider_config_status
    from free_claude_code.config.admin.values import load_value_state

    status_by_provider = {
        entry["provider_id"]: entry
        for entry in provider_config_status(load_value_state())
    }

    assert set(status_by_provider) == set(PROVIDER_CATALOG)
    for provider_id, desc in PROVIDER_CATALOG.items():
        assert status_by_provider[provider_id]["display_name"] == desc.display_name
        expected_kind = "local" if desc.local else "remote"
        assert status_by_provider[provider_id]["kind"] == expected_kind


def test_cloudflare_account_id_is_admin_provider_field() -> None:
    entry = FIELD_BY_KEY["CLOUDFLARE_ACCOUNT_ID"]

    assert entry.settings_attr == "cloudflare_account_id"
    assert entry.section_id == "providers"
    assert entry.secret is False


def test_vertex_project_and_location_are_admin_provider_fields() -> None:
    project = FIELD_BY_KEY["VERTEX_PROJECT_ID"]
    location = FIELD_BY_KEY["VERTEX_LOCATION"]

    assert project.settings_attr == "vertex_project_id"
    assert project.section_id == "providers"
    assert project.secret is False
    assert location.settings_attr == "vertex_location"
    assert location.default == "global"


def test_vertex_admin_status_uses_project_configuration_not_an_api_key() -> None:
    from free_claude_code.config.admin.status import provider_config_status

    def vertex_status(project_id: str) -> dict[str, object]:
        statuses = provider_config_status(
            {
                "VERTEX_PROJECT_ID": {"value": project_id},
                "VERTEX_LOCATION": {"value": "global"},
            }
        )
        return next(status for status in statuses if status["provider_id"] == "vertex")

    assert vertex_status("")["status"] == "missing_config"
    assert vertex_status("")["label"] == "Missing configuration"
    assert vertex_status("")["configuration"] == "VERTEX_PROJECT_ID"
    assert vertex_status("vertex-project")["status"] == "configured"

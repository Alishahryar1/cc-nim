from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.config.admin.values import MASKED_SECRET
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.server_urls import local_admin_url
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, provider_manager_for_app, runtime_for_app


def _local_client(app):
    return TestClient(app, client=("127.0.0.1", 50000))


def _set_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _clear_process_config(monkeypatch) -> None:
    for key in (
        "MODEL",
        "NVIDIA_NIM_API_KEY",
        "HUGGINGFACE_API_KEY",
        "OPENROUTER_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "BEDROCK_BASE_URL",
        "BEDROCK_PROXY",
        "OLLAMA_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "PROXY_AUTH_ENABLED",
        "TELEGRAM_PROXY_URL",
        "FCC_ENV_FILE",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "GITHUB_MODELS_TOKEN",
        "SAMBANOVA_API_KEY",
        "HOST",
        "PORT",
        "FCC_OPEN_BROWSER",
        "VOICE_NOTE_ENABLED",
        "WHISPER_DEVICE",
        "LOG_FILE",
        "ZAI_BASE_URL",
        "CLAUDE_WORKSPACE",
        "CLAUDE_CLI_BIN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_admin_page_is_loopback_only(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()

    assert _local_client(app).get("/admin").status_code == 200
    remote_client = TestClient(app, client=("203.0.113.10", 50000))
    assert remote_client.get("/admin").status_code == 403


@pytest.mark.parametrize(
    "path",
    (
        "/admin",
        "/admin/assets/admin.css",
        "/admin/assets/admin.js",
        "/admin/api/config",
    ),
)
def test_admin_responses_are_never_cached(monkeypatch, tmp_path, path):
    _set_home(monkeypatch, tmp_path)
    response = _local_client(create_test_app()).get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("path", "client_host", "expected_status"),
    (
        ("/admin", "203.0.113.10", 403),
        ("/admin/assets/missing.js", "127.0.0.1", 404),
    ),
)
def test_admin_http_errors_are_never_cached(
    monkeypatch,
    tmp_path,
    path,
    client_host,
    expected_status,
):
    _set_home(monkeypatch, tmp_path)
    client = TestClient(create_test_app(), client=(client_host, 50000))

    response = client.get(path)

    assert response.status_code == expected_status
    assert response.headers["cache-control"] == "no-store"


def test_admin_validation_errors_are_never_cached(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    response = _local_client(create_test_app()).post(
        "/admin/api/config/validate",
        content="{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"


def test_admin_unexpected_errors_are_never_cached(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    client = TestClient(
        create_test_app(),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    with patch(
        "free_claude_code.api.admin_routes.load_config_response",
        side_effect=RuntimeError("test error"),
    ):
        response = client.get("/admin/api/config")

    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"


def test_admin_cache_policy_does_not_match_similar_public_paths(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    response = _local_client(create_test_app()).get("/administrator")

    assert response.status_code == 404
    assert "cache-control" not in response.headers


def test_admin_api_fetches_bypass_browser_cache():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'cache: "no-store"' in script


def test_admin_connected_account_login_preopens_sign_in_window():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'window.open("about:blank", "_blank")' in script
    assert "popup.location.replace(target)" in script
    assert "if (popup) popup.close()" in script
    assert '"Reconnect"' in script
    assert '"Copy code"' in script
    assert "Restart your agent to refresh its model picker." in script
    assert 'window.confirm("Disconnect this ChatGPT account from FCC?")' in script


class _FakeConnectedAccount:
    def __init__(self) -> None:
        self.connected = False
        self.revision = 0
        self.cancelled = False

    def is_connected(self) -> bool:
        return self.connected

    def status(self) -> ConnectedAccountStatus:
        return ConnectedAccountStatus(
            provider_id="openai",
            state=(
                ConnectedAccountState.CONNECTED
                if self.connected
                else ConnectedAccountState.DISCONNECTED
            ),
            connected=self.connected,
            revision=self.revision,
            email="safe@example.com" if self.connected else None,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        return ConnectedAccountStatus(
            provider_id="openai",
            state=ConnectedAccountState.CONNECTING,
            connected=False,
            revision=self.revision,
            attempt_id="login_safe",
            mode=mode,
            authorization_url="https://auth.openai.com/safe",
        )

    async def cancel_login(self) -> ConnectedAccountStatus:
        self.cancelled = True
        return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        self.connected = False
        self.revision += 1
        return self.status()

    async def close(self) -> None:
        return None


def test_admin_connected_account_routes_are_safe_loopback_only_and_uncached(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    account = _FakeConnectedAccount()
    app = create_test_app(connected_accounts={"openai": account})
    client = _local_client(app)

    status_response = client.get("/admin/api/providers/openai/auth")
    login_response = client.post(
        "/admin/api/providers/openai/auth/login",
        json={"mode": "browser"},
    )
    cancel_response = client.post("/admin/api/providers/openai/auth/cancel")

    assert status_response.status_code == 200
    assert status_response.headers["cache-control"] == "no-store"
    assert status_response.json()["state"] == "disconnected"
    assert login_response.status_code == 200
    assert login_response.json() == {
        "provider_id": "openai",
        "state": "connecting",
        "connected": False,
        "revision": 0,
        "attempt_id": "login_safe",
        "mode": "browser",
        "authorization_url": "https://auth.openai.com/safe",
    }
    assert "token" not in login_response.text.lower()
    assert cancel_response.status_code == 200
    assert account.cancelled is True
    remote = TestClient(app, client=("203.0.113.10", 50000))
    assert remote.get("/admin/api/providers/openai/auth").status_code == 403


def test_admin_rejects_auth_routes_for_non_connected_provider(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)

    response = _local_client(create_test_app()).get(
        "/admin/api/providers/nvidia_nim/auth"
    )

    assert response.status_code == 404


def test_admin_page_no_longer_renders_generated_env_panel(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()

    response = _local_client(app).get("/admin")

    assert response.status_code == 200
    assert "Generated Env" not in response.text
    assert "envPreview" not in response.text


def test_admin_page_no_longer_renders_global_status_header(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()

    response = _local_client(app).get("/admin")

    assert response.status_code == 200
    assert "Local Admin" not in response.text
    assert "serverStatus" not in response.text
    assert "modelBadge" not in response.text


def test_admin_static_no_longer_fetches_global_status_header():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'api("/admin/api/status")' not in script
    assert "updateHeader" not in script
    assert '"Running"' not in script
    assert "serverStatus" not in script
    assert "modelBadge" not in script


def test_admin_static_hides_managed_source_label():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'managed_env: "",' in script
    assert "hasOwnProperty.call(labels, source)" in script
    assert 'parts.push("locked")' in script
    assert "sourceEl.textContent = source" in script


def test_admin_static_places_reasoning_fields_in_model_config():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert 'sections: ["models", "reasoning", "web_tools"]' in script
    assert 'sections: ["models", "thinking", "web_tools"]' not in script


def test_admin_static_model_combobox_owns_dropdown_and_search_behavior():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )
    styles = Path("src/free_claude_code/api/admin_static/admin.css").read_text(
        encoding="utf-8"
    )

    assert 'api("/admin/api/models" + (refresh ? "/refresh" : "")' in script
    assert 'field.type === "model" || field.type === "optional_model"' in script
    assert 'input.setAttribute("role", "combobox")' in script
    assert 'listbox.setAttribute("role", "listbox")' in script
    assert 'toggle.className = "model-combobox-toggle"' in script
    assert "class ModelCombobox" in script
    assert 'input.addEventListener("click", () => this.open())' in script
    assert "value.toLocaleLowerCase().includes(normalizedQuery)" in script
    assert 'event.key === "ArrowDown" || event.key === "ArrowUp"' in script
    assert "this.setActive(this.visibleOptions.length - 1)" in script
    assert 'event.key === "Enter"' in script
    assert 'event.key === "Escape"' in script
    assert 'document.createElement("datalist")' not in script
    assert ".model-combobox-list" in styles
    assert ".model-combobox-option.active" in styles
    assert styles.count("background-image: var(--dropdown-chevron)") == 2


def test_admin_static_model_combobox_preserves_custom_slugs_and_none_semantics():
    script = Path("src/free_claude_code/api/admin_static/admin.js").read_text(
        encoding="utf-8"
    )

    assert '? ["None", ...state.modelOptions]' in script
    assert "You can still enter a custom slug." in script
    assert 'input.dataset.fieldType === "optional_model"' in script
    assert "return null;" in script
    assert "await hydrateModelOptions();" in script
    assert "Model fields remain editable" in script
    assert "result.failed_providers || []" in script
    assert '"warn"' in script


def test_admin_config_masks_secrets_and_exposes_manifest(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    keys = {field["key"] for field in body["fields"]}
    assert "MODEL_FABLE" in keys
    assert "REASONING_FABLE" in keys
    assert "ANTHROPIC_AUTH_TOKEN" in keys
    assert "PROXY_AUTH_ENABLED" in keys
    assert "LOG_LEVEL" in keys
    assert "OPENROUTER_API_KEY" in keys
    assert "AWS_BEARER_TOKEN_BEDROCK" in keys
    assert "BEDROCK_BASE_URL" in keys
    assert "FIREWORKS_API_KEY" in keys
    assert "CLOUDFLARE_API_TOKEN" in keys
    assert "CLOUDFLARE_ACCOUNT_ID" in keys
    assert "GITHUB_MODELS_TOKEN" in keys
    assert "GEMINI_API_KEY" in keys
    assert "GROQ_API_KEY" in keys
    assert "SAMBANOVA_API_KEY" in keys
    assert "TELEGRAM_PROXY_URL" in keys
    assert "CEREBRAS_API_KEY" in keys
    assert "OLLAMA_API_KEY" in keys
    assert "FCC_OPEN_BROWSER" in keys
    assert "ZAI_BASE_URL" not in keys
    assert "CLAUDE_WORKSPACE" not in keys
    assert "CLAUDE_CLI_BIN" not in keys
    assert "LOG_FILE" not in keys
    auth_field = next(
        field for field in body["fields"] if field["key"] == "ANTHROPIC_AUTH_TOKEN"
    )
    assert auth_field["secret"] is True
    assert auth_field["value"] == MASKED_SECRET
    assert auth_field["source"] == "default"
    assert auth_field["nullable"] is False
    telegram_proxy_field = next(
        field for field in body["fields"] if field["key"] == "TELEGRAM_PROXY_URL"
    )
    assert telegram_proxy_field["secret"] is True
    assert telegram_proxy_field["value"] is None
    assert telegram_proxy_field["configured"] is False
    assert telegram_proxy_field["nullable"] is True
    assert body["paths"] == {"managed": str(tmp_path / ".fcc" / ".env")}
    catalog_smoke_keys = {
        f"FCC_SMOKE_MODEL_{provider_id.upper()}" for provider_id in PROVIDER_CATALOG
    }
    actual_smoke_keys = {
        field["key"]
        for field in body["fields"]
        if field["key"].startswith("FCC_SMOKE_MODEL_")
    }
    assert actual_smoke_keys == catalog_smoke_keys | {
        "FCC_SMOKE_MODEL_MISTRAL_REASONING"
    }
    open_browser_field = next(
        field for field in body["fields"] if field["key"] == "FCC_OPEN_BROWSER"
    )
    assert open_browser_field["type"] == "boolean"
    assert open_browser_field["value"] == "true"
    assert open_browser_field["restart_required"] is False
    progress_timeout_field = next(
        field for field in body["fields"] if field["key"] == "PROVIDER_PROGRESS_TIMEOUT"
    )
    assert progress_timeout_field["label"] == "Provider Progress Timeout"
    assert progress_timeout_field["section"] == "runtime"
    assert progress_timeout_field["type"] == "number"
    assert progress_timeout_field["value"] == "600.0"
    assert progress_timeout_field["advanced"] is True
    assert progress_timeout_field["restart_required"] is False
    assert "non-empty protocol event" in progress_timeout_field["description"]
    assert "Independent of HTTP Read Timeout" in progress_timeout_field["description"]
    model_field_types = {
        field["key"]: field["type"]
        for field in body["fields"]
        if field["key"]
        in {"MODEL", "MODEL_FABLE", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"}
    }
    assert model_field_types == {
        "MODEL": "model",
        "MODEL_FABLE": "optional_model",
        "MODEL_OPUS": "optional_model",
        "MODEL_SONNET": "optional_model",
        "MODEL_HAIKU": "optional_model",
    }
    reasoning_policy = next(
        field for field in body["fields"] if field["key"] == "REASONING_POLICY"
    )
    assert reasoning_policy["section"] == "reasoning"
    assert reasoning_policy["type"] == "select"
    assert reasoning_policy["value"] == "client"
    assert reasoning_policy["options"] == [
        {"value": "off", "label": "Off"},
        {"value": "client", "label": "From client"},
        {"value": "low", "label": "Low"},
        {"value": "medium", "label": "Medium"},
        {"value": "high", "label": "High"},
        {"value": "xhigh", "label": "X-High"},
        {"value": "max", "label": "Max"},
    ]
    route_reasoning = next(
        field for field in body["fields"] if field["key"] == "REASONING_FABLE"
    )
    assert route_reasoning["options"] == [
        {"value": "inherit", "label": "Inherit"},
        *reasoning_policy["options"],
    ]
    restart_required = {
        field["key"] for field in body["fields"] if field["restart_required"] is True
    }
    assert {
        "ANTHROPIC_AUTH_TOKEN",
        "DEBUG_PLATFORM_EDITS",
        "DEBUG_SUBAGENT_STACK",
        "LOG_RAW_API_PAYLOADS",
        "LOG_API_ERROR_TRACEBACKS",
        "LOG_RAW_MESSAGING_CONTENT",
        "LOG_RAW_CLI_DIAGNOSTICS",
        "LOG_MESSAGING_ERROR_DETAILS",
    } <= restart_required


def test_admin_models_include_configured_and_cached_canonical_slugs():
    settings = Settings()
    settings.model = "nvidia_nim/configured-model"
    settings.model_opus = "open_router/anthropic/configured-opus"
    settings.open_router_api_key = "open-router-key"
    app = create_test_app(settings)
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("anthropic/configured-opus"),
            ProviderModelInfo("meta/llama-3.3"),
        },
    )

    response = _local_client(app).get("/admin/api/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            "nvidia_nim/configured-model",
            "open_router/anthropic/configured-opus",
            "open_router/meta/llama-3.3",
        ],
        "failed_providers": [],
    }


def test_admin_model_refresh_returns_the_updated_canonical_catalog():
    settings = Settings()
    settings.model = "deepseek/deepseek-chat"
    settings.deepseek_api_key = "deepseek-key"
    app = create_test_app(settings)
    runtime = app.state.services.admin

    async def refresh_models() -> ProviderModelRefreshResult:
        provider_manager_for_app(app).cache_model_infos(
            "deepseek",
            {ProviderModelInfo("deepseek-reasoner")},
        )
        return ProviderModelRefreshResult(refreshed_provider_ids=("deepseek",))

    runtime.refresh_models = AsyncMock(side_effect=refresh_models)

    response = _local_client(app).post("/admin/api/models/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"],
        "failed_providers": [],
    }
    runtime.refresh_models.assert_awaited_once_with()


def test_admin_model_refresh_reports_partial_provider_failures():
    settings = Settings()
    settings.model = "deepseek/deepseek-chat"
    app = create_test_app(settings)
    runtime = app.state.services.admin
    runtime.refresh_models = AsyncMock(
        return_value=ProviderModelRefreshResult(
            refreshed_provider_ids=("deepseek",),
            failed_provider_ids=("open_router",),
        )
    )

    response = _local_client(app).post("/admin/api/models/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["deepseek/deepseek-chat"],
        "failed_providers": ["open_router"],
    }


def test_admin_config_preserves_managed_env_source_contract(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("MODEL=open_router/managed-model\n", encoding="utf-8")
    app = create_test_app()

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    model_field = next(field for field in body["fields"] if field["key"] == "MODEL")
    assert model_field["source"] == "managed_env"
    assert model_field["locked"] is False


def test_admin_apply_persists_open_browser_for_next_launch(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"FCC_OPEN_BROWSER": False}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == []
    assert body["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }
    managed_env = tmp_path / ".fcc" / ".env"
    assert "FCC_OPEN_BROWSER=false" in managed_env.read_text(encoding="utf-8")


def test_admin_apply_hot_publishes_provider_progress_timeout(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PROVIDER_PROGRESS_TIMEOUT": 900}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == []
    assert body["restart"]["required"] is False
    assert (
        provider_manager_for_app(app).current_settings().provider_progress_timeout
        == 900.0
    )
    managed_env = tmp_path / ".fcc" / ".env"
    assert "PROVIDER_PROGRESS_TIMEOUT=900" in managed_env.read_text(encoding="utf-8")


def test_admin_rejects_invalid_provider_progress_timeout_without_writing(
    monkeypatch,
    tmp_path,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PROVIDER_PROGRESS_TIMEOUT": 0}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert any("PROVIDER_PROGRESS_TIMEOUT" in error for error in body["errors"])
    managed_env = tmp_path / ".fcc" / ".env"
    assert "PROVIDER_PROGRESS_TIMEOUT=" not in managed_env.read_text(encoding="utf-8")


def test_admin_apply_masks_telegram_proxy_credentials(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()
    proxy_url = "https://user:password@proxy.example:8443"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"TELEGRAM_PROXY_URL": proxy_url}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "TELEGRAM_PROXY_URL=********" in body["env_preview"]
    assert proxy_url not in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert f"TELEGRAM_PROXY_URL={proxy_url}" in text


def test_admin_apply_null_removes_optional_secret(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\nTELEGRAM_PROXY_URL=https://secret.invalid\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"TELEGRAM_PROXY_URL": None}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert "TELEGRAM_PROXY_URL=" not in env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("submitted", [MASKED_SECRET, "", "   "])
def test_admin_apply_masked_or_blank_secret_is_unchanged(
    monkeypatch,
    tmp_path,
    submitted,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "FCC_CONFIG_SCHEMA=1\nOPENROUTER_API_KEY=original-secret\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"OPENROUTER_API_KEY": submitted}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert "OPENROUTER_API_KEY=original-secret" in env_file.read_text(encoding="utf-8")


def test_admin_validate_rejects_bad_model_shape(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/validate",
        json={"values": {"MODEL": "missing-provider-prefix"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert any("provider type" in error for error in body["errors"])


def test_admin_apply_writes_complete_managed_env_and_masks_preview(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "open_router/test-model",
                "OPENROUTER_API_KEY": "router-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "OPENROUTER_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "OPENROUTER_API_KEY=router-secret" in text
    assert "ANTHROPIC_AUTH_TOKEN=" not in text
    assert "PROXY_AUTH_ENABLED=" not in text
    assert "HOST=" not in text
    assert "PORT=" not in text
    assert body["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }


def test_admin_apply_writes_fireworks_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "fireworks/test-model",
                "FIREWORKS_API_KEY": "fw-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "FIREWORKS_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=fireworks/test-model" in text
    assert "FIREWORKS_API_KEY=fw-secret" in text


def test_admin_apply_writes_gemini_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "gemini/models/gemini-3.1-flash-lite",
                "GEMINI_API_KEY": "gm-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "GEMINI_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=gemini/models/gemini-3.1-flash-lite" in text
    assert "GEMINI_API_KEY=gm-secret" in text


def test_admin_apply_writes_groq_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "groq/llama-3.3-70b-versatile",
                "GROQ_API_KEY": "gq-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "GROQ_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=groq/llama-3.3-70b-versatile" in text
    assert "GROQ_API_KEY=gq-secret" in text


def test_admin_apply_writes_sambanova_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "sambanova/Meta-Llama-3.3-70B-Instruct",
                "SAMBANOVA_API_KEY": "sn-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "SAMBANOVA_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=sambanova/Meta-Llama-3.3-70B-Instruct" in text
    assert "SAMBANOVA_API_KEY=sn-secret" in text


def test_admin_apply_writes_cerebras_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "cerebras/llama3.1-8b",
                "CEREBRAS_API_KEY": "cb-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "CEREBRAS_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=cerebras/llama3.1-8b" in text
    assert "CEREBRAS_API_KEY=cb-secret" in text


def test_admin_apply_writes_bedrock_region_config_and_masks_key(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "bedrock/openai.gpt-oss-120b",
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-secret",
                "BEDROCK_BASE_URL": ("https://bedrock-mantle.us-west-2.api.aws/v1"),
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "AWS_BEARER_TOKEN_BEDROCK=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=bedrock/openai.gpt-oss-120b" in text
    assert "AWS_BEARER_TOKEN_BEDROCK=bedrock-secret" in text
    assert "BEDROCK_BASE_URL=https://bedrock-mantle.us-west-2.api.aws/v1" in text


def test_admin_apply_writes_cloudflare_fields_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "cloudflare/@cf/moonshotai/kimi-k2.6",
                "CLOUDFLARE_API_TOKEN": "cf-secret",
                "CLOUDFLARE_ACCOUNT_ID": "cf-account",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "CLOUDFLARE_API_TOKEN=********" in body["env_preview"]
    assert "CLOUDFLARE_ACCOUNT_ID=cf-account" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=cloudflare/@cf/moonshotai/kimi-k2.6" in text
    assert "CLOUDFLARE_API_TOKEN=cf-secret" in text
    assert "CLOUDFLARE_ACCOUNT_ID=cf-account" in text


def test_admin_apply_writes_huggingface_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "huggingface/openai/gpt-oss-120b:fastest",
                "HUGGINGFACE_API_KEY": "hf-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == ["HUGGINGFACE_API_KEY"]
    assert "HUGGINGFACE_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=huggingface/openai/gpt-oss-120b:fastest" in text
    assert "HUGGINGFACE_API_KEY=hf-secret" in text


@pytest.mark.parametrize(
    ("device", "credential_key"),
    [
        ("nvidia_nim", "NVIDIA_NIM_API_KEY"),
        ("cpu", "HUGGINGFACE_API_KEY"),
    ],
)
def test_admin_key_change_requires_restart_for_active_voice_backend(
    monkeypatch,
    tmp_path,
    device,
    credential_key,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "VOICE_NOTE_ENABLED=true",
                f"WHISPER_DEVICE={device}",
                f"{credential_key}=old-key",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {credential_key: "new-key"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == [credential_key]
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": [credential_key],
    }


@pytest.mark.parametrize(
    ("key", "initial", "updated"),
    [
        ("ANTHROPIC_AUTH_TOKEN", "old-token", "new-token"),
        ("DEBUG_PLATFORM_EDITS", "true", "false"),
        ("DEBUG_SUBAGENT_STACK", "true", "false"),
        ("LOG_RAW_API_PAYLOADS", "true", "false"),
        ("LOG_API_ERROR_TRACEBACKS", "true", "false"),
        ("LOG_RAW_MESSAGING_CONTENT", "true", "false"),
        ("LOG_RAW_CLI_DIAGNOSTICS", "true", "false"),
        ("LOG_MESSAGING_ERROR_DETAILS", "true", "false"),
    ],
)
def test_admin_constructor_captured_setting_requires_restart(
    monkeypatch,
    tmp_path,
    key,
    initial,
    updated,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f"{key}={initial}\n", encoding="utf-8")
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {key: updated}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == [key]
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": [key],
    }


def test_admin_apply_writes_cohere_key_and_masks_preview(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "cohere/command-a-plus-05-2026",
                "COHERE_API_KEY": "cohere-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "COHERE_API_KEY=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=cohere/command-a-plus-05-2026" in text
    assert "COHERE_API_KEY=cohere-secret" in text


def test_admin_apply_writes_github_models_token_and_masks_preview(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "MODEL": "github_models/openai/gpt-4.1",
                "GITHUB_MODELS_TOKEN": "github-secret",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert "GITHUB_MODELS_TOKEN=********" in body["env_preview"]
    env_file = tmp_path / ".fcc" / ".env"
    text = env_file.read_text(encoding="utf-8")
    assert "MODEL=github_models/openai/gpt-4.1" in text
    assert "GITHUB_MODELS_TOKEN=github-secret" in text


def test_admin_apply_preserves_hidden_diagnostics_and_smoke_values(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=nvidia_nim/old-model",
                "LOG_RAW_API_PAYLOADS=true",
                "FCC_SMOKE_MODEL_ZAI=zai/smoke-model",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "open_router/test-model"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "LOG_RAW_API_PAYLOADS=true" in text
    assert "FCC_SMOKE_MODEL_ZAI=zai/smoke-model" in text


def test_admin_apply_preserves_unrecognized_managed_zai_base_url(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=zai/glm-5.2",
                "ZAI_API_KEY=zai-secret",
                "ZAI_BASE_URL=https://custom.zai.invalid/v1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "zai/glm-5.2"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "ZAI_API_KEY=zai-secret" in text
    assert "ZAI_BASE_URL=https://custom.zai.invalid/v1" in text


def test_admin_apply_preserves_unrecognized_managed_assignments(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "MODEL=open_router/test-model",
                "CLAUDE_WORKSPACE=C:/custom/workspace",
                "CLAUDE_CLI_BIN=claude-custom",
                "",
            ]
        ),
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "open_router/test-model"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    text = env_file.read_text("utf-8")
    assert "MODEL=open_router/test-model" in text
    assert "CLAUDE_WORKSPACE=C:/custom/workspace" in text
    assert "CLAUDE_CLI_BIN=claude-custom" in text


def test_admin_apply_restart_required_reports_automatic_restart(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    callbacks: list[str] = []

    async def restart_callback() -> None:
        callbacks.append("restart")

    app = create_test_app(restart_callback=restart_callback)

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PORT": "9090"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == ["PORT"]
    assert body["restart"] == {
        "required": True,
        "automatic": True,
        "admin_url": "http://127.0.0.1:9090/admin",
        "fields": ["PORT"],
    }
    assert callbacks == ["restart"]


def test_admin_apply_restart_required_reports_manual_fallback(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PORT": "9091"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["pending_fields"] == ["PORT"]
    assert body["restart"] == {
        "required": True,
        "automatic": False,
        "admin_url": None,
        "fields": ["PORT"],
    }


def test_admin_process_env_values_are_locked_and_not_written(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.setenv("MODEL", "open_router/process-model")
    app = create_test_app()

    config = _local_client(app).get("/admin/api/config").json()
    model_field = next(field for field in config["fields"] if field["key"] == "MODEL")
    assert model_field["locked"] is True
    assert model_field["source"] == "process"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MODEL": "deepseek/managed-model"}},
    )

    assert response.status_code == 200
    env_file = tmp_path / ".fcc" / ".env"
    assert "deepseek/managed-model" not in env_file.read_text("utf-8")


def test_admin_never_reads_arbitrary_current_directory_env(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MODEL=deepseek/deepseek-chat\nDEEPSEEK_API_KEY=deepseek-secret\n",
        encoding="utf-8",
    )
    app = create_test_app()

    config = _local_client(app).get("/admin/api/config").json()
    model_field = next(field for field in config["fields"] if field["key"] == "MODEL")
    assert model_field["value"] == "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    assert model_field["source"] == "default"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {}},
    )

    assert response.status_code == 200
    managed_text = (tmp_path / ".fcc" / ".env").read_text("utf-8")
    assert "MODEL=deepseek/deepseek-chat" not in managed_text
    assert "DEEPSEEK_API_KEY=deepseek-secret" not in managed_text


def test_admin_migration_removes_blank_required_managed_values(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "MESSAGING_PLATFORM=\nWHISPER_DEVICE=\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"NVIDIA_NIM_API_KEY": "nim-secret"}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    managed_text = (tmp_path / ".fcc" / ".env").read_text("utf-8")
    assert "NVIDIA_NIM_API_KEY=nim-secret" in managed_text
    assert "MESSAGING_PLATFORM=" not in managed_text
    assert "WHISPER_DEVICE=" not in managed_text


def test_admin_migrates_empty_auth_token_to_explicit_disabled_state(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "ANTHROPIC_AUTH_TOKEN=\nPROVIDER_RATE_LIMIT=1\n",
        encoding="utf-8",
    )
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"PROVIDER_RATE_LIMIT": "2"}},
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    managed_lines = env_file.read_text("utf-8").splitlines()
    assert not any(line.startswith("ANTHROPIC_AUTH_TOKEN=") for line in managed_lines)
    assert "PROXY_AUTH_ENABLED=false" in managed_lines
    assert "PROVIDER_RATE_LIMIT=2" in managed_lines


@pytest.mark.parametrize(
    ("submitted", "message"),
    [
        (None, "this setting cannot be removed"),
        ("", "this setting cannot be blank"),
        ("   ", "this setting cannot be blank"),
    ],
)
def test_admin_apply_rejects_missing_required_value_without_writing(
    monkeypatch,
    tmp_path,
    submitted,
    message,
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    env_file = tmp_path / ".fcc" / ".env"
    env_file.parent.mkdir(parents=True)
    original = "ANTHROPIC_AUTH_TOKEN=\nMESSAGING_PLATFORM=none\n"
    env_file.write_text(original, encoding="utf-8")
    app = create_test_app()
    _local_client(app).get("/admin/api/config")
    baseline = env_file.read_bytes()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {"MESSAGING_PLATFORM": submitted}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is False
    assert body["valid"] is False
    assert f"MESSAGING_PLATFORM: {message}" in body["errors"]
    assert env_file.read_bytes() == baseline


def test_admin_apply_preserves_false_and_numeric_zero(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={
            "values": {
                "FCC_OPEN_BROWSER": False,
                "HTTP_WRITE_TIMEOUT": 0,
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["applied"] is True
    managed = (tmp_path / ".fcc" / ".env").read_text(encoding="utf-8")
    assert "FCC_OPEN_BROWSER=false" in managed
    assert "HTTP_WRITE_TIMEOUT=0" in managed


def test_admin_local_provider_status_reports_reachable(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            return httpx.Response(200, json={"data": []})

    with patch("free_claude_code.api.admin_routes.httpx.AsyncClient", FakeAsyncClient):
        response = _local_client(app).get("/admin/api/providers/local-status")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert {provider["status"] for provider in providers} == {"reachable"}


def test_admin_config_exposes_structured_provider_configuration_targets(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)

    response = _local_client(create_test_app()).get("/admin/api/config")

    assert response.status_code == 200
    providers = {
        provider["provider_id"]: provider
        for provider in response.json()["provider_status"]
    }
    assert providers["nvidia_nim"]["configuration_keys"] == ["NVIDIA_NIM_API_KEY"]
    assert providers["nvidia_nim"]["missing_configuration_keys"] == [
        "NVIDIA_NIM_API_KEY"
    ]
    assert "configuration" not in providers["nvidia_nim"]
    assert providers["lmstudio"]["status"] == "configured"
    assert providers["lmstudio"]["configuration_keys"] == ["LM_STUDIO_BASE_URL"]


def test_admin_local_provider_failure_does_not_return_exception_text(
    monkeypatch, tmp_path
):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    class FailingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            raise RuntimeError(
                "Provider rejected credential CREDENTIAL[unrecognized-format-987654321]"
            )

    with patch(
        "free_claude_code.api.admin_routes.httpx.AsyncClient",
        return_value=FailingAsyncClient(),
    ):
        response = _local_client(app).get("/admin/api/providers/local-status")

    assert response.status_code == 200
    providers = response.json()["providers"]
    assert {provider["status"] for provider in providers} == {"offline"}
    for provider in providers:
        assert provider["message"] == (
            "Could not connect. Verify the URL and that the local provider is running."
        )
        assert "CREDENTIAL[unrecognized-format-987654321]" not in provider["message"]
        assert "RuntimeError" not in provider["message"]
        assert "error_type" not in provider


@pytest.mark.parametrize(
    "result",
    (
        {"provider_id": "nvidia_nim", "ok": True, "models": ["model-a"]},
        {
            "provider_id": "nvidia_nim",
            "ok": False,
            "message": "NVIDIA_NIM_API_KEY is not set.",
        },
    ),
)
def test_admin_provider_test_route_preserves_runtime_result(
    monkeypatch, tmp_path, result
):
    _set_home(monkeypatch, tmp_path)
    app = create_test_app()
    runtime = runtime_for_app(app)

    with patch.object(runtime, "test_provider", new=AsyncMock(return_value=result)):
        response = _local_client(app).post(
            "/admin/api/providers/nvidia_nim/test",
            json={},
        )

    assert response.status_code == 200
    assert response.json() == result


def test_admin_launch_url_uses_loopback_for_wildcard_host():
    settings = Settings.model_construct(host="0.0.0.0", port=8082)

    assert local_admin_url(settings) == "http://127.0.0.1:8082/admin"

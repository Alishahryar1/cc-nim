from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.admin.values import MASKED_SECRET
from free_claude_code.config.server_urls import local_admin_url
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


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
        "OLLAMA_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
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


def test_admin_config_masks_secrets_and_exposes_manifest(monkeypatch, tmp_path):
    _set_home(monkeypatch, tmp_path)
    _clear_process_config(monkeypatch)
    app = create_test_app()

    response = _local_client(app).get("/admin/api/config")

    assert response.status_code == 200
    body = response.json()
    keys = {field["key"] for field in body["fields"]}
    assert "MODEL_FABLE" in keys
    assert "ENABLE_FABLE_THINKING" in keys
    assert "ANTHROPIC_AUTH_TOKEN" in keys
    assert "OPENROUTER_API_KEY" in keys
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
    assert auth_field["source"] == "template"
    telegram_proxy_field = next(
        field for field in body["fields"] if field["key"] == "TELEGRAM_PROXY_URL"
    )
    assert telegram_proxy_field["secret"] is True
    open_browser_field = next(
        field for field in body["fields"] if field["key"] == "FCC_OPEN_BROWSER"
    )
    assert open_browser_field["type"] == "boolean"
    assert open_browser_field["value"] == "true"
    assert open_browser_field["restart_required"] is False
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
    assert "ANTHROPIC_AUTH_TOKEN=" in text
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
    assert body["pending_fields"] == []
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


def test_admin_apply_omits_stale_zai_base_url(monkeypatch, tmp_path):
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
    assert "ZAI_BASE_URL" not in text


def test_admin_apply_omits_stale_fixed_claude_runtime_settings(monkeypatch, tmp_path):
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
    assert "CLAUDE_WORKSPACE" not in text
    assert "CLAUDE_CLI_BIN" not in text


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


def test_admin_first_apply_migrates_repo_env(monkeypatch, tmp_path):
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
    assert model_field["value"] == "deepseek/deepseek-chat"
    assert model_field["source"] == "repo_env"

    response = _local_client(app).post(
        "/admin/api/config/apply",
        json={"values": {}},
    )

    assert response.status_code == 200
    managed_text = (tmp_path / ".fcc" / ".env").read_text("utf-8")
    assert "MODEL=deepseek/deepseek-chat" in managed_text
    assert "DEEPSEEK_API_KEY=deepseek-secret" in managed_text


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


def test_admin_launch_url_uses_loopback_for_wildcard_host():
    settings = Settings.model_construct(host="0.0.0.0", port=8082)

    assert local_admin_url(settings) == "http://127.0.0.1:8082/admin"

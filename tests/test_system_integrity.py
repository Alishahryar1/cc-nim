import pathlib

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture
def client():
    return TestClient(
        app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8082"
    )


@pytest.fixture
def env_config():
    env_map = {}
    env_path = pathlib.Path("/home/trappy/.fcc/.env")
    if not env_path.exists():
        env_path = pathlib.Path.home() / ".config/free-claude-code/.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env_map[k.strip()] = v.strip().strip('"').strip("\x27")
    return env_map


def test_public_health_route_accessible(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_config_schema_validation(client, env_config):
    payload = {
        "model": env_config.get("DEFAULT_MODEL", "nvidia_nim/moonshotai/kimi-k2.5"),
        "model_opus": env_config.get(
            "MODEL_OPUS", "open_router/google/gemini-2.5-flash"
        ),
        "model_sonnet": env_config.get(
            "MODEL_SONNET", "nvidia_nim/moonshotai/kimi-k2.5"
        ),
        "model_haiku": env_config.get(
            "MODEL_HAIKU", "open_router/google/gemini-2.5-flash"
        ),
    }
    headers = {"Host": "127.0.0.1:8082"}
    response = client.post("/admin/api/config/validate", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json().get("valid") is True


@pytest.mark.parametrize(
    "provider",
    [
        "nvidia_nim",
        "open_router",
        "gemini",
        "deepseek",
        "opencode",
        "opencode_go",
        "wafer",
        "kimi",
        "zai",
    ],
)
def test_configured_provider_connectivity_schema(client, provider):
    headers = {"Host": "127.0.0.1:8082"}
    response = client.post(f"/admin/api/providers/{provider}/test", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert data.get("provider_id") == provider

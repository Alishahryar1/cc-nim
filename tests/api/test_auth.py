from unittest.mock import patch

from fastapi.testclient import TestClient

from free_claude_code.api.dependencies import get_settings
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app

app = create_test_app()


def test_proxy_auth_requires_canonical_bearer_header():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "s3cr3t"
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("free_claude_code.api.routes.get_token_count", return_value=1):
        r = client.post("/v1/messages/count_tokens", json=payload)
        assert r.status_code == 401
        assert r.json() == {"detail": "Missing proxy authentication token"}
        assert r.headers["request-id"].startswith("req_")
        assert "x-should-retry" not in r.headers

        for headers in (
            {"X-API-Key": "s3cr3t"},
            {"anthropic-auth-token": "s3cr3t"},
        ):
            r = client.post(
                "/v1/messages/count_tokens",
                json=payload,
                headers=headers,
            )
            assert r.status_code == 401
            assert r.json() == {"detail": "Missing proxy authentication token"}

        r = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 1

    app.dependency_overrides.clear()


def test_proxy_auth_ignores_conflicting_legacy_headers():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "b3artoken"
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("free_claude_code.api.routes.get_token_count", return_value=2):
        r = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={
                "Authorization": "Bearer b3artoken",
                "X-API-Key": "stale-anthropic-key",
                "anthropic-auth-token": "stale-proxy-token",
            },
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 2

        r = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={
                "Authorization": "Bearer wrong",
                "X-API-Key": "b3artoken",
            },
        )
        assert r.status_code == 401
        assert r.json() == {"detail": "Invalid proxy authentication token"}

    app.dependency_overrides.clear()


def test_anthropic_auth_token_normalizes_configured_whitespace():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "  spaced-token  \n"
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("free_claude_code.api.routes.get_token_count", return_value=3):
        r = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={"Authorization": "Bearer spaced-token"},
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 3

    app.dependency_overrides.clear()


def test_anthropic_auth_token_applies_to_models_endpoint():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "models-token"
    app.dependency_overrides[get_settings] = lambda: settings

    r = client.get("/v1/models")
    assert r.status_code == 401
    assert r.headers["x-request-id"] == r.headers["request-id"]
    assert "x-should-retry" not in r.headers

    r = client.get("/v1/models", headers={"Authorization": "Bearer models-token"})
    assert r.status_code == 200
    assert "data" in r.json()

    app.dependency_overrides.clear()


def test_root_get_requires_auth_but_root_probes_are_public():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "root-token"
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.get("/")
    assert response.status_code == 401

    head = client.head("/")
    assert head.status_code == 204
    assert head.headers["Allow"] == "GET, HEAD, OPTIONS"

    options = client.options("/")
    assert options.status_code == 204
    assert options.headers["Allow"] == "GET, HEAD, OPTIONS"

    app.dependency_overrides.clear()

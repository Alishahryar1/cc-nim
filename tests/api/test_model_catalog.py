"""Tests for gateway /v1/models allowlist filtering."""

from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, provider_manager_for_app


def _settings(
    *,
    model: str = "deepseek/deepseek-chat",
    model_fable: str | None = None,
    model_opus: str | None = "open_router/anthropic/claude-opus",
    model_haiku: str | None = "deepseek/deepseek-chat",
    model_allowlist: list[str] | None = None,
    provider_model_allowlists: dict[str, list[str]] | None = None,
    nvidia_nim_api_key: str = "",
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=None,
        model_haiku=model_haiku,
        anthropic_auth_token="",
        deepseek_api_key="deepseek-key",
        open_router_api_key="open-router-key",
        wafer_api_key="wafer-key",
        nvidia_nim_api_key=nvidia_nim_api_key,
        model_allowlist=model_allowlist or [],
        provider_model_allowlists=provider_model_allowlists or {},
    )


def _cache_models(app, provider_id: str, *model_ids: str) -> None:
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {ProviderModelInfo(model_id) for model_id in model_ids},
    )


def _get_model_ids(response) -> list[str]:
    return [item["id"] for item in response.json()["data"]]


def test_gateway_models_list_respects_per_provider_allowlists():
    """Models from a provider with a restriction that are NOT in its allowlist are absent."""
    settings = _settings(
        model="nvidia_nim/nvidia/nemotron-3-super",
        model_opus=None,
        provider_model_allowlists={
            "nvidia_nim": ["nvidia/nemotron-3-super"],
        },
    )
    app = create_test_app(settings)
    _cache_models(app, "nvidia_nim", "nvidia/nemotron-3-super", "nvidia/other-model")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # Allowed model should be present (both normal and no-thinking variants)
    assert "anthropic/nvidia_nim/nvidia/nemotron-3-super" in ids
    assert "claude-3-freecc-no-thinking/nvidia_nim/nvidia/nemotron-3-super" in ids

    # Disallowed model should be absent
    assert "anthropic/nvidia_nim/nvidia/other-model" not in ids
    assert "claude-3-freecc-no-thinking/nvidia_nim/nvidia/other-model" not in ids


def test_gateway_models_list_per_provider_allowlist_allows_specific_models():
    """Models IN the allowlist are present with both variants."""
    settings = _settings(
        model_opus=None,
        provider_model_allowlists={
            "open_router": ["anthropic/claude-opus", "meta/llama-3.3"],
        },
    )
    app = create_test_app(settings)
    _cache_models(
        app,
        "open_router",
        "anthropic/claude-opus",
        "meta/llama-3.3",
        "google/gemini-pro",
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # Allowed models should be present
    assert "anthropic/open_router/anthropic/claude-opus" in ids
    assert "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus" in ids
    assert "anthropic/open_router/meta/llama-3.3" in ids
    assert "claude-3-freecc-no-thinking/open_router/meta/llama-3.3" in ids

    # Disallowed model should be absent
    assert "anthropic/open_router/google/gemini-pro" not in ids
    assert "claude-3-freecc-no-thinking/open_router/google/gemini-pro" not in ids


def test_gateway_models_list_falls_back_to_global_allowlist():
    """Providers with no per-provider entry fall back to global model_allowlist."""
    settings = _settings(
        model_opus=None,
        model_allowlist=["open_router/anthropic/claude-opus"],
    )
    app = create_test_app(settings)
    _cache_models(app, "open_router", "anthropic/claude-opus", "meta/llama-3.3")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # Only globally allowed model should be present
    assert "anthropic/open_router/anthropic/claude-opus" in ids
    assert "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus" in ids

    # Model not in global allowlist should be absent
    assert "anthropic/open_router/meta/llama-3.3" not in ids
    assert "claude-3-freecc-no-thinking/open_router/meta/llama-3.3" not in ids


def test_gateway_models_list_empty_allowlists_shows_all():
    """Empty allowlists => all models still listed (unchanged behavior)."""
    settings = _settings(model_opus=None)
    app = create_test_app(settings)
    _cache_models(app, "open_router", "anthropic/claude-opus", "meta/llama-3.3")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # All models should be present
    assert "anthropic/open_router/anthropic/claude-opus" in ids
    assert "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus" in ids
    assert "anthropic/open_router/meta/llama-3.3" in ids
    assert "claude-3-freecc-no-thinking/open_router/meta/llama-3.3" in ids


def test_gateway_models_list_supported_claude_models_always_present():
    """SUPPORTED_CLAUDE_MODELS still always present even with a restrictive allowlist."""
    settings = _settings(
        model_opus=None,
        model_allowlist=["open_router/anthropic/claude-opus"],
    )
    app = create_test_app(settings)

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # Claude compatibility models should always be present
    assert "claude-opus-4-20250514" in ids
    assert "claude-sonnet-4-20250514" in ids
    assert "claude-haiku-4-20250514" in ids
    assert "claude-3-opus-20240229" in ids


def test_gateway_models_list_configured_refs_respect_allowlist():
    """Configured model refs also respect allowlist filtering."""
    settings = _settings(
        model="nvidia_nim/nvidia/nemotron-3-super",
        model_opus=None,
        provider_model_allowlists={
            "nvidia_nim": [
                "nvidia/other-model"
            ],  # nemotron-3-super is NOT in the allowlist
        },
    )
    app = create_test_app(settings)

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # Configured model should be filtered out because it's not in the allowlist
    assert "anthropic/nvidia_nim/nvidia/nemotron-3-super" not in ids
    assert "claude-3-freecc-no-thinking/nvidia_nim/nvidia/nemotron-3-super" not in ids


def test_gateway_models_list_mixed_allowlist_scenarios():
    """Test complex scenario with both per-provider and global allowlists."""
    settings = _settings(
        model="deepseek/deepseek-chat",
        model_opus=None,
        model_allowlist=["deepseek/deepseek-chat"],
        provider_model_allowlists={
            "open_router": ["anthropic/claude-opus"],
        },
    )
    app = create_test_app(settings)
    _cache_models(app, "open_router", "anthropic/claude-opus", "meta/llama-3.3")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # deepseek model should be present (in global allowlist)
    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "claude-3-freecc-no-thinking/deepseek/deepseek-chat" in ids

    # open_router: claude-opus should be present (in per-provider allowlist)
    assert "anthropic/open_router/anthropic/claude-opus" in ids
    assert "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus" in ids

    # open_router: llama-3.3 should be absent (not in per-provider allowlist, and open_router has per-provider entry)
    assert "anthropic/open_router/meta/llama-3.3" not in ids
    assert "claude-3-freecc-no-thinking/open_router/meta/llama-3.3" not in ids

    # Claude compatibility models should always be present
    assert "claude-opus-4-20250514" in ids


def test_gateway_models_list_disabled_via_admin_api(monkeypatch, tmp_path):
    """Disabling a provider via the admin API removes its models from /v1/models."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    settings = _settings(
        model="nvidia_nim/nvidia/nemotron-3-super",
        model_opus=None,
        nvidia_nim_api_key="test-key",
    )
    app = create_test_app(settings)
    _cache_models(
        app,
        "nvidia_nim",
        "nvidia/nemotron-3-super",
        "nvidia/other-model",
    )

    client = TestClient(app, client=("127.0.0.1", 50000))

    # Verify nvidia_nim models are present before disabling
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = _get_model_ids(response)
    assert "anthropic/nvidia_nim/nvidia/nemotron-3-super" in ids
    assert "anthropic/nvidia_nim/nvidia/other-model" in ids

    # Disable nvidia_nim via admin API
    response = client.post(
        "/admin/api/providers/nvidia_nim/allowlist",
        json={"models": [], "restricted": True},
    )
    assert response.status_code == 200
    assert response.json()["restricted"] is True

    # The admin API persists the allowlist to the managed env file.
    from free_claude_code.config.admin.persistence import load_provider_allowlists

    allowlists = load_provider_allowlists()
    assert allowlists.get("nvidia_nim") == []

    # A restarted app that reloads settings with the updated allowlist
    # must exclude the disabled provider's models from /v1/models.
    restarted_settings = _settings(
        model="nvidia_nim/nvidia/nemotron-3-super",
        model_opus=None,
        nvidia_nim_api_key="test-key",
        provider_model_allowlists={"nvidia_nim": []},
    )
    restarted_app = create_test_app(restarted_settings)
    _cache_models(
        restarted_app,
        "nvidia_nim",
        "nvidia/nemotron-3-super",
        "nvidia/other-model",
    )

    response = TestClient(restarted_app).get("/v1/models")
    assert response.status_code == 200
    ids = _get_model_ids(response)
    assert "anthropic/nvidia_nim/nvidia/nemotron-3-super" not in ids
    assert "anthropic/nvidia_nim/nvidia/other-model" not in ids
    assert "claude-opus-4-20250514" in ids


def test_gateway_models_list_disabled_provider_shows_no_models():
    """A provider restricted to zero models exposes no models via /v1/models."""
    settings = _settings(
        model="nvidia_nim/nvidia/nemotron-3-super",
        model_opus=None,
        provider_model_allowlists={
            "nvidia_nim": [],  # restricted to nothing => provider disabled
        },
    )
    app = create_test_app(settings)
    _cache_models(
        app,
        "nvidia_nim",
        "nvidia/nemotron-3-super",
        "nvidia/other-model",
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = _get_model_ids(response)

    # No nvidia_nim variants should appear (provider disabled)
    assert "anthropic/nvidia_nim/nvidia/nemotron-3-super" not in ids
    assert "claude-3-freecc-no-thinking/nvidia_nim/nvidia/nemotron-3-super" not in ids
    assert "anthropic/nvidia_nim/nvidia/other-model" not in ids
    assert "claude-3-freecc-no-thinking/nvidia_nim/nvidia/other-model" not in ids

    # Claude compatibility models remain present
    assert "claude-opus-4-20250514" in ids

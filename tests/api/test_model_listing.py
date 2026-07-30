from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    clear_picker_aliases,
    seed_picker_aliases,
)
from tests.api.support import create_test_app, provider_manager_for_app


def _settings(
    *,
    model: str = "deepseek/deepseek-chat",
    model_fable: str | None = None,
    model_opus: str | None = "open_router/anthropic/claude-opus",
    model_haiku: str | None = "deepseek/deepseek-chat",
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
    )


def _cache_models(app, provider_id: str, *model_ids: str) -> None:
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {ProviderModelInfo(model_id) for model_id in model_ids},
    )


def test_models_list_includes_configured_refs_cached_provider_models_and_aliases():
    app = create_test_app(_settings())
    _cache_models(app, "deepseek", "deepseek-chat")
    _cache_models(
        app,
        "open_router",
        "meta/llama-3.3",
        "anthropic/claude-opus",
    )

    # Seed picker aliases from cached models (as ApplicationRuntime.start() would do)
    clear_picker_aliases()

    manager = provider_manager_for_app(app)
    refs = [info.model_id for info in manager.cached_prefixed_model_infos()]
    seed_picker_aliases(refs)

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    ids = [item["id"] for item in data["data"]]

    # Sorted refs: deepseek/deepseek-chat (0001), open_router/anthropic/claude-opus (0002), open_router/meta/llama-3.3 (0003)
    assert any(id_.startswith("claude-sonnet-nim-0001") for id_ in ids)
    assert any(id_.startswith("claude-sonnet-nim-0002") for id_ in ids)
    assert any(id_.startswith("claude-sonnet-nim-0003") for id_ in ids)

    # Each ref gets both thinking and no-thinking picker aliases
    import re

    deepseek_ids = [
        id_ for id_ in ids if re.match(r"claude-sonnet-nim-0001(-no-thinking)?$", id_)
    ]
    opus_ids = [
        id_ for id_ in ids if re.match(r"claude-sonnet-nim-0002(-no-thinking)?$", id_)
    ]
    llama_ids = [
        id_ for id_ in ids if re.match(r"claude-sonnet-nim-0003(-no-thinking)?$", id_)
    ]

    assert len(deepseek_ids) == 2  # thinking + no-thinking
    assert len(opus_ids) == 2
    assert len(llama_ids) == 2

    display_names = {item["id"]: item["display_name"] for item in data["data"]}

    # Both variants have display_name = provider_model_ref (no-thinking encoded in ID)
    for id_ in deepseek_ids:
        assert display_names[id_] == "deepseek/deepseek-chat"
    for id_ in opus_ids:
        assert display_names[id_] == "open_router/anthropic/claude-opus"
    for id_ in llama_ids:
        assert display_names[id_] == "open_router/meta/llama-3.3"

    assert "claude-sonnet-4-20250514" in ids
    assert "claude-fable-5" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]
    assert data["has_more"] is False


def test_models_list_uses_thinking_metadata_for_cached_models():
    app = create_test_app(_settings(model_opus=None))
    manager = provider_manager_for_app(app)
    _cache_models(app, "deepseek", "deepseek-chat")
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/reasoning-model" in ids
    assert "claude-3-freecc-no-thinking/open_router/reasoning-model" in ids
    assert "anthropic/open_router/plain-model" not in ids
    assert "claude-3-freecc-no-thinking/open_router/plain-model" in ids


def test_models_list_uses_cached_metadata_for_configured_refs():
    app = create_test_app(
        _settings(
            model="open_router/plain-model",
            model_opus=None,
            model_haiku=None,
        )
    )
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {ProviderModelInfo("plain-model", supports_thinking=False)},
    )

    response = TestClient(app).get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/plain-model" not in ids
    assert ids[0] == "claude-3-freecc-no-thinking/open_router/plain-model"


def test_models_list_includes_cached_wafer_models():
    app = create_test_app(
        _settings(
            model="wafer/DeepSeek-V4-Pro",
            model_opus=None,
            model_haiku=None,
        )
    )
    _cache_models(app, "wafer", "DeepSeek-V4-Pro", "MiniMax-M2.7")

    response = TestClient(app).get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/wafer/DeepSeek-V4-Pro" in ids
    assert "claude-3-freecc-no-thinking/wafer/DeepSeek-V4-Pro" in ids
    assert "anthropic/wafer/MiniMax-M2.7" in ids
    assert "claude-3-freecc-no-thinking/wafer/MiniMax-M2.7" in ids


def test_models_list_works_with_empty_discovery_catalog():
    app = create_test_app(_settings())

    # Clear picker aliases to test unseeded fallback behavior
    clear_picker_aliases()

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids[:4] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
    ]
    assert "claude-sonnet-4-20250514" in ids

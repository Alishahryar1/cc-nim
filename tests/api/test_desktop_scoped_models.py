"""End-to-end scoping of picker aliases to the desktop-only API mount."""

from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core import gateway_model_ids as gmi
from tests.api.support import create_test_app


def _settings() -> Settings:
    settings = Settings()
    settings.model = "deepseek/deepseek-chat"
    settings.model_fable = None
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    return settings


def setup_function(_: object) -> None:
    gmi.seed_picker_aliases(["deepseek/deepseek-chat"])


def teardown_function(_: object) -> None:
    gmi.clear_picker_aliases()


def test_bare_models_route_never_serves_aliases() -> None:
    app = create_test_app(_settings())

    ids = [m["id"] for m in TestClient(app).get("/v1/models").json()["data"]]

    assert all(not i.startswith("claude-sonnet-nim") for i in ids)
    assert "anthropic/deepseek/deepseek-chat" in ids


def test_prefixed_models_route_serves_aliases() -> None:
    app = create_test_app(_settings())

    response = TestClient(app).get("/claude-desktop/v1/models")

    assert response.status_code == 200
    data = response.json()
    ids = [m["id"] for m in data["data"]]
    # The alias replaces the wrapper id for the same ref.
    assert "claude-sonnet-nim-0001" in ids
    assert "anthropic/deepseek/deepseek-chat" not in ids
    labels = {
        m["display_name"]
        for m in data["data"]
        if m["id"].startswith("claude-sonnet-nim")
    }
    assert "deepseek/deepseek-chat" in labels


def test_custom_prefix_is_honored_from_settings() -> None:
    app = create_test_app(
        Settings.model_construct(
            desktop_gateway_prefix="my-gateway",
            model="deepseek/deepseek-chat",
            model_fable=None,
            model_opus=None,
            model_sonnet=None,
            model_haiku=None,
        )
    )

    bare = TestClient(app).get("/v1/models")
    prefixed = TestClient(app).get("/my-gateway/v1/models")

    assert all(not m["id"].startswith("claude-sonnet-nim") for m in bare.json()["data"])
    assert any(m["id"].startswith("claude-sonnet-nim") for m in prefixed.json()["data"])


def test_messages_round_trip_works_under_desktop_prefix() -> None:
    """Claude Desktop chats against the prefixed base URL, so the full API
    surface must exist under the prefix too."""
    from unittest.mock import patch

    app = create_test_app()

    with (
        patch(
            "free_claude_code.api.routes.get_token_count",
            return_value=1,
        ),
    ):
        response = TestClient(app).post(
            "/claude-desktop/v1/messages/count_tokens",
            json={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200


def test_advertised_alias_routes_through_prefixed_api() -> None:
    """An alias served by the desktop catalog must be routable end to end."""
    from unittest.mock import patch

    from free_claude_code.application.routing import (
        ModelRouter,
        RoutedTokenCountRequest,
    )
    from free_claude_code.core.anthropic import TokenCountRequest

    app = create_test_app(_settings())
    routed_refs: list[str] = []
    original_resolver = ModelRouter.resolve_token_count_request

    def spy(
        self: ModelRouter, request: TokenCountRequest
    ) -> RoutedTokenCountRequest:
        routed = original_resolver(self, request)
        routed_refs.append(routed.resolved.primary.provider_model_ref)
        return routed

    with (
        patch("free_claude_code.api.routes.get_token_count", return_value=1),
        patch.object(
            ModelRouter,
            "resolve_token_count_request",
            autospec=True,
            side_effect=spy,
        ),
    ):
        response = TestClient(app).post(
            "/claude-desktop/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-nim-0001",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    # The advertised alias reached the router and resolved to its seeded ref.
    assert routed_refs == ["deepseek/deepseek-chat"]

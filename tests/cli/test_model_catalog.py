from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.model_catalog import (
    ClientModel,
    catalog_wire_slug_for_ref,
    client_models_from_response,
    fetch_proxy_models_response,
)
from free_claude_code.core.json_types import JsonObject


def test_client_models_project_nested_direct_refs_in_source_order() -> None:
    assert client_models_from_response(
        {
            "data": [
                {
                    "id": "nvidia_nim/nvidia/nemotron-3-super",
                    "provider_model_ref": "nvidia_nim/nvidia/nemotron-3-super",
                    "display_name": "Display 0",
                },
                {
                    "id": "open_router/meta-llama/llama-3.3-70b",
                    "provider_model_ref": "open_router/meta-llama/llama-3.3-70b",
                    "display_name": "Display 1",
                },
            ]
        }
    ) == (
        ClientModel(
            wire_slug="nvidia_nim/nvidia/nemotron-3-super",
            provider_model_ref="nvidia_nim/nvidia/nemotron-3-super",
            display_name="Display 0",
            allows_reasoning=True,
        ),
        ClientModel(
            wire_slug="open_router/meta-llama/llama-3.3-70b",
            provider_model_ref="open_router/meta-llama/llama-3.3-70b",
            display_name="Display 1",
            allows_reasoning=True,
        ),
    )


def test_client_models_keep_no_thinking_direct_route() -> None:
    models = client_models_from_response(
        {
            "data": [
                {
                    "id": "claude-3-freecc-no-thinking/nvidia_nim/provider-model",
                    "provider_model_ref": "nvidia_nim/provider-model",
                    "display_name": "Display 0",
                }
            ]
        }
    )

    assert models == (
        ClientModel(
            wire_slug="claude-3-freecc-no-thinking/nvidia_nim/provider-model",
            provider_model_ref="nvidia_nim/provider-model",
            display_name="Display 0",
            allows_reasoning=False,
        ),
    )


def test_client_models_keep_no_thinking_only_route() -> None:
    assert client_models_from_response(
        {
            "data": [
                {
                    "id": "claude-3-freecc-no-thinking/open_router/plain-model",
                    "provider_model_ref": "open_router/plain-model",
                    "display_name": "Display 0",
                }
            ]
        }
    ) == (
        ClientModel(
            wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
            provider_model_ref="open_router/plain-model",
            display_name="Display 0",
            allows_reasoning=False,
        ),
    )


def test_client_models_ignore_compatibility_unknown_and_malformed_entries() -> None:
    payload: JsonObject = {
        "data": [
            {"id": "claude-opus-4-20250514"},
            {"id": "unknown/model", "provider_model_ref": 123},
            {"id": "   ", "provider_model_ref": "open_router/model"},
            {"id": "open_router/model", "provider_model_ref": "open_router/"},
            {"id": 123},
            "not-an-object",
        ]
    }

    assert client_models_from_response(payload) == ()
    assert client_models_from_response({"data": "not-a-list"}) == ()


def test_client_models_deduplicate_wire_slugs_deterministically() -> None:
    models = client_models_from_response(
        {
            "data": [
                {
                    "id": "gemini/models/gemini-test",
                    "provider_model_ref": "gemini/models/gemini-test",
                    "display_name": "Display 0",
                },
                {
                    "id": "gemini/models/gemini-test",
                    "provider_model_ref": "gemini/models/gemini-test",
                    "display_name": "Display 1",
                },
                {
                    "id": "open_router/provider/test",
                    "provider_model_ref": "open_router/provider/test",
                    "display_name": "Display 2",
                },
            ]
        }
    )

    assert [model.wire_slug for model in models] == [
        "gemini/models/gemini-test",
        "open_router/provider/test",
    ]
    assert models[0].display_name == "Display 0"


class _ModelsResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _ModelsResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info

    def read(self) -> bytes:
        return self._body


def test_fetch_proxy_models_uses_canonical_bearer_request() -> None:
    with patch(
        "free_claude_code.cli.launchers.model_catalog.open_local_request",
        return_value=_ModelsResponse(b'{"data": []}'),
    ) as open_local_request:
        response = fetch_proxy_models_response("http://127.0.0.1:9191/", "proxy-token")

    assert response == {"data": []}
    request = open_local_request.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:9191/v1/models?view=responses"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer proxy-token"


def test_fetch_proxy_models_can_request_messages_view() -> None:
    with patch(
        "free_claude_code.cli.launchers.model_catalog.open_local_request",
        return_value=_ModelsResponse(b'{"data": []}'),
    ) as open_local_request:
        response = fetch_proxy_models_response(
            "http://127.0.0.1:9191/",
            "proxy-token",
            view="messages",
        )

    assert response == {"data": []}
    request = open_local_request.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:9191/v1/models?view=messages"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer proxy-token"


def test_fetch_proxy_models_rejects_non_object_json() -> None:
    with (
        patch(
            "free_claude_code.cli.launchers.model_catalog.open_local_request",
            return_value=_ModelsResponse(b"[]"),
        ),
        pytest.raises(ValueError, match="JSON object"),
    ):
        fetch_proxy_models_response("http://127.0.0.1:9191", "proxy-token")


def _client_model(wire_slug: str, provider_model_ref: str) -> ClientModel:
    return ClientModel(
        wire_slug=wire_slug,
        provider_model_ref=provider_model_ref,
        display_name=provider_model_ref,
        allows_reasoning=wire_slug == provider_model_ref,
    )


def test_catalog_wire_slug_prefers_the_advertised_no_thinking_slug() -> None:
    models = (
        _client_model(
            "claude-3-freecc-no-thinking/open_router/vendor/chat-model",
            "open_router/vendor/chat-model",
        ),
    )

    assert (
        catalog_wire_slug_for_ref(models, "open_router/vendor/chat-model")
        == "claude-3-freecc-no-thinking/open_router/vendor/chat-model"
    )


def test_catalog_wire_slug_keeps_a_directly_advertised_ref() -> None:
    models = (
        _client_model("open_router/vendor/chat-model", "open_router/vendor/chat-model"),
    )

    assert (
        catalog_wire_slug_for_ref(models, "open_router/vendor/chat-model")
        == "open_router/vendor/chat-model"
    )


def test_catalog_wire_slug_falls_back_when_the_catalog_omits_the_ref() -> None:
    models = (_client_model("open_router/vendor/other", "open_router/vendor/other"),)

    assert (
        catalog_wire_slug_for_ref(models, "open_router/vendor/chat-model")
        == "open_router/vendor/chat-model"
    )
    assert catalog_wire_slug_for_ref((), "open_router/vendor/chat-model") == (
        "open_router/vendor/chat-model"
    )


def test_catalog_wire_slug_passes_through_an_unset_model() -> None:
    assert catalog_wire_slug_for_ref((), None) is None
    assert catalog_wire_slug_for_ref((), "") == ""

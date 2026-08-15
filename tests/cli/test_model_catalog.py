from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers.model_catalog import (
    ClientModel,
    client_models_from_response,
    fetch_proxy_models_response,
)
from free_claude_code.core.json_types import JsonObject


def _models_payload(*model_ids: str) -> JsonObject:
    return {
        "data": [
            {
                "id": model_id,
                "display_name": f"Display {index}",
            }
            for index, model_id in enumerate(model_ids)
        ]
    }


def test_client_models_project_nested_direct_refs_in_source_order() -> None:
    assert client_models_from_response(
        _models_payload(
            "anthropic/nvidia_nim/nvidia/nemotron-3-super",
            "open_router/meta-llama/llama-3.3-70b",
        )
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


def test_client_models_suppress_no_thinking_alias_when_normal_model_exists() -> None:
    models = client_models_from_response(
        _models_payload(
            "claude-3-freecc-no-thinking/nvidia_nim/provider-model",
            "anthropic/nvidia_nim/provider-model",
        )
    )

    assert [model.wire_slug for model in models] == ["nvidia_nim/provider-model"]


def test_client_models_keep_no_thinking_only_route() -> None:
    assert client_models_from_response(
        _models_payload("claude-3-freecc-no-thinking/open_router/plain-model")
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
            {"id": "anthropic/unknown/model"},
            {"id": 123},
            "not-an-object",
        ]
    }

    assert client_models_from_response(payload) == ()
    assert client_models_from_response({"data": "not-a-list"}) == ()


def test_client_models_deduplicate_wire_slugs_deterministically() -> None:
    models = client_models_from_response(
        _models_payload(
            "anthropic/gemini/models/gemini-test",
            "gemini/models/gemini-test",
            "anthropic/open_router/provider/test",
        )
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
    assert request.full_url == "http://127.0.0.1:9191/v1/models"
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

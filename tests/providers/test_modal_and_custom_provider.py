import json
import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.providers.runtime.factory import create_provider


def test_modal_provider_creation_and_headers():
    settings = Settings(
        MODAL_PROXY_TOKEN_ID="wk-test-id",
        MODAL_PROXY_TOKEN_SECRET="test-secret",
        MODAL_BASE_URL="https://modal-endpoint.example.com/v1",
    )
    provider = create_provider("modal", settings)
    assert provider._base_url == "https://modal-endpoint.example.com/v1"
    headers = provider._client.default_headers
    assert headers["Modal-Key"] == "wk-test-id"
    assert headers["Modal-Secret"] == "test-secret"


def test_custom_provider_creation_and_headers():
    custom_headers = json.dumps({"Custom-Header": "Value123"})
    settings = Settings(
        CUSTOM_API_KEY="custom-key-xyz",
        CUSTOM_BASE_URL="https://custom-endpoint.example.com/v1",
        CUSTOM_HEADERS_JSON=custom_headers,
        MODAL_PROXY_TOKEN_ID="wk-test-id",
        MODAL_PROXY_TOKEN_SECRET="test-secret",
    )
    provider = create_provider("custom", settings)
    assert provider._base_url == "https://custom-endpoint.example.com/v1"
    headers = provider._client.default_headers
    assert headers["Custom-Header"] == "Value123"
    assert headers["Modal-Key"] == "wk-test-id"
    assert headers["Modal-Secret"] == "test-secret"

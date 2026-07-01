"""Tests for the Azure AI Foundry (OpenAI-compatible v1) provider."""

from unittest.mock import patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from providers.azure_foundry import AzureFoundryProvider
from providers.base import ProviderConfig
from providers.exceptions import InvalidRequestError

_BASE_URL = "https://example-resource.services.ai.azure.com/openai/v1"


def _config(base_url: str | None = _BASE_URL) -> ProviderConfig:
    return ProviderConfig(
        api_key="test_azure_key",
        base_url=base_url,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture
def azure_provider():
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        yield AzureFoundryProvider(_config(), max_tokens=8192)


def test_init_sets_base_url_and_key():
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_client:
        provider = AzureFoundryProvider(_config(), max_tokens=8192)
    assert provider._api_key == "test_azure_key"
    assert provider._base_url == _BASE_URL
    assert mock_client.called


def test_missing_base_url_raises_invalid_request():
    with pytest.raises(InvalidRequestError, match="AZURE_FOUNDRY_BASE_URL is not set"):
        AzureFoundryProvider(_config(base_url=None))


def test_build_request_body_openai_chat(azure_provider):
    request = MessagesRequest(
        model="Kimi-K2.6",
        max_tokens=50,
        messages=[Message(role="user", content="hi")],
    )
    body = azure_provider._build_request_body(request)
    assert body["model"] == "Kimi-K2.6"
    assert body["max_tokens"] == 50
    assert body["messages"][0]["role"] == "user"


def test_prepare_create_body_clamps_over_budget(azure_provider):
    clamped = azure_provider._prepare_create_body({"max_tokens": 64000})
    assert clamped["max_tokens"] == 8192


def test_prepare_create_body_passes_through_under_budget(azure_provider):
    body = {"max_tokens": 2000}
    result = azure_provider._prepare_create_body(body)
    assert result["max_tokens"] == 2000
    # Must not mutate the caller's body.
    assert body["max_tokens"] == 2000


def test_prepare_create_body_clamps_max_completion_tokens(azure_provider):
    clamped = azure_provider._prepare_create_body({"max_completion_tokens": 64000})
    assert clamped["max_completion_tokens"] == 8192


def test_no_clamp_when_cap_unset():
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = AzureFoundryProvider(_config(), max_tokens=None)
    result = provider._prepare_create_body({"max_tokens": 64000})
    assert result["max_tokens"] == 64000

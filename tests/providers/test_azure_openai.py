"""Tests for the Azure OpenAI Responses v1 provider."""

from free_claude_code.providers.azure_responses.provider import AzureResponsesProvider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
)

AZURE_OPENAI_BASE_URL = "https://example-resource.openai.azure.com/openai/v1/"


def test_init_uses_resource_v1_url_and_api_key() -> None:
    config = make_provider_config(
        api_key="azure-key",
        base_url=AZURE_OPENAI_BASE_URL,
    )
    provider = AzureResponsesProvider(
        config,
        admission=immediate_admission(),
    )
    assert provider._client_headers["api-key"] == "azure-key"
    assert str(provider._client.base_url) == AZURE_OPENAI_BASE_URL.rstrip("/") + "/"

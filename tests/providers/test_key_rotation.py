import pytest
from providers.openai_compat import OpenAIChatTransport
from providers.base import ProviderConfig
from config.settings import Settings

class MockOpenAIProvider(OpenAIChatTransport):
    def _build_request_body(self, request, thinking_enabled=None):
        return {"model": "test", "messages": []}

def test_key_rotation_logic():
    config = ProviderConfig(
        api_key="key1",
        api_keys=["key1", "key2", "key3"]
    )

    provider = MockOpenAIProvider(
        config,
        provider_name="test",
        base_url="http://localhost",
        api_key="key1"
    )

    assert provider._get_next_api_key() == "key1"
    assert provider._get_next_api_key() == "key2"
    assert provider._get_next_api_key() == "key3"
    assert provider._get_next_api_key() == "key1"

def test_single_key_logic():
    config = ProviderConfig(
        api_key="key1",
        api_keys=["key1"]
    )

    provider = MockOpenAIProvider(
        config,
        provider_name="test",
        base_url="http://localhost",
        api_key="key1"
    )

    assert provider._get_next_api_key() == "key1"
    assert provider._get_next_api_key() == "key1"

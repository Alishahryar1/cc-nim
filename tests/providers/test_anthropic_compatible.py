"""Tests for generic Anthropic-compatible provider."""

from unittest.mock import MagicMock

import pytest

from free_claude_code.config.provider_catalog import ANTHROPIC_COMPATIBLE_DEFAULT_BASE
from free_claude_code.providers.anthropic_compatible import (
    AnthropicCompatibleProvider,
)
from free_claude_code.providers.base import ProviderConfig


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "custom-anthropic-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        self.extra_body = kwargs.get("extra_body")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def ant_config():
    return ProviderConfig(
        api_key="test_ant_key",
        base_url="http://custom-anthropic-server:8000/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def ant_provider(ant_config):
    return AnthropicCompatibleProvider(ant_config)


def test_init(ant_config):
    """Test provider initialization with custom base URL."""
    provider = AnthropicCompatibleProvider(ant_config)
    assert provider._api_key == "test_ant_key"
    assert provider._base_url == "http://custom-anthropic-server:8000/v1"


def test_default_base_url_constant():
    assert ANTHROPIC_COMPATIBLE_DEFAULT_BASE == "https://api.anthropic.com/v1"


def test_build_request_body_basic(ant_provider):
    """Basic native messages request body conversion."""
    req = MockRequest()
    body = ant_provider._build_request_body(req)

    assert body["model"] == "custom-anthropic-model"
    assert body["system"] == "System prompt"
    assert body["messages"][0]["role"] == "user"

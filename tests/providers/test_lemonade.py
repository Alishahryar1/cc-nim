"""Tests for Lemonade OpenAI Chat provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.lemonade import LEMONADE_DEFAULT_BASE, LemonadeProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.stream = True
        self.tools = []
        self.tool_choice = None
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none=True):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": self.system,
            "stream": self.stream,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "extra_body": self.extra_body,
            "thinking": {"enabled": self.thinking.enabled} if self.thinking else None,
        }


@pytest.fixture
def lemonade_config():
    return ProviderConfig(
        api_key="lemonade",
        base_url="http://localhost:13305",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch(
        "providers.transports.openai_chat.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def lemonade_provider(lemonade_config):
    return LemonadeProvider(lemonade_config)


def test_init(lemonade_config):
    """Test provider initialization."""
    with patch("openai.AsyncOpenAI"):
        provider = LemonadeProvider(lemonade_config)
        assert provider._base_url == "http://localhost:13305"
        assert provider._provider_name == "LEMONADE"


def test_init_uses_default_base_url():
    """Test that provider uses default base URL when not configured."""
    config = ProviderConfig(api_key="lemonade", base_url=None)
    with patch("openai.AsyncOpenAI"):
        provider = LemonadeProvider(config)
        assert provider._base_url == LEMONADE_DEFAULT_BASE


def test_init_uses_default_api_key():
    """Test that provider uses default API key when not configured."""
    config = ProviderConfig(
        base_url="http://localhost:13305",
        api_key="",
        rate_limit=10,
        rate_window=60,
    )
    with patch("openai.AsyncOpenAI"):
        provider = LemonadeProvider(config)
        assert provider._api_key == "lemonade"


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="lemonade",
        base_url="http://localhost:13305/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("openai.AsyncOpenAI"):
        provider = LemonadeProvider(config)
        assert provider._base_url == "http://localhost:13305"


def test_build_request_body(lemonade_provider):
    """Test that request body is built correctly."""
    req = MockRequest()
    body = lemonade_provider._build_request_body(req, thinking_enabled=False)
    assert body["model"] == "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M"
    assert "messages" in body

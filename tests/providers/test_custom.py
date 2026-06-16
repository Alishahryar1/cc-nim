"""Tests for the custom OpenAI-compatible provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.custom import CustomProvider
from providers.exceptions import AuthenticationError

_CUSTOM_BASE = "http://gateway.test/v1"


class MockMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs) -> None:
        self.model = "x/example-model:tag"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def custom_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test_custom_key",
        base_url=_CUSTOM_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def custom_provider(custom_config: ProviderConfig) -> CustomProvider:
    return CustomProvider(custom_config)


def test_init_uses_configured_base_url_and_key(custom_config: ProviderConfig) -> None:
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = CustomProvider(custom_config)
        assert provider._api_key == "test_custom_key"
        assert provider._base_url == _CUSTOM_BASE
        mock_openai.assert_called_once()


def test_init_rejects_missing_base_url() -> None:
    config = ProviderConfig(api_key="key", base_url="")
    with pytest.raises(AuthenticationError, match="CUSTOM_URL_PROVIDER"):
        CustomProvider(config)


def test_init_rejects_missing_api_key() -> None:
    config = ProviderConfig(api_key="", base_url=_CUSTOM_BASE)
    with pytest.raises(AuthenticationError, match="CUSTOM_API_KEY"):
        CustomProvider(config)


def test_build_request_body_basic(custom_provider: CustomProvider) -> None:
    req = MockRequest()
    body = custom_provider._build_request_body(req)

    assert body["model"] == "x/example-model:tag"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body


def test_build_request_body_preserves_slashes_in_model_name(
    custom_provider: CustomProvider,
) -> None:
    req = MockRequest(model="mannix/gemma4-98e-v7-coder:latest")
    body = custom_provider._build_request_body(req)

    assert body["model"] == "mannix/gemma4-98e-v7-coder:latest"

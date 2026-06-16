"""Tests for Custom OpenAI-compatible provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import ConfiguredChatModelRef
from providers.base import ProviderConfig
from providers.custom_openai import CustomOpenAIProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "gpt-4o"
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
def custom_openai_config():
    return ProviderConfig(
        api_key="test_custom_openai_key",
        base_url="http://localhost:3000/v1",
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

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
def custom_openai_provider(custom_openai_config):
    return CustomOpenAIProvider(custom_openai_config)


def test_init(custom_openai_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = CustomOpenAIProvider(custom_openai_config)
        assert provider._api_key == "test_custom_openai_key"
        assert provider._base_url == "http://localhost:3000/v1"
        assert provider._provider_name == "CUSTOM_OPENAI"
        mock_openai.assert_called_once()


def test_build_request_body_basic(custom_openai_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = MockRequest()
    body = custom_openai_provider._build_request_body(req)

    assert body["model"] == "gpt-4o"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body
    assert "max_tokens" not in body


def test_build_request_body_thinking_enabled(custom_openai_provider):
    req = MockRequest()
    req.thinking.enabled = True
    body = custom_openai_provider._build_request_body(req, thinking_enabled=True)
    assert body["model"] == "gpt-4o"


def test_build_request_body_thinking_disabled(custom_openai_provider):
    req = MockRequest()
    req.thinking.enabled = False
    body = custom_openai_provider._build_request_body(req, thinking_enabled=False)
    assert body["model"] == "gpt-4o"


def test_build_request_body_preserves_caller_extra_body(custom_openai_provider):
    req = MockRequest(extra_body={"test_option": True})
    body = custom_openai_provider._build_request_body(req)
    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("test_option") is True


@pytest.mark.asyncio
async def test_list_model_ids_success(custom_openai_provider):
    """Test successful model list retrieval from upstream models.list."""
    mock_model_list = MagicMock()
    # Mock return list of models
    mock_model_1 = MagicMock()
    mock_model_1.id = "custom-gpt-4o"
    mock_model_2 = MagicMock()
    mock_model_2.id = "custom-gpt-4o-mini"

    mock_model_list.data = [mock_model_1, mock_model_2]

    with patch.object(
        custom_openai_provider._client.models, "list", new_callable=AsyncMock
    ) as mock_list:
        mock_list.return_value = mock_model_list
        model_ids = await custom_openai_provider.list_model_ids()
        assert model_ids == frozenset({"custom-gpt-4o", "custom-gpt-4o-mini"})


@pytest.mark.asyncio
async def test_list_model_ids_failure_fallback_configured(custom_openai_provider):
    """Test fallback to configured models when upstream list fails."""
    mock_settings = MagicMock()
    mock_settings.configured_chat_model_refs.return_value = (
        ConfiguredChatModelRef(
            model_ref="custom_openai/my-model-1",
            provider_id="custom_openai",
            model_id="my-model-1",
            sources=("MODEL",),
        ),
        ConfiguredChatModelRef(
            model_ref="custom_openai/my-model-2",
            provider_id="custom_openai",
            model_id="my-model-2",
            sources=("MODEL_SONNET",),
        ),
        ConfiguredChatModelRef(
            model_ref="nvidia_nim/nim-model",
            provider_id="nvidia_nim",
            model_id="nim-model",
            sources=("MODEL_OPUS",),
        ),
    )

    with (
        patch.object(
            custom_openai_provider._client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ),
        patch("config.settings.get_settings", return_value=mock_settings),
    ):
        model_ids = await custom_openai_provider.list_model_ids()
        assert model_ids == frozenset({"my-model-1", "my-model-2"})


@pytest.mark.asyncio
async def test_list_model_ids_failure_fallback_default(custom_openai_provider):
    """Test fallback to default gpt-4o-mini when upstream fails and no configured custom_openai models."""
    mock_settings = MagicMock()
    mock_settings.configured_chat_model_refs.return_value = (
        ConfiguredChatModelRef(
            model_ref="nvidia_nim/nim-model",
            provider_id="nvidia_nim",
            model_id="nim-model",
            sources=("MODEL",),
        ),
    )

    with (
        patch.object(
            custom_openai_provider._client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ),
        patch("config.settings.get_settings", return_value=mock_settings),
    ):
        model_ids = await custom_openai_provider.list_model_ids()
        assert model_ids == frozenset({"auto", "gpt-4o-mini"})

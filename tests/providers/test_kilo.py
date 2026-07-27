"""Tests for the Kilo.ai OpenAI-chat provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import KILO_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def kilo_config():
    return ProviderConfig(
        api_key="test_kilo_key",
        base_url=KILO_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def kilo_provider(kilo_config):
    return profiled_provider(
        "kilo",
        kilo_config,
        admission=immediate_admission(),
    )


def test_default_base_url():
    assert KILO_DEFAULT_BASE == "https://api.kilo.ai/api/gateway"


def test_init_uses_openai_chat_provider(kilo_provider):
    assert isinstance(kilo_provider, OpenAIChatProvider)
    assert kilo_provider._api_key == "test_kilo_key"
    assert kilo_provider._base_url == KILO_DEFAULT_BASE
    assert kilo_provider._provider_name == "KILO"


def test_build_request_body_openai_shape(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.5",
            "messages": [Message(role="user", content="Hello")],
            "max_tokens": 100,
        }
    )

    body = kilo_provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == "anthropic/claude-sonnet-4.5"
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100


def test_build_request_body_forwards_caller_extra_body(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"custom_field": "value"},
        }
    )

    body = kilo_provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body.get("extra_body", {}).get("custom_field") == "value"


def test_build_request_body_sends_reasoning_object(kilo_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = kilo_provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body.get("extra_body", {}).get("reasoning") is not None


def test_build_request_body_sends_reasoning_disabled(kilo_provider):
    from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy

    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    body = kilo_provider._build_request_body(
        request,
        reasoning=ReasoningPolicy(control=ReasoningControl.OFF),
    )

    assert body["extra_body"]["reasoning"] == {"enabled": False}


@pytest.mark.asyncio
async def test_model_list_uses_openai_client_models_endpoint(kilo_provider):
    kilo_provider._client.models.list = AsyncMock(
        return_value=MagicMock(
            data=[
                MagicMock(id="anthropic/claude-sonnet-4.5"),
                MagicMock(id="openai/gpt-5.4"),
            ]
        )
    )

    assert await kilo_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("anthropic/claude-sonnet-4.5"),
            ProviderModelInfo("openai/gpt-5.4"),
        }
    )

    kilo_provider._client.models.list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(kilo_provider):
    kilo_provider._client = MagicMock()
    kilo_provider._client.close = AsyncMock()

    await kilo_provider.cleanup()

    kilo_provider._client.close.assert_awaited_once()

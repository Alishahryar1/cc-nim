"""Tests for the FPT AI Factory OpenAI-chat provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import FPT_AI_FACTORY_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def fpt_config():
    return ProviderConfig(
        api_key="test_fpt_key",
        base_url=FPT_AI_FACTORY_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def fpt_provider(fpt_config):
    return profiled_provider(
        "fpt_ai_factory",
        fpt_config,
        admission=immediate_admission(),
    )


def test_default_base_url():
    assert FPT_AI_FACTORY_DEFAULT_BASE == "https://mkp-api.fptcloud.com/v1"


def test_init_uses_openai_chat_provider(fpt_provider):
    assert isinstance(fpt_provider, OpenAIChatProvider)
    assert fpt_provider._api_key == "test_fpt_key"
    assert fpt_provider._base_url == FPT_AI_FACTORY_DEFAULT_BASE
    assert fpt_provider._provider_name == "FPT_AI_FACTORY"


def test_build_request_body_openai_shape(fpt_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-oss-120b",
            "messages": [Message(role="user", content="Hello")],
            "max_tokens": 100,
        }
    )

    body = fpt_provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == "gpt-oss-120b"
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100


def test_build_request_body_omits_reasoning_controls(fpt_provider):
    # FPT's OpenAI-compatible gateway exposes no reasoning control parameter
    # (supported_parameters carries none), so the profile must never emit one.
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-oss-120b",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = fpt_provider._build_request_body(request, reasoning=reasoning_for(request))

    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "reasoning" not in body.get("extra_body", {})


@pytest.mark.asyncio
async def test_model_list_uses_openai_client_models_endpoint(fpt_provider):
    fpt_provider._client.models.list = AsyncMock(
        return_value=MagicMock(
            data=[
                MagicMock(id="gpt-oss-120b"),
                MagicMock(id="DeepSeek-V4-Flash"),
            ]
        )
    )

    assert await fpt_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("gpt-oss-120b"),
            ProviderModelInfo("DeepSeek-V4-Flash"),
        }
    )

    fpt_provider._client.models.list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(fpt_provider):
    fpt_provider._client = MagicMock()
    fpt_provider._client.close = AsyncMock()

    await fpt_provider.cleanup()

    fpt_provider._client.close.assert_awaited_once()

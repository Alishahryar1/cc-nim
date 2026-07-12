"""Tests for the Z.ai OpenAI-chat Coding Plan provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import ZAI_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import passthrough_rate_limiter, profiled_provider


@pytest.fixture
def zai_provider():
    return profiled_provider(
        "zai",
        ProviderConfig(
            api_key="test_zai_key",
            base_url=ZAI_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=True,
        ),
        rate_limiter=passthrough_rate_limiter(),
    )


def test_init_uses_openai_chat_coding_endpoint(zai_provider):
    assert isinstance(zai_provider, OpenAIChatProvider)
    assert zai_provider._api_key == "test_zai_key"
    assert zai_provider._base_url == "https://api.z.ai/api/coding/paas/v4"


def test_build_request_body_openai_chat(zai_provider):
    request = MessagesRequest(
        model="glm-5.2",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
    )

    body = zai_provider._build_request_body(request)

    assert body["model"] == "glm-5.2"
    assert body["max_tokens"] == 100
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }


def test_build_request_body_default_max_tokens(zai_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )

    body = zai_provider._build_request_body(request)

    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_rejects_caller_extra_body(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "extra_body": {"x": 1},
        }
    )

    with pytest.raises(InvalidRequestError, match=r"Z\.ai Chat Completions"):
        zai_provider._build_request_body(request)


def test_build_request_body_disables_zai_thinking(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "disabled"},
        }
    )

    body = zai_provider._build_request_body(request)

    assert body["extra_body"]["thinking"] == {"type": "disabled"}


def test_build_request_body_replays_prior_reasoning_content(zai_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "glm-5.2",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "prior"}],
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )

    body = zai_provider._build_request_body(request)

    assert body["messages"][0]["reasoning_content"] == "prior"
    assert body["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(zai_provider):
    zai_provider._client = MagicMock()
    zai_provider._client.close = AsyncMock()

    await zai_provider.cleanup()

    zai_provider._client.close.assert_awaited_once()

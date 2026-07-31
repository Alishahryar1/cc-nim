"""Tests for the OrcaRouter OpenAI-chat provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import ORCAROUTER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest, Tool
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def orcarouter_config():
    return ProviderConfig(
        api_key="test-orcarouter-key",
        base_url=ORCAROUTER_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def orcarouter_provider(orcarouter_config):
    return profiled_provider(
        "orcarouter",
        orcarouter_config,
        admission=immediate_admission(),
    )


def test_default_base_url():
    assert ORCAROUTER_DEFAULT_BASE == "https://api.orcarouter.ai/v1"


def test_init_uses_openai_chat_provider(orcarouter_provider):
    assert isinstance(orcarouter_provider, OpenAIChatProvider)
    assert orcarouter_provider._api_key == "test-orcarouter-key"
    assert orcarouter_provider._base_url == ORCAROUTER_DEFAULT_BASE
    assert orcarouter_provider._provider_name == "ORCAROUTER"


def test_build_request_body_openai_shape_and_named_effort(orcarouter_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [Message(role="user", content="Hello")],
            "tools": [
                Tool(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }
    )

    body = orcarouter_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "anthropic/claude-sonnet-4.6"
    assert body["messages"][0] == {"role": "user", "content": "Hello"}
    assert body["tools"][0]["function"]["name"] == "echo"
    # The gateway rejects a top-level ``reasoning`` object, so effort travels in the
    # OpenAI-native ``reasoning_effort`` field. A budget-only request carries no named
    # effort, so the profile's enabled fallback applies.
    assert body["reasoning_effort"] == "medium"
    assert "reasoning" not in body


def test_build_request_body_omits_effort_when_reasoning_disabled(orcarouter_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
            "thinking": {"type": "disabled"},
        }
    )

    body = orcarouter_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    # Routed upstreams do not share one "reasoning off" token, so the field is left out
    # instead of sending a value some of them would reject.
    assert "reasoning_effort" not in body


def test_build_request_body_honors_effective_no_thinking(orcarouter_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "Explore the codebase."}],
        }
    )

    body = orcarouter_provider._build_request_body(request, reasoning=REASONING_OFF)

    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_lists_models_from_openai_models_endpoint(orcarouter_provider):
    orcarouter_provider._client.models.list = AsyncMock(
        return_value=MagicMock(
            data=[
                MagicMock(id="anthropic/claude-sonnet-4.6"),
                MagicMock(id="openai/gpt-5.5"),
            ]
        )
    )

    assert await orcarouter_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("anthropic/claude-sonnet-4.6"),
            ProviderModelInfo("openai/gpt-5.5"),
        }
    )

    orcarouter_provider._client.models.list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(orcarouter_provider):
    orcarouter_provider._client = MagicMock()
    orcarouter_provider._client.close = AsyncMock()

    await orcarouter_provider.cleanup()

    orcarouter_provider._client.close.assert_awaited_once()
